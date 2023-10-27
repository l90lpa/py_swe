
from math import sqrt

import numpy as np

from jax import jvp, jit, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax

from mpi4py import MPI
# Abusing mpi4jax by exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType
import mpi4jax

from shallow_water.model import advance_model_n_steps
from shallow_water.geometry import RectangularDomain, create_domain_par_geometry, add_ghost_geometry, add_halo_geometry, at_local_domain
from shallow_water.state import State, create_local_field_zeros, create_local_field_unit_random, create_local_field_random, create_local_field_tsunami_height, create_local_field_ones
from shallow_water.tlm import advance_tlm_n_steps
from shallow_water.adm import advance_adm_n_steps

import validation.linearization_checks as lc

mpi4jax_comm = MPI.COMM_WORLD
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()


def create_padded_b(geometry, dtype):
    padded_geometry = add_halo_geometry(geometry, 1)
    padded_geometry = add_ghost_geometry(padded_geometry, 1)
    return create_local_field_zeros(padded_geometry, dtype)


if __name__ == "__main__":

    ### Parameters
    
    xmax = ymax = 100000
    nx = ny = 11
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    num_steps = 10
    domain = RectangularDomain(nx, ny)
    geometry = create_domain_par_geometry(rank, size, domain)
    b = create_padded_b(geometry, jnp.float64)
    rng = np.random.default_rng(12345)

    ### Functions

    def primalArg():
        return State(create_local_field_zeros(geometry, jnp.float64),
                     create_local_field_zeros(geometry, jnp.float64),
                     create_local_field_ones(geometry, jnp.float64),)

    def tangentArg():
        return State(create_local_field_unit_random(geometry, jnp.float64, rng=rng),
                     create_local_field_unit_random(geometry, jnp.float64, rng=rng),
                     create_local_field_unit_random(geometry, jnp.float64, rng=rng),)
    
    def cotangentArg():
        return State(create_local_field_unit_random(geometry, jnp.float64, rng=rng),
                     create_local_field_unit_random(geometry, jnp.float64, rng=rng),
                     create_local_field_unit_random(geometry, jnp.float64, rng=rng),)
    
    def scale(a,x):
        return State(a * x.u, a * x.v, a * x.h)
        
    def add(x,y):
        return State(x.u + y.u, x.v + y.v, x.h + y.h)
            
    def dot(x,y):
        dot_  = jnp.sum(x.u * y.u)
        dot_ += jnp.sum(x.v * y.v)
        dot_ += jnp.sum(x.h * y.h)
        dot_, _ = mpi4jax.allreduce(dot_, op=MPI.SUM, comm=mpi4jax_comm)
        return dot_.item()
    
    def norm(x):
        return jnp.sqrt(dot(x,x)).item()

    def m(s):
        padded_geometry = add_halo_geometry(geometry, 1)
        padded_geometry = add_ghost_geometry(padded_geometry, 1)

        zeros_field = create_local_field_zeros(padded_geometry, jnp.float64)

        u = zeros_field.at[at_local_domain(padded_geometry)].set(s.u)
        v = zeros_field.at[at_local_domain(padded_geometry)].set(s.v)
        h = zeros_field.at[at_local_domain(padded_geometry)].set(s.h)
        padded_state = State(u, v, h)

        padded_state_new = advance_model_n_steps(padded_state, padded_geometry, HashableMPIType(mpi4jax_comm), b, num_steps, dt, dx, dy)

        return State(padded_state_new.u[at_local_domain(padded_geometry)],
                     padded_state_new.v[at_local_domain(padded_geometry)],
                     padded_state_new.h[at_local_domain(padded_geometry)])
    
    def tlm(s, ds):
        padded_geometry = add_halo_geometry(geometry, 1)
        padded_geometry = add_ghost_geometry(padded_geometry, 1)

        zeros_field = create_local_field_zeros(padded_geometry, jnp.float64)

        u = zeros_field.at[at_local_domain(padded_geometry)].set(s.u)
        v = zeros_field.at[at_local_domain(padded_geometry)].set(s.v)
        h = zeros_field.at[at_local_domain(padded_geometry)].set(s.h)
        padded_state = State(u, v, h)

        du = zeros_field.at[at_local_domain(padded_geometry)].set(ds.u)
        dv = zeros_field.at[at_local_domain(padded_geometry)].set(ds.v)
        dh = zeros_field.at[at_local_domain(padded_geometry)].set(ds.h)
        padded_dstate = State(du, dv, dh)

        padded_state_new, padded_dstate_new = advance_tlm_n_steps(padded_state, padded_dstate, padded_geometry, HashableMPIType(mpi4jax_comm), b, num_steps, dt, dx, dy)

        return State(padded_state_new.u[at_local_domain(padded_geometry)],
                     padded_state_new.v[at_local_domain(padded_geometry)],
                     padded_state_new.h[at_local_domain(padded_geometry)]), State(padded_dstate_new.u[at_local_domain(padded_geometry)],
                     padded_dstate_new.v[at_local_domain(padded_geometry)],
                     padded_dstate_new.h[at_local_domain(padded_geometry)])
    
    def adm(s, Ds):
        padded_geometry = add_halo_geometry(geometry, 1)
        padded_geometry = add_ghost_geometry(padded_geometry, 1)

        zeros_field = create_local_field_zeros(padded_geometry, jnp.float64)

        u = zeros_field.at[at_local_domain(padded_geometry)].set(s.u)
        v = zeros_field.at[at_local_domain(padded_geometry)].set(s.v)
        h = zeros_field.at[at_local_domain(padded_geometry)].set(s.h)
        padded_state = State(u, v, h)

        Du = zeros_field.at[at_local_domain(padded_geometry)].set(Ds.u)
        Dv = zeros_field.at[at_local_domain(padded_geometry)].set(Ds.v)
        Dh = zeros_field.at[at_local_domain(padded_geometry)].set(Ds.h)
        padded_dstate = State(Du, Dv, Dh)

        padded_state_new, padded_dstate_new = advance_adm_n_steps(padded_state, padded_dstate, padded_geometry, HashableMPIType(mpi4jax_comm), b, num_steps, dt, dx, dy)

        return State(padded_state_new.u[at_local_domain(padded_geometry)],
                     padded_state_new.v[at_local_domain(padded_geometry)],
                     padded_state_new.h[at_local_domain(padded_geometry)]), State(padded_dstate_new.u[at_local_domain(padded_geometry)],
                     padded_dstate_new.v[at_local_domain(padded_geometry)],
                     padded_dstate_new.h[at_local_domain(padded_geometry)])

    ### Tests

    if rank == 0:
        print("Test TLM Linearity:")
    success, absolute_error = lc.testTLMLinearity(tlm, primalArg, tangentArg, scale, norm, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", absolute_error = ", absolute_error)

    if rank == 0:
        print("Test TLM Approximation:")
    success, relative_error = lc.testTLMApprox(m, tlm, primalArg, tangentArg, scale, add, norm, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", relative error = ", relative_error)

    if rank == 0:
        print("Test ADM Approximation:")
    success, relative_error = lc.testADMApprox(tlm, adm, primalArg, tangentArg, cotangentArg, dot, norm, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", relative error = ", relative_error)