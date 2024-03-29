import pytest
 
from math import sqrt

import numpy as np

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax

from mpi4py import MPI
# Abusing mpi4jax by exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType
import mpi4jax

from py_swe.model import advance_model_n_steps
from py_swe.geometry import Vec2, RectangularGrid, create_geometry
from py_swe.state import State, create_local_field_zeros, create_local_field_unit_random, create_local_field_tsunami_height, pad_state, unpad_state
from py_swe.tlm import advance_tlm_n_steps
from py_swe.adm import advance_adm_n_steps

from .utils import linearization_checks as lc

mpi4jax_comm = MPI.COMM_WORLD
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()
mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)

### Parameters

origin = Vec2(0.0, 0.0)
extent = Vec2(100000.0, 100000.0)
nx = ny = 101
dx = dy = extent.x / (nx - 1.0)
g = 9.81
dt = 0.68 * dx / sqrt(g * 5030)
num_steps = 10
grid = RectangularGrid(nx, ny)
geometry_unpadded = create_geometry(rank, size, grid, 0, 0, origin, extent)
geometry_padded = create_geometry(rank, size, grid, 1, 1, origin, extent)
b = create_local_field_zeros(geometry_padded, jnp.float64)
rng = np.random.default_rng(12345)

### Functions

def primalArg():
    return State(create_local_field_zeros(geometry_unpadded, jnp.float64),
                 create_local_field_zeros(geometry_unpadded, jnp.float64),
                 create_local_field_tsunami_height(geometry_unpadded, jnp.float64),)

def tangentArg():
    return State(create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),
                 create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),
                 create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),)

def cotangentArg():
    return State(create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),
                 create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),
                 create_local_field_unit_random(geometry_unpadded, jnp.float64, rng=rng),)

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
    s_padded = pad_state(s, geometry_padded)

    s_padded = advance_model_n_steps(s_padded, geometry_padded, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy)

    return unpad_state(s_padded, geometry_padded)

def tlm(s, ds):
    s_padded = pad_state(s, geometry_padded)
    ds_padded = pad_state(ds, geometry_padded)

    s_padded, ds_padded = advance_tlm_n_steps(s_padded, ds_padded, geometry_padded, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy)

    s_new = unpad_state(s_padded, geometry_padded)
    ds_new = unpad_state(ds_padded, geometry_padded)

    return s_new, ds_new

def adm(s, Ds):
    s_padded = pad_state(s, geometry_padded)
    Ds_padded = pad_state(Ds, geometry_padded)

    s_padded, Ds_padded = advance_adm_n_steps(s_padded, Ds_padded, geometry_padded, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy)

    s_new = unpad_state(s_padded, geometry_padded)
    Ds_new = unpad_state(Ds_padded, geometry_padded)

    return s_new, Ds_new


def test_tlm_linearity():
    success, absolute_error = lc.testTLMLinearity(tlm, primalArg, tangentArg, scale, norm, 1.0e-15)
    if rank == 0:
        print(f"Test TLM Linearity: success ={success}, absolute error={absolute_error}")
    assert success

def test_tlm_approx():
    success, relative_error = lc.testTLMApprox(m, tlm, primalArg, tangentArg, scale, add, norm, 2.0e-7)
    if rank == 0:
        print(f"Test TLM Approx: success ={success}, relative error={relative_error}")
    assert success

def test_spectral_theorem():
    success, absolute_error = lc.testSpectralTheorem(tlm, adm, primalArg, tangentArg, cotangentArg, dot, 1.0e-15)
    if rank == 0:
        print(f"Test Spectral Theorem (\"Dot Product Test\"): success ={success}, absolute error={absolute_error}")
    assert success