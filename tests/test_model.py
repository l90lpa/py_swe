import pytest

from math import sqrt, floor, exp

import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI

from shallow_water.geometry import create_par_geometry, RectangularDomain, at_locally_owned
from shallow_water.state import create_local_field_zeros, gather_global_field
from shallow_water.model import advance_model_n_steps, calculate_max_wavespeed
from shallow_water.runtime_context import mpi4py_comm
from shallow_water.state import State
from shallow_water.scan_functions import py_scan

rank = mpi4py_comm.Get_rank()
size = mpi4py_comm.Get_size()
root = 0 

def test_model_1():

    xmax = ymax = 100000.0
    nx = ny = 101
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 1

    domain = RectangularDomain(nx, ny)
    geometry = create_par_geometry(rank, size, domain)
    zero_field = create_local_field_zeros(geometry, jnp.float64)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = jnp.copy(zero_field)
    s = State(u, v, h)
    b = jnp.copy(zero_field)
    max_wavespeed, _ = calculate_max_wavespeed(h, geometry)

    s_new = advance_model_n_steps(s, max_wavespeed, geometry, b, num_steps, dt, dx, dy, py_scan)

    u_locally_owned = np.array(s_new.u[at_locally_owned(geometry)])
    v_locally_owned = np.array(s_new.v[at_locally_owned(geometry)])
    h_locally_owned = np.array(s_new.h[at_locally_owned(geometry)])
    
    u_global = gather_global_field(u_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    v_global = gather_global_field(v_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    h_global = gather_global_field(h_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

    if rank == root:
        assert np.all(np.equal(u_global, [0.0]))
        assert np.all(np.equal(v_global, [0.0]))
        assert np.all(np.equal(h_global, [0.0]))

def test_model_2():

    xmax = ymax = 100000.0
    nx = ny = 101
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 1

    domain = RectangularDomain(nx, ny)
    geometry = create_par_geometry(rank, size, domain)
    zero_field = create_local_field_zeros(geometry, jnp.float64)
    
    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = jnp.copy(zero_field)
    h = h.at[at_locally_owned(geometry)].set(10.0)
    s = State(u, v, h)
    b = jnp.copy(zero_field)
    max_wavespeed, _ = calculate_max_wavespeed(h, geometry)

    s_new = advance_model_n_steps(s, max_wavespeed, geometry, b, num_steps, dt, dx, dy, py_scan)

    u_locally_owned = np.array(s_new.u[at_locally_owned(geometry)])
    v_locally_owned = np.array(s_new.v[at_locally_owned(geometry)])
    h_locally_owned = np.array(s_new.h[at_locally_owned(geometry)])
    
    u_global = gather_global_field(u_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    v_global = gather_global_field(v_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    h_global = gather_global_field(h_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

    if rank == root:
        assert np.all(np.equal(u_global, [0.0]))
        assert np.all(np.equal(v_global, [0.0]))
        assert np.all(np.equal(h_global, [10.0]))

def test_model_3():
    assert size == 4

    xmax = ymax = 10000.0
    nx = ny = 11
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 100

    domain = RectangularDomain(nx, ny)
    geometry = create_par_geometry(rank, size, domain)
    zero_field = create_local_field_zeros(geometry, jnp.float64)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = jnp.copy(zero_field)
    # Create tsunami pluse IC
    xmid = xmax / 2.0
    ymid = ymax / 2.0
    sigma = floor(xmax / 20.0)
    h_global = np.zeros((nx,ny), dtype=jnp.float64)
    for j in range(ny):
        for i in range(nx):
            dsqr = ((i) * dx - xmid)**2 + ((j) * dy - ymid)**2
            h_global[i,j] = (5000.0 + (5030.0 - 5000.0) * exp(-dsqr / sigma**2))
    if rank == 0:
        h = h.at[at_locally_owned(geometry)].set(h_global[0:5, 0:5])
    if rank == 1:
        h = h.at[at_locally_owned(geometry)].set(h_global[5:11, 0:5])
    if rank == 2:
        h = h.at[at_locally_owned(geometry)].set(h_global[0:5, 5:11])
    if rank == 3:
        h = h.at[at_locally_owned(geometry)].set(h_global[5:11, 5:11])
    s = State(u, v, h)

    b = jnp.copy(zero_field)
    max_wavespeed, _ = calculate_max_wavespeed(h, geometry)

    s_new = advance_model_n_steps(s, max_wavespeed, geometry, b, num_steps, dt, dx, dy, py_scan)


    u_locally_owned = np.array(s_new.u[at_locally_owned(geometry)])
    v_locally_owned = np.array(s_new.v[at_locally_owned(geometry)])
    h_locally_owned = np.array(s_new.h[at_locally_owned(geometry)])
    u_global = gather_global_field(u_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    v_global = gather_global_field(v_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    h_global = gather_global_field(h_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

    if rank == root:
        rms_u = sqrt(np.sum(u_global ** 2) / (nx * ny))
        rms_v = sqrt(np.sum(v_global ** 2) / (nx * ny))
        rms_h = sqrt(np.sum(h_global ** 2) / (nx * ny))

        print('rms_u = {}, rms_v = {}, rms_h = {}'.format(rank, rms_u, rms_v, rms_h))
        assert abs(rms_u - 0.0016110910527818616) == 0.0 #< 10.e-15
        assert abs(rms_v - 0.0016110910527825017) == 0.0 #< 10.e-15
        assert abs(rms_h - 5000.371968791809)     == 0.0 #< 10.e-15


# def m(u):
#     return solver_rk3(u, v, dx, num_points, dt, num_steps)

# def TLM(u, du):
#     return solver_rk3_tlm(u, du, v, dx, num_points, dt, num_steps)

# def ADM(u, Dv):
#     return solver_rk3_adm(u, Dv, v, dx, num_points, dt, num_steps)

# def alldot(a,b):
#     l_dot = np.dot(a, b)
#     dot, _ = mpi4jax.allreduce(l_dot, op=MPI.SUM, comm=comm)
#     return dot

# def allnorm(x):
#     norm = np.sqrt(alldot(x,x))
#     return norm

# def testTLMLinearity(TLM, tol):
#     N = 120 // size
#     if rank == 0:
#         rng = np.random.default_rng(12345)
#         u0 = rng.random((size, N), dtype=np.float64)
#         du = rng.random((size, N), dtype=np.float64)
#     else:
#         u0 = np.empty((N,))
#         du = np.empty((N,))
        
#     l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
#     l_du, _ = mpi4jax.scatter(du, 0, comm=comm)
#     l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))
#     l_dv2 = np.array(TLM(jnp.array(l_u0), jnp.array(2.0*l_du)))
    
#     absolute_error = allnorm(l_dv2 - 2.0*l_dv)
    
#     return absolute_error < tol, absolute_error

# def testTLMApprox(m, TLM, tol):
#     N = 120 // size
#     if rank == 0:
#         rng = np.random.default_rng(12345)
#         u0 = rng.random((size, N), dtype=np.float64)
#         du = rng.random((size, N), dtype=np.float64)
#     else:
#         u0 = np.empty((N,))
#         du = np.empty((N,))
        
#     l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
#     l_du, _ = mpi4jax.scatter(du, 0, comm=comm)
    
#     l_v0 = m(jnp.array(l_u0))
#     l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))
    
#     scale = 1.0

#     absolute_errors = []
#     relavite_errors = []
#     other = []
#     for i in range(15):
#         l_v1 = np.array(m(jnp.array(l_u0 + (scale * l_du))))
#         absolute_error = allnorm((scale * l_dv) - (l_v1 - l_v0))
#         absolute_errors.append(absolute_error)
#         other.append(allnorm(l_v1 - l_v0))
#         relative_error = absolute_error / other[-1]
        
#         relavite_errors.append(relative_error)
#         scale /= 10.0

#     # if rank == 0:
#     #     print(absolute_errors)
#     #     print(relavite_errors)
#     # mpi4jax.barrier(comm=comm)
#     min_relative_error = np.min(relavite_errors)

#     return min_relative_error < tol, min_relative_error

# def testADMApprox(TLM, ADM, tol):
#     N = 120 // size
#     rng = np.random.default_rng(12345)
#     if rank == 0:
#         u0 = rng.random((size, N), dtype=np.float64)
#         du = rng.random((size, N), dtype=np.float64)
#     else:
#         u0 = np.empty((N,))
#         du = np.empty((N,))
        
#     l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
#     l_du, _ = mpi4jax.scatter(du, 0, comm=comm)

#     l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))

#     M = jnp.size(l_dv)
#     if rank == 0:
#         Dv = rng.random((size, M), dtype=np.float64)
#     else:
#         Dv = np.empty((M,))
    
#     l_Dv, _ = mpi4jax.scatter(Dv, 0, comm=comm)
    
#     l_Du = np.array(ADM(jnp.array(l_u0), jnp.array(l_Dv))).flatten()
    
    
#     absolute_error = np.abs(alldot(l_dv, l_Dv) - alldot(l_du, l_Du))
#     return absolute_error < tol, absolute_error

# if rank == 0:
#     print("Test TLM Linearity:")
# success, absolute_error = testTLMLinearity(TLM, 1.0e-13)
# if rank == 0:
#     print("success = ", success, ", absolute_error = ", absolute_error)

# if rank == 0:
#     print("Test TLM Approximation:")
# success, relative_error = testTLMApprox(m, TLM, 1.0e-13)
# if rank == 0:
#     print("success = ", success, ", relative error = ", relative_error)

# if rank == 0:
#     print("Test ADM Approximation:")
# success, absolute_error = testADMApprox(TLM, ADM, 1.0e-13)
# if rank == 0:
#     print("success = ", success, ", absolute_error = ", absolute_error)




