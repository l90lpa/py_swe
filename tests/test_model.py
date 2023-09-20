import pytest

from math import sqrt, floor, exp

import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI

from shallow_water.geometry import create_par_geometry, RectangularDomain, get_locally_owned_range, at_locally_owned
from shallow_water.state import create_local_field_zeros, calculate_max_wavespeed, gather_global_field
from shallow_water.model import advance_model_n_steps
from shallow_water.visualize import visualize_locally_owned_field
from shallow_water.runtime_context import mpi4py_comm

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
    b = jnp.copy(zero_field)

    new_u, new_v, new_h = advance_model_n_steps(u, v, h, 1.0, geometry, b, num_steps, dt, dx, dy)

    u_locally_owned = np.array(new_u[at_locally_owned(geometry)])
    v_locally_owned = np.array(new_v[at_locally_owned(geometry)])
    h_locally_owned = np.array(new_h[at_locally_owned(geometry)])
    
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
    b = jnp.copy(zero_field)
    h = jnp.copy(zero_field)

    h = h.at[at_locally_owned(geometry)].set(10.0)

    new_u, new_v, new_h = advance_model_n_steps(u, v, h, 1.0, geometry, b, num_steps, dt, dx, dy)

    u_locally_owned = np.array(new_u[at_locally_owned(geometry)])
    v_locally_owned = np.array(new_v[at_locally_owned(geometry)])
    h_locally_owned = np.array(new_h[at_locally_owned(geometry)])
    
    u_global = gather_global_field(u_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    v_global = gather_global_field(v_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
    h_global = gather_global_field(h_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

    if rank == root:
        assert np.all(np.equal(u_global, [0.0]))
        assert np.all(np.equal(v_global, [0.0]))
        assert np.all(np.equal(h_global, [10.0]))

# def test_model_3():

#     xmax = ymax = 10000.0
#     nx = ny = 11
#     dx = dy = xmax / (nx - 1.0)
#     g = 9.81
#     dt = 0.68 * dx / sqrt(g * 5030.0)
#     num_steps = 1

#     domain = RectangularDomain(nx, ny)
#     geometry = create_par_geometry(mpi4jax_comm, rank, size, domain)
#     locally_owned_field = 0 * jnp.ones((geometry.locally_owned_extent_x,geometry.locally_owned_extent_y), dtype=jnp.float64)
#     zero_field = create_par_field(locally_owned_field, geometry)

#     u = jnp.copy(zero_field.value)
#     v = jnp.copy(zero_field.value)
#     h = jnp.copy(zero_field.value)
#     b = jnp.copy(zero_field.value)

#     # Create tsunami pluse IC
#     xmid = xmax / 2.0
#     ymid = ymax / 2.0
#     sigma = floor(xmax / 20.0)
#     start, end = get_locally_owned_range(geometry)
#     h_global = np.zeros((nx,ny), dtype=jnp.float64)
#     for j in range(ny):
#         for i in range(nx):
#             dsqr = ((i) * dx - xmid)**2 + ((j) * dy - ymid)**2
#             h_global[i,j] = (5000.0 + (5030.0 - 5000.0) * exp(-dsqr / sigma**2))

#     if rank == 0:
#         h = h.at[start.x:end.x, start.y:end.y].set(h_global[0:5, 0:5])
#     if rank == 1:
#         h = h.at[start.x:end.x, start.y:end.y].set(h_global[5:11, 0:5])
#     if rank == 2:
#         h = h.at[start.x:end.x, start.y:end.y].set(h_global[0:5, 5:11])
#     if rank == 3:
#         h = h.at[start.x:end.x, start.y:end.y].set(h_global[5:11, 5:11])

#     state = State(u,v,h,1.0,geometry)

#     start, end = get_locally_owned_range(geometry)

#     locally_owned_field = np.array(state.h[start.x:end.x,start.y:end.y])
#     visualize_locally_owned_field("ic.png", locally_owned_field, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

#     new_state = advance_model_n_steps(state, b, num_steps, dt, dx, dy)

#     locally_owned_field = np.array(new_state.h[start.x:end.x,start.y:end.y])
#     visualize_locally_owned_field("step{}.png".format(num_steps), locally_owned_field, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

#     u_locally_owned = np.array(state.u[start.x:end.x,start.y:end.y])
#     v_locally_owned = np.array(state.v[start.x:end.x,start.y:end.y])
#     h_locally_owned = np.array(state.h[start.x:end.x,start.y:end.y])
    
#     u_global = gather_global_field(u_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
#     v_global = gather_global_field(v_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)
#     h_global = gather_global_field(h_locally_owned, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

#     rms_u = sqrt(np.sum(u_global * u_global) / (nx * ny))
#     rms_v = sqrt(np.sum(v_global * v_global) / (nx * ny))
#     rms_h = sqrt(np.sum(h_global * h_global) / (nx * ny))



#     if rank == root:
#         print('rank({}): rms_u = {}, rms_v = {}, rms_h = {}'.format(rank, rms_u, rms_v, rms_h))
#         assert abs(rms_u - 0.00161019683016338) < 10.e-12
#         assert abs(rms_v - 0.00161019683016338) < 10.e-12
#         assert abs(rms_h - 5000.37196249264) < 10.e-12

