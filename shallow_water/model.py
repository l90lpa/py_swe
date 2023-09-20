from functools import partial
import jax.numpy as jnp
from jax import jit
import mpi4jax

from .exchange_halos import exchange_state_halos
from .geometry import ParGeometry, get_locally_owned_range, at_locally_owned
from .runtime_context import mpi4jax_comm


@partial(jit, static_argnames=['geometry'])
def apply_model(u, v, h, u_new, v_new, h_new, geometry: ParGeometry, b, dt: float, dx: float, dy: float):

    dtdx = dt / dx
    dtdy = dt / dy
    g = 9.81

    i = j = slice(1, -1)
    i_plus_1 = j_plus_1 = slice(2, None)
    i_minus_1 = j_minus_1 = slice(0, -2)
    
    u_new_interior = ((u[i_plus_1, j] + u[i_minus_1, j] + u[i, j_plus_1] + u[i, j_minus_1]) / 4.0 
             -0.5 * dtdx * ((u[i_plus_1, j]**2) / 2.0 - (u[i_minus_1, j]**2) / 2.0)
             -0.5 * dtdy * v[i, j] * (u[i, j_plus_1] - u[i, j_minus_1])
             -0.5 * dtdx * g * (h[i_plus_1, j] - h[i_minus_1, j]))
    
    v_new_interior = ((v[i_plus_1, j] + v[i_minus_1, j] + v[i, j_plus_1] + v[i, j_minus_1]) / 4.0 
             -0.5 * dtdy * ((v[i, j_plus_1]**2) / 2.0 - (v[i, j_minus_1]**2) / 2.0)
             -0.5 * dtdx * u[i, j] * (v[i_plus_1, j] - v[i_minus_1, j])
             -0.5 * dtdy * g * (h[i, j_plus_1] - h[i, j_minus_1]))
    
    h_new_interior = ((h[i_plus_1, j] + h[i_minus_1, j] + h[i, j_plus_1] + h[i, j_minus_1]) / 4.0 
             -0.5 * dtdx * u[i,j] * ((h[i_plus_1, j] - b[i_plus_1, j]) - (h[i_minus_1, j] - b[i_minus_1, j]))
             -0.5 * dtdy * v[i,j] * ((h[i, j_plus_1] - b[i, j_plus_1]) - (h[i, j_minus_1] - b[i, j_minus_1]))
             -0.5 * dtdx * (h[i,j] - b[i,j]) * (u[i_plus_1, j] - u[i_minus_1, j])
             -0.5 * dtdy * (h[i,j] - b[i,j]) * (v[i, j_plus_1] - v[i, j_minus_1]))
    
    start_x =  1 if geometry.pg_local_topology.west  == -1 else 0
    end_x   = -1 if geometry.pg_local_topology.east  == -1 else None
    start_y =  1 if geometry.pg_local_topology.south == -1 else 0
    end_y   = -1 if geometry.pg_local_topology.north == -1 else None
    
    u_new = u_new.at[slice(start_x, end_x),slice(start_y, end_y)].set(u_new_interior)
    v_new = v_new.at[slice(start_x, end_x),slice(start_y, end_y)].set(v_new_interior)
    h_new = h_new.at[slice(start_x, end_x),slice(start_y, end_y)].set(h_new_interior)

    return u_new, v_new, h_new


@partial(jit, static_argnames=['geometry'])
def apply_boundary_conditions(u, v, h, u_new, v_new, h_new, geometry):

    start, end = get_locally_owned_range(geometry)
    x_slice = slice(start.x, end.x)
    y_slice = slice(start.y, end.y)

    if geometry.pg_local_topology.south == -1:
        u_new = u_new.at[:, 0].set( u[x_slice, start.y + 1])
        v_new = v_new.at[:, 0].set(-v[x_slice, start.y + 1])
        h_new = h_new.at[:, 0].set( h[x_slice, start.y + 1])

    if geometry.pg_local_topology.north == -1:
        u_new = u_new.at[:, -1].set( u[x_slice, end.y - 2])
        v_new = v_new.at[:, -1].set(-v[x_slice, end.y - 2])
        h_new = h_new.at[:, -1].set( h[x_slice, end.y - 2])

    if geometry.pg_local_topology.west == -1:
        u_new = u_new.at[0, :].set(-u[start.x + 1, y_slice])
        v_new = v_new.at[0, :].set( v[start.x + 1, y_slice])
        h_new = h_new.at[0, :].set( h[start.x + 1, y_slice])
        
    if geometry.pg_local_topology.east == -1:
        u_new= u_new.at[-1, :].set(-u[end.x - 2, y_slice])
        v_new= v_new.at[-1, :].set( v[end.x - 2, y_slice])
        h_new= h_new.at[-1, :].set( h[end.x - 2, y_slice])

    return u_new, v_new, h_new

def advance_model_n_steps(u, v, h, max_wavespeed, geometry, b, n_steps: int, dt: float, dx: float, dy: float):

    if max_wavespeed > 0.0:
        maxdt = 0.68 * min([dx, dy]) / max_wavespeed
        if dt > maxdt:
            print("WARNING: time step, dt = ", dt, ", is too large, it should be <= ", maxdt)

    u_new = jnp.empty_like(u, shape=(geometry.locally_owned_extent_x, geometry.locally_owned_extent_y))
    v_new = jnp.empty_like(v, shape=(geometry.locally_owned_extent_x, geometry.locally_owned_extent_y))
    h_new = jnp.empty_like(h, shape=(geometry.locally_owned_extent_x, geometry.locally_owned_extent_y))
    
    for i in range(n_steps):
        u, v, h, _ = exchange_state_halos(u, v, h, geometry)
        u_new, v_new, h_new = apply_boundary_conditions(u, v, h, u_new, v_new, h_new, geometry)
        u_new, v_new, h_new = apply_model(u, v, h, u_new, v_new, h_new, geometry, b, dt, dx, dy)
        u = u.at[at_locally_owned(geometry)].set(u_new)
        v = v.at[at_locally_owned(geometry)].set(v_new)
        h = h.at[at_locally_owned(geometry)].set(h_new)

    return u, v, h

@partial(jit, static_argnames=['geometry'])
def calculate_max_wavespeed(h, geometry, token=None):

    g = 9.81

    local_max_h = jnp.max(h[at_locally_owned(geometry)])
    
    # Currently mpi4jax.Allreduce only supports MPI.SUM op, hence we cannot currently do the following:
    # global_max_h, token = mpi4py_comm.Allreduce(local_max_h, MPI.MAX, comm=mpi4jax_comm, token=token)

    size = mpi4jax_comm.Get_size()
    sendbuf = local_max_h * jnp.ones((size,),dtype=h.dtype)
    recvbuf, token = mpi4jax.alltoall(sendbuf, comm=mpi4jax_comm, token=token)
    global_max_h = jnp.max(recvbuf)

    return jnp.sqrt(g * global_max_h), token