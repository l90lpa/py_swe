from functools import partial
import jax.numpy as jnp
from jax import jit
from jax.lax import create_token
import mpi4jax

from .exchange_halos import exchange_state_halos
from .geometry import ParGeometry, get_locally_owned_range, at_locally_owned, at_locally_owned_interior
from .runtime_context import mpi4jax_comm
from .state import State


@partial(jit, static_argnames=['geometry'])
def apply_model(s_new, s, geometry: ParGeometry, b, dt: float, dx: float, dy: float):

    (u_new, v_new, h_new) = s_new
    (u, v, h) = s

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
    
    u_new = u_new.at[at_locally_owned_interior(geometry)].set(u_new_interior)
    v_new = v_new.at[at_locally_owned_interior(geometry)].set(v_new_interior)
    h_new = h_new.at[at_locally_owned_interior(geometry)].set(h_new_interior)

    return State(u_new, v_new, h_new)


@partial(jit, static_argnames=['geometry'])
def apply_boundary_conditions(s_new, s, geometry):

    (u_new, v_new, h_new) = s_new
    (u, v, h) = s

    start, end = get_locally_owned_range(geometry)
    x_slice = slice(start.x, end.x)
    y_slice = slice(start.y, end.y)

    if geometry.pg_local_topology.south == -1:
        u_new = u_new.at[x_slice, start.y].set( u[x_slice, start.y + 1])
        v_new = v_new.at[x_slice, start.y].set(-v[x_slice, start.y + 1])
        h_new = h_new.at[x_slice, start.y].set( h[x_slice, start.y + 1])

    if geometry.pg_local_topology.north == -1:
        u_new = u_new.at[x_slice, end.y - 1].set( u[x_slice, end.y - 2])
        v_new = v_new.at[x_slice, end.y - 1].set(-v[x_slice, end.y - 2])
        h_new = h_new.at[x_slice, end.y - 1].set( h[x_slice, end.y - 2])

    if geometry.pg_local_topology.west == -1:
        u_new = u_new.at[start.x, y_slice].set(-u[start.x + 1, y_slice])
        v_new = v_new.at[start.x, y_slice].set( v[start.x + 1, y_slice])
        h_new = h_new.at[start.x, y_slice].set( h[start.x + 1, y_slice])
        
    if geometry.pg_local_topology.east == -1:
        u_new = u_new.at[end.x - 1, y_slice].set(-u[end.x - 2, y_slice])
        v_new = v_new.at[end.x - 1, y_slice].set( v[end.x - 2, y_slice])
        h_new = h_new.at[end.x - 1, y_slice].set( h[end.x - 2, y_slice])

    return State(u_new, v_new, h_new)

@partial(jit, static_argnames=['geometry'])
def advance_model_1_steps(s_new, s, token, geometry, b, dt, dx, dy):

    s_exc, s, token = exchange_state_halos(s, geometry)
    s_new = apply_boundary_conditions(s_new, s_exc, geometry)
    s_new = apply_model(s_new, s_exc, geometry, b, dt, dx, dy)

    return s_new, s, token

def advance_model_n_steps(s, max_wavespeed, geometry, b, n_steps: int, dt: float, dx: float, dy: float):

    if max_wavespeed > 0.0:
        maxdt = 0.68 * min([dx, dy]) / max_wavespeed
        if dt > maxdt:
            print("WARNING: time step, dt = ", dt, ", is too large, it should be <= ", maxdt)

    s_new = State(jnp.empty_like(s.u), jnp.empty_like(s.v), jnp.empty_like(s.h))
    token = create_token()

    for _ in range(n_steps):
        s, _, token = advance_model_1_steps(s_new, s, token, geometry, b, dt, dx, dy)

    return s

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