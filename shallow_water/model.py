import numpy as np
import jax.numpy as jnp
from jax import jit

from .state import State
from .exchange_halos import exchange_state_halos
from .geometry import get_locally_owned_range


def apply_model(state: State, b, dt: float, dx: float, dy: float):

    dtdx = dt / dx
    dtdy = dt / dy
    g = 9.81

    u = state.u
    v = state.v
    h = state.h

    i = j = slice(1, -1)
    i_plus_1 = j_plus_1 = slice(2, None)
    i_minus_1 = j_minus_1 = slice(0, -2)
    
    u_new = ((u[i_plus_1, j] + u[i_minus_1, j] + u[i, j_plus_1] + u[i, j_minus_1]) / 4.0 
             -dtdx * (u[i_plus_1, j]**2 - u[i_minus_1, j]**2) / 4.0
             -0.5 * dtdy * v[i, j] * (u[i, j_plus_1] - u[i, j_minus_1])
             -0.5 * dtdx * g * (h[i_plus_1, j] - h[i_minus_1, j]))
    
    v_new = ((v[i_plus_1, j] + v[i_minus_1, j] + v[i, j_plus_1] + v[i, j_minus_1]) / 4.0 
             -dtdy * (v[i, j_plus_1]**2 - v[i, j_minus_1]**2) / 4.0
             -0.5 * dtdx * u[i, j] * (v[i_plus_1, j] - v[i_minus_1, j])
             -0.5 * dtdy * g * (h[i, j_plus_1] - h[i, j_minus_1]))
    
    h_new = ((h[i_plus_1, j] + h[i_minus_1, j] + h[i, j_plus_1] + h[i, j_minus_1]) / 4.0 
             -0.5 * dtdx * u[i,j] * ((h[i_plus_1, j] - b[i_plus_1, j]) - (h[i_minus_1, j] - b[i_minus_1, j]))
             -0.5 * dtdy * v[i,j] * ((h[i, j_plus_1] - b[i, j_plus_1]) - (h[i, j_minus_1] - b[i, j_minus_1]))
             -0.5 * dtdx * (h[i,j] - b[i,j]) * (u[i_plus_1, j] - u[i_minus_1, j])
             -0.5 * dtdy * (h[i,j] - b[i,j]) * (v[i, j_plus_1] - v[i, j_minus_1]))

    
    state.u = state.u.at[i,j].set(u_new)
    state.v = state.v.at[i,j].set(v_new)
    state.h = state.h.at[i,j].set(h_new)

    return state


def apply_boundary_conditions(state: State):

    start, end = get_locally_owned_range(state.geometry)
    x_slice = slice(start.x, end.x)
    y_slice = slice(start.y, end.y)

    if state.geometry.pg_local_topology.south == -1:
        state.u = state.u.at[x_slice, start.y].set( state.u[x_slice, start.y + 1])
        state.v = state.v.at[x_slice, start.y].set(-state.v[x_slice, start.y + 1])
        state.h = state.h.at[x_slice, start.y].set( state.h[x_slice, start.y + 1])

    if state.geometry.pg_local_topology.north == -1:
        state.u = state.u.at[x_slice, end.y - 1].set( state.u[x_slice, end.y - 2])
        state.v = state.v.at[x_slice, end.y - 1].set(-state.v[x_slice, end.y - 2])
        state.h = state.h.at[x_slice, end.y - 1].set( state.h[x_slice, end.y - 2])

    if state.geometry.pg_local_topology.west == -1:
        state.u = state.u.at[start.x, y_slice].set(-state.u[start.x + 1, y_slice])
        state.v = state.v.at[start.x, y_slice].set( state.v[start.x + 1, y_slice])
        state.h = state.h.at[start.x, y_slice].set( state.h[start.x + 1, y_slice])
        
    if state.geometry.pg_local_topology.east == -1:
        state.u = state.u.at[end.x - 1, y_slice].set(-state.u[end.x - 2, y_slice])
        state.v = state.v.at[end.x - 1, y_slice].set( state.v[end.x - 2, y_slice])
        state.h = state.h.at[end.x - 1, y_slice].set( state.h[end.x - 2, y_slice])

    return state

def advance_model_n_steps(state: State, b, n_steps: int, dt: float, dx: float, dy: float):

    if state.max_wavespeed > 0.0:
        maxdt = 0.68 * min([dx, dy]) / state.max_wavespeed
        if dt > maxdt:
            print("WARNING: time step, dt = ", dt, ", is too large, it should be <= ", maxdt)

    for i in range(n_steps):
        if state.geometry.pg_info.rank == 0:
            print('step {}'.format(i + 1))
        state, _ = exchange_state_halos(state)
        state = apply_model(state, b, dt, dx, dy)
        state = apply_boundary_conditions(state)

    return state