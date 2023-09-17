from math import min

import numpy as np
import jax.numpy as jnp

from .state import Geometry, State

def apply_model(state: State, dt: float):

    dtdx = dt / state.geometry.dx
    dtdy = dt / state.geometry.dy
    g = 9.81

    u = state.u
    v = state.v
    h = state.h

    i = j = slice(1,-1)
    i_plus_1 = j_plus_1 = slice(2)
    i_minus_1 = j_minus_1 = slice(0,-2)
    
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
    
    state.u = u_new
    state.v = v_new
    state.h = h_new

    return state

def apply_boundary_conditions(state: State):
    g = state.geometry

    if state.geometry.north == -1:
        state.u[g.xps:g.xpe, g.yps] =  state.u[g.xps:g.xpe, g.yps + 1]
        state.v[g.xps:g.xpe, g.yps] = -state.v[g.xps:g.xpe, g.yps + 1]
        state.h[g.xps:g.xpe, g.yps] =  state.h[g.xps:g.xpe, g.yps + 1]

    if state.geometry.south == -1:
        state.u[g.xps:g.xpe, g.ype] =  state.u[g.xps:g.xpe, g.ype - 1]
        state.v[g.xps:g.xpe, g.ype] = -state.v[g.xps:g.xpe, g.ype - 1]
        state.h[g.xps:g.xpe, g.ype] =  state.h[g.xps:g.xpe, g.ype - 1]

    if state.geometry.west == -1:
        state.u[g.xps, g.yps:g.ype] = -state.u[g.xps + 1, g.yps:g.ype]
        state.v[g.xps, g.yps:g.ype] =  state.v[g.xps + 1, g.yps:g.ype]
        state.h[g.xps, g.yps:g.ype] =  state.h[g.xps + 1, g.yps:g.ype]

    if state.geometry.east == -1:
        state.u[g.xpe, g.yps:g.ype] = -state.u[g.xpe - 1, g.yps:g.ype]
        state.v[g.xpe, g.yps:g.ype] =  state.v[g.xpe - 1, g.yps:g.ype]
        state.h[g.xpe, g.yps:g.ype] =  state.h[g.xpe - 1, g.yps:g.ype]

    return state

def advance_model_n_steps(state: State, n_steps: int, dt: float):

    dx = state.geometry.dx
    dy = state.geometry.dy

    if state.max_wavespeed > 0.0:
        maxdt = 0.68 * min(dx, dy) / state.max_wavespeed
        if dt > maxdt:
            print("WARNING: time step, dt = ", dt, ", is too large, it should be <= ", maxdt)

    for i in range(n_steps):
        exchange_halos(state)
        apply_model(state, dt)
        apply_boundary_conditions(state)


if __name__ == "__main__":
    print("hello")