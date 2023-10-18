from functools import partial
import jax.numpy as jnp
from jax import lax, debug, jit
import mpi4jax
# Abusing mpi4jax by exposing unpack_hashable, to unpack HashableMPIType which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import unpack_hashable

from .exchange_halos import exchange_state_halos
from .geometry import ParGeometry, get_locally_owned_range, at_local_domain
from .state import State
from .ode_integrate import integrate, forward_euler_solver_step


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
    
    u_new_interior = -u[i,j] + ((u[i_plus_1, j] + u[i_minus_1, j] + u[i, j_plus_1] + u[i, j_minus_1]) / 4.0 
             -0.5 * dtdx * ((u[i_plus_1, j]**2) / 2.0 - (u[i_minus_1, j]**2) / 2.0)
             -0.5 * dtdy * v[i, j] * (u[i, j_plus_1] - u[i, j_minus_1])
             -0.5 * dtdx * g * (h[i_plus_1, j] - h[i_minus_1, j]))
    
    v_new_interior = -v[i,j] + ((v[i_plus_1, j] + v[i_minus_1, j] + v[i, j_plus_1] + v[i, j_minus_1]) / 4.0 
             -0.5 * dtdy * ((v[i, j_plus_1]**2) / 2.0 - (v[i, j_minus_1]**2) / 2.0)
             -0.5 * dtdx * u[i, j] * (v[i_plus_1, j] - v[i_minus_1, j])
             -0.5 * dtdy * g * (h[i, j_plus_1] - h[i, j_minus_1]))
    
    h_new_interior = -h[i,j] + ((h[i_plus_1, j] + h[i_minus_1, j] + h[i, j_plus_1] + h[i, j_minus_1]) / 4.0 
             -0.5 * dtdx * u[i,j] * ((h[i_plus_1, j] - b[i_plus_1, j]) - (h[i_minus_1, j] - b[i_minus_1, j]))
             -0.5 * dtdy * v[i,j] * ((h[i, j_plus_1] - b[i, j_plus_1]) - (h[i, j_minus_1] - b[i, j_minus_1]))
             -0.5 * dtdx * (h[i,j] - b[i,j]) * (u[i_plus_1, j] - u[i_minus_1, j])
             -0.5 * dtdy * (h[i,j] - b[i,j]) * (v[i, j_plus_1] - v[i, j_minus_1]))
    
    interior = at_local_domain(geometry)
    u_new = u_new.at[interior].set(u_new_interior / dt)
    v_new = v_new.at[interior].set(v_new_interior / dt)
    h_new = h_new.at[interior].set(h_new_interior / dt)

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


def shallow_water_dynamics(s_new, s, token, geometry, comm_wrapped, b, dt, dx, dy):

    s_exc, s, token = exchange_state_halos(s, geometry, comm_wrapped, token=token)
    s_new = apply_boundary_conditions(s_new, s_exc, geometry)
    s_new = apply_model(s_new, s_exc, geometry, b, dt, dx, dy)

    return s_new, s, token


def calculate_max_wavespeed(h, geometry, comm_wrapped, token=None):

    g = 9.81

    local_max_h = jnp.max(h[at_local_domain(geometry)])
    
    # Currently mpi4jax.Allreduce only supports MPI.SUM op, hence we cannot currently do the following:
    # global_max_h, token = mpi4py_comm.Allreduce(local_max_h, MPI.MAX, comm=mpi4jax_comm, token=token)

    comm = unpack_hashable(comm_wrapped)
    size = comm.Get_size()
    sendbuf = local_max_h * jnp.ones((size,),dtype=h.dtype)
    recvbuf, token = mpi4jax.alltoall(sendbuf, comm=comm, token=token)
    global_max_h = jnp.max(recvbuf)

    return jnp.sqrt(g * global_max_h), token

def shallow_water_model(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    
    token = lax.create_token()

    max_wavespeed, token = calculate_max_wavespeed(s.h, geometry, comm_wrapped, token)
    maxdt = 0.68 * jnp.min(jnp.array([dx, dy])) / max_wavespeed
    lax.cond(max_wavespeed > 0.0,
             lambda : lax.cond(dt > maxdt, 
                               lambda: debug.print("WARNING: time step, dt = {}, is too large, it should be <= {}", dt, maxdt),
                               lambda: None),
             lambda : None)

    s_new = State(jnp.empty_like(s.u), jnp.empty_like(s.v), jnp.empty_like(s.h))

    def shallow_water_dynamics_(s_new, s, token):
        return shallow_water_dynamics(s_new, s, token, geometry, comm_wrapped, b, dt, dx, dy)
    
    def forward_euler_solver_step_(f, dt, s_new, s, token):
        return forward_euler_solver_step(f, dt, s_new, s, token, geometry)

    ic = (s_new, s, token)

    fc = integrate(shallow_water_dynamics_, ic, dt, n_steps, forward_euler_solver_step_)
    
    s = fc[1]

    return s


def advance_model_n_steps(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    s = shallow_water_model(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return s