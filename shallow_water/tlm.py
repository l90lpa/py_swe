
from jax import jvp, jit
import jax.numpy as jnp

from .exchange_halos import exchange_state_halos
from .model import apply_model, apply_boundary_conditions, advance_model_n_steps
from .state import State
from .geometry import at_local_domain


def shallow_water_dynamics_tlm(primals, tangents, geometry, comm_wrapped, b, dt, dx, dy):

    s_new, s = primals
    s_new_t, s_t = tangents

    def exchange_state_halos_wrapper(s):
        s_exc, s, _ = exchange_state_halos(s, geometry, comm_wrapped)
        return s_exc, s
    
    # we cannot jit this due to ordering probem in compiled code for MPI routines
    (s_exc, s), (s_exc_t, s_t) = jvp(exchange_state_halos_wrapper, (s,), (s_t,))

    def model(s_new, s_exc, s):
        s_new = apply_boundary_conditions(s_new, s_exc, geometry)
        s_new = apply_model(s_new, s_exc, geometry, b, dt, dx, dy)

        (u_new, v_new, h_new) = s_new
        (u, v, h) = s

        interior = at_local_domain(geometry)
        
        u_new = u_new.at[interior].set(u[interior] + dt * u_new[interior])
        v_new = v_new.at[interior].set(v[interior] + dt * v_new[interior])
        h_new = h_new.at[interior].set(h[interior] + dt * h_new[interior])
        
        s_new = State(u_new, v_new, h_new)

        return s_new
    
    @jit
    def model_jvp_jit(primals, tangents):
        return jvp(model, primals, tangents)
    
    s_new, s_new_t = model_jvp_jit((s_new, s_exc, s), (s_new_t, s_exc_t, s_t))

    return (s_new, s), (s_new_t, s_t)


def shallow_water_model_tlm(s, s_t, geometry, comm_wrapped, b, n_steps, dt, dx, dy):

    s_new = State(jnp.empty_like(s.u), jnp.empty_like(s.v), jnp.empty_like(s.h))
    s_new_t = State(jnp.zeros_like(s.u), jnp.zeros_like(s.v), jnp.zeros_like(s.h))

    for _ in range(n_steps):
        (s, s_new), (s_t, s_new_t) = shallow_water_dynamics_tlm((s_new, s), (s_new_t, s_t), geometry, comm_wrapped, b, dt, dx, dy)

    return (s,), (s_t,)


# This is a convience wrapper
def advance_tlm_n_steps(s, s_t, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    primals, tangents = shallow_water_model_tlm(s, s_t, geometry, comm_wrapped, b, n_steps, dt, dx, dy)   
    return primals[0], tangents[0]