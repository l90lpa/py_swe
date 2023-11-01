
from functools import partial

from jax import jvp, jit
import jax.numpy as jnp

from .model import shallow_water_model_w_padding

@partial(jit, static_argnames=["geometry", "n_steps", "comm_wrapped"])
def _shallow_water_model_tlm(s, token, ds, dtoken, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    def sw_model(s, token):
        return shallow_water_model_w_padding(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    return jvp(sw_model, (s, token), (ds, dtoken))


def shallow_water_model_tlm(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):

    tok = jnp.empty((1,))
    dtok = jnp.empty((1,))
    (s_new, _), (ds_new, _) = _shallow_water_model_tlm(s, tok, ds, dtok, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    
    return s_new, ds_new


# This is a convience wrapper
def advance_tlm_n_steps(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    primals, tangents = shallow_water_model_tlm(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return primals, tangents