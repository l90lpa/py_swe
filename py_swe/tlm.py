
from functools import partial

from jax import jvp, jit
import jax.numpy as jnp

from .model import shallow_water_model

def shallow_water_model_tlm(s, token, ds, dtoken, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    def sw_model(s, token):
        return shallow_water_model(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    return jvp(sw_model, (s, token), (ds, dtoken))


# This is a convience wrapper
@partial(jit, static_argnames=["geometry", "n_steps", "comm_wrapped"])
def advance_tlm_n_steps(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    tok = jnp.empty((1,))
    dtok = jnp.empty((1,))
    (s_new, _), (ds_new, _) = shallow_water_model_tlm(s, tok, ds, dtok, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return s_new, ds_new