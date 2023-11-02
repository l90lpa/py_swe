
from functools import partial

from jax import vjp, jit
import jax.numpy as jnp

from .model import shallow_water_model_w_padding

@partial(jit, static_argnames=["geometry", "n_steps", "comm_wrapped"])
def shallow_water_model_adm_w_padding(s, token, Ds, Dtoken, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    def sw_model(s, token):
        return shallow_water_model_w_padding(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    primals, sw_model_vjp = vjp(sw_model, s, token)
    cotangents = sw_model_vjp((Ds, Dtoken))
    return primals, cotangents


# This is a convience wrapper
def advance_adm_w_padding_n_steps(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    tok = jnp.empty((1,))
    Dtok = jnp.empty((1,))
    (s_new, _), (Ds_new, _) = shallow_water_model_adm_w_padding(s, tok, Ds, Dtok, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return s_new, Ds_new