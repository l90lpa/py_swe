
from functools import partial

from jax import vjp, jit
import jax.numpy as jnp

from .model import shallow_water_model_w_padding

@partial(jit, static_argnames=["geometry", "n_steps", "comm_wrapped"])
def _shallow_water_model_adm(s, token, Ds, Dtoken, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    def sw_model(s, token):
        return shallow_water_model_w_padding(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    primals, sw_model_vjp = vjp(sw_model, s, token)
    cotangents = sw_model_vjp((Ds, Dtoken))
    return primals, cotangents


def shallow_water_model_adm(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):

    tok = jnp.empty((1,))
    Dtok = jnp.empty((1,))
    
    (y, _), (Dx, _) = _shallow_water_model_adm(s, tok, Ds, Dtok, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    
    return y, Dx


# This is a convience wrapper
def advance_adm_n_steps(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    primals, cotangents = shallow_water_model_adm(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return primals, cotangents