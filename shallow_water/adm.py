
from jax import vjp, jit
import jax.numpy as jnp

from .model import shallow_water_model


def shallow_water_model_adm(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):

    def sw_model(s, token):
        return shallow_water_model(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    
    @jit
    def sw_model_vjp_jit(x, token, Dy, Dtoken):
        primals, sw_model_vjp = vjp(sw_model, x, token)
        cotangents = sw_model_vjp((Dy, Dtoken))
        return primals, cotangents

    tok = jnp.empty((1,))
    Dtok = jnp.empty((1,))
    (y, _), (Dx, _) = sw_model_vjp_jit(s, tok, Ds, Dtok)
    
    return y, Dx


# This is a convience wrapper
def advance_adm_n_steps(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    primals, cotangents = shallow_water_model_adm(s, Ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return primals, cotangents