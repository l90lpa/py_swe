
from jax import jvp, jit
import jax.numpy as jnp

from .model import shallow_water_model_w_padding


def shallow_water_model_tlm(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):

    def sw_model(s, token):
        return shallow_water_model_w_padding(s, geometry, comm_wrapped, b, n_steps, dt, dx, dy, token)
    
    @jit
    def sw_model_jvp_jit(x, token, dx, dtoken):
        return jvp(sw_model, (x, token), (dx, dtoken))

    tok = jnp.empty((1,))
    dtok = jnp.empty((1,))
    (y, _), (dy, _) = sw_model_jvp_jit(s, tok, ds, dtok)
    
    return y, dy


# This is a convience wrapper
def advance_tlm_n_steps(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy):
    primals, tangents = shallow_water_model_tlm(s, ds, geometry, comm_wrapped, b, n_steps, dt, dx, dy)
    return primals, tangents