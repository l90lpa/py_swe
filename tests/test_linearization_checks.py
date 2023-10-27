
from math import sqrt

import numpy as np

import validation.linearization_checks as lc

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


if __name__ == "__main__":

    shape = (3,3)
    dtype = jnp.float64
    rng = np.random.default_rng(12345)

    ## argument generators
    def primalArg():
        return jnp.array(rng.normal(0.0, 1.0, shape), dtype=dtype)

    def tangentArg():
        return jnp.array(rng.normal(0.0, 1.0, shape), dtype=dtype)
    
    def cotangentArg():
        return jnp.array(rng.normal(0.0, 1.0, shape), dtype=dtype)
    
    ## vector field operations for the arguments
    def scale(a, x):
        return a * x
    
    def add(x, y):
        return x + y
            
    def dot(x,y):
        dot_ = jnp.sum(x * y)
        return dot_
    
    def norm(x):
        return jnp.sqrt(dot(x,x))

    ## test function and its linearizations
    def identity(s):
        return s
    
    def identity_tlm(s, ds):
        return identity(s), ds
    
    def identity_adm(s, ds):
        return identity(s), ds
    

    print("Test TLM Linearity:")
    success, absolute_error = lc.testTLMLinearity(identity_tlm, primalArg, tangentArg, scale, norm, 1.0e-15)
    print("success = ", success, ", absolute_error = ", absolute_error)

    print("Test TLM Approximation:")
    success, relative_error = lc.testTLMApprox(identity, identity_tlm, primalArg, tangentArg, scale, add, norm, 1.0e-15)
    print("success = ", success, ", relative error = ", relative_error)

    print("Test ADM Approximation:")
    success, absolute_error = lc.testSpectralTheorem(identity_tlm, identity_adm, primalArg, tangentArg, cotangentArg, dot, 1.0e-15)
    print("success = ", success, ", absolute_error = ", absolute_error)