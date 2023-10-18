
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

    def randomInput():
        return rng.random(shape, dtype=dtype)
    
    def randomOutput():
        return rng.random(shape, dtype=dtype)
    
    def axpyOp(a,x,y):
        if isinstance(y, type(None)):
            return a * x
        else:
            return (a * x) + y
            
    def dot(x,y):
        dot_ = jnp.sum(x * y)
        return dot_
    
    def norm(x):
        return jnp.sqrt(dot(x,x))


    def identity(s):
        return s
    
    def identity_tlm(s, ds):
        return identity(s), ds
    
    def identity_adm(s, ds):
        return identity(s), ds
    

    print("Test TLM Linearity:")
    success, absolute_error = lc.testTLMLinearity(identity_tlm, randomInput, axpyOp, norm, 1.0e-15)
    print("success = ", success, ", absolute_error = ", absolute_error)

    print("Test TLM Approximation:")
    success, relative_error = lc.testTLMApprox(identity, identity_tlm, randomInput, axpyOp, norm, 1.0e-15)
    print("success = ", success, ", relative error = ", relative_error)

    print("Test ADM Approximation:")
    success, absolute_error = lc.testADMApprox(identity_tlm, identity_adm, randomInput, randomOutput, dot, 1.0e-15)
    print("success = ", success, ", absolute_error = ", absolute_error)