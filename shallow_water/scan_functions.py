
from jax.lax import scan

def jax_scan(f, y, n_steps):
    y, _ = scan(f, y, None, n_steps)
    return y

def py_scan(f, y, n_steps):
    for _ in range(n_steps):
        y, _ = f(y, None)
    return y