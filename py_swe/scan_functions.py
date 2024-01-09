
from jax.lax import scan
import equinox

def jax_scan(f, y, n_steps):
    y, _ = scan(f, y, None, n_steps)
    return y

def jax_scan_checkpointed(f, y, n_steps):
    y, _ = equinox.internal.scan(f, y, None, n_steps, kind='checkpointed', checkpoints='all')
    return y

def py_scan(f, y, n_steps):
    for _ in range(n_steps):
        y, _ = f(y, None)
    return y