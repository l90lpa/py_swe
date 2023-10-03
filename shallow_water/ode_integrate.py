
from .geometry import at_locally_owned_interior
from .scan_functions import jax_scan
from .state import State


def forward_euler_solver_step(f, dt, s_new, s, token, geometry):
    s_new, s, token = f(s_new, s, token)
    
    (u_new, v_new, h_new) = s_new
    (u, v, h) = s

    interior = at_locally_owned_interior(geometry)
    u_new = u_new.at[interior].set(u[interior] + dt * u_new[interior])
    v_new = v_new.at[interior].set(v[interior] + dt * v_new[interior])
    h_new = h_new.at[interior].set(h[interior] + dt * h_new[interior])

    return s, State(u_new, v_new, h_new), token


def integrate(dynamics,
              ic,
              dt,
              nt,
              solver_step,
              *,
              scan_function=jax_scan):
    
    def solver_wrapped(y, _):
        y = solver_step(dynamics, dt, *y)
        return y, None
    
    return scan_function(solver_wrapped, ic, nt)