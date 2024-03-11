
from math import sqrt, ceil
import time

import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
# Exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType

from py_swe.geometry import Vec2, create_geometry, RectangularGrid
from py_swe.state import create_local_field_zeros, create_local_state_tsunami_pulse
from py_swe.model import advance_model_n_steps
from py_swe.visualize import save_global_state_domain_on_root

mpi4py_comm = MPI.COMM_WORLD
mpi4jax_comm = MPI.COMM_WORLD.Clone()
mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()
root = 0

if __name__ == "__main__":
    
    xmax = ymax = 100000.0
    nx = ny = 100
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    tmax = 150
    num_steps = ceil(tmax / dt)

    grid = RectangularGrid(nx, ny)
    geometry = create_geometry(rank, size, grid, 1, 1, Vec2(0.0, 0.0), Vec2(xmax, ymax))
    b = create_local_field_zeros(geometry, jnp.float64)
    s0 = create_local_state_tsunami_pulse(geometry, jnp.float64)

    save_global_state_domain_on_root(s0, geometry, root, mpi4py_comm, "step-0.png", "Saved initial condition.")

    if rank == root:
        print(f"Starting compilation.")
        start = time.perf_counter()

    model_compiled = advance_model_n_steps.lower(s0, geometry, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy).compile()

    if rank == root:
        end = time.perf_counter()
        print(f"Compilation completed in {end - start} seconds.")
        print(f"Starting simulation with {num_steps} steps...")
        start = time.perf_counter()

    sN = model_compiled(s0, b, dt, dx, dy)
    sN.u.block_until_ready()

    if rank == root:
        end = time.perf_counter()
        print(f"Simulation completed in {end - start} seconds, with an average time per step of {(end - start) / num_steps} seconds.")

    save_global_state_domain_on_root(sN, geometry, root, mpi4py_comm, f"step-{num_steps}.png", "Saved final condition.")
    
