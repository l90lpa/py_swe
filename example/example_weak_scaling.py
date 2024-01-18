
from math import sqrt, ceil
import time

import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
# Exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType

from py_swe.geometry import Vec2, create_geometry_w_padding, RectangularGrid
from py_swe.state import create_local_field_zeros, create_local_state_tsunami_pulse
from py_swe.model import shallow_water_model_w_padding
from py_swe.visualize import save_global_state_domain_on_root


mpi4py_comm = MPI.COMM_WORLD
mpi4jax_comm = MPI.COMM_WORLD.Clone()
mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()
root = 0


if __name__ == "__main__":
    
    xmax = ymax = 100000.0
    # Choose number of global spatial degrees of freedom based on the number of processes to force a fixed number of spartial
    # degrees of freedom per process for weak scaling analysis
    spatial_df_per_process = 1024**2
    nx = ny = int(sqrt(size * spatial_df_per_process))
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    tmax = 150
    num_steps = ceil(tmax / dt)
    num_steps = 50

    grid = RectangularGrid(nx, ny)
    geometry = create_geometry_w_padding(rank, size, grid, Vec2(xmax, ymax))
    b = create_local_field_zeros(geometry, jnp.float64)
    s0 = create_local_state_tsunami_pulse(geometry, jnp.float64)

    token = jnp.empty((1,))


    # save_global_state_domain_on_root(s0, geometry, root, mpi4py_comm, "step-0.png", "Saved initial condition.")
    
    compile_times = []
    execution_times = []
    num_iter = 10
    for iter in range(num_iter):
        start = time.perf_counter()

        model_compiled = shallow_water_model_w_padding.lower(s0, geometry, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy, token).compile()

        end = time.perf_counter()

        compile_times.append(end - start)

        start = time.perf_counter()

        sN, _ = model_compiled(s0, b, dt, dx, dy, token)
        sN.u.block_until_ready()

        end = time.perf_counter()
        
        execution_times.append(end - start)
        
        if rank == root:
            print(f"Iteration {iter} complete.")

    print(f"mpi_size={size}, mpi_rank={rank}, nt={num_steps}, nx={nx}, ny={ny}, local_nx={geometry.local_domain.grid_extent.x}, local_ny={geometry.local_domain.grid_extent.y}\ncomplitation time: mean={np.mean(np.array(compile_times))}, std={np.std(np.array(compile_times))}\nexecution time: mean={np.mean(np.array(execution_times))}, std={np.std(np.array(execution_times))}")


    # save_global_state_domain_on_root(sN, geometry, root, mpi4py_comm, f"step-{num_steps}.png", "Saved final condition.")
    
