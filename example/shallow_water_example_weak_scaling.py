
from math import sqrt, ceil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("QtAgg")

from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
# Exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType

from shallow_water.geometry import Vec2, create_domain_par_geometry, add_halo_geometry, add_ghost_geometry, RectangularGrid, at_local_domain
from shallow_water.state import create_local_field_zeros, gather_global_field, create_local_field_tsunami_height
from shallow_water.model import shallow_water_model_w_padding, advance_model_w_padding_n_steps
from shallow_water.state import State


mpi4py_comm = MPI.COMM_WORLD
mpi4jax_comm = MPI.COMM_WORLD.Clone()
mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()
root = 0


def create_par_geometry(rank, size, grid, extent):
    grid = RectangularGrid(grid.nx, grid.ny)
    geometry = create_domain_par_geometry(rank, size, grid, Vec2(0.0, 0.0), extent)
    geometry = add_ghost_geometry(geometry, 1)
    geometry = add_halo_geometry(geometry, 1)
    return geometry


def initial_condition(geometry):
    zero_field = create_local_field_zeros(geometry, jnp.float64)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = create_local_field_tsunami_height(geometry, jnp.float64)

    return State(u, v, h)


def gather_global_state_domain(s, geometry, root):
    s_local_domain = tree_map(lambda x: np.array(x[at_local_domain(geometry)]), s)
    s_global = tree_map(lambda x: gather_global_field(x, geometry.global_pg.nxprocs, geometry.global_pg.nyprocs, root, geometry.local_pg.rank, mpi4py_comm), s_local_domain)
    return s_global


def save_state_figure(state, filename):

    def reorientate(x):
        return np.fliplr(np.rot90(x, k=3))
    
    def downsample(x, n):
        nx = np.size(x, axis=0)
        ns = nx // n
        return x[::ns,::ns]

    # make a color map of fixed colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'white', 'red'], 256)

    # modify data layout so that it displays as expected (x horizontal and y vertical, with origin in bottom left corner)
    u = reorientate(state.u)
    v = reorientate(state.v)
    h = reorientate(state.h)

    x = y = np.linspace(0, np.size(u, axis=0)-1, np.size(u, axis=0))
    xx, yy = np.meshgrid(x, y)

    # downsample velocity vector field to make it easier to read
    xx = downsample(xx, 20)
    yy = downsample(yy, 20)
    u = downsample(u, 20)
    v = downsample(v, 20)

    fig, ax = plt.subplots()
    # tell imshow about color map so that only set colors are used
    img = ax.imshow(h, interpolation='nearest', cmap=cmap, origin='lower')
    ax.quiver(xx,yy,u,v)
    plt.colorbar(img,cmap=cmap)
    plt.grid(True,color='black')
    plt.savefig(filename)


def save_global_state_domain_on_root(s, geometry, root, filename, msg):
    s_global = gather_global_state_domain(s, geometry, root)
    if rank == root:
        save_state_figure(s_global, filename)
        print(msg)


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
    geometry = create_par_geometry(rank, size, grid, Vec2(xmax, ymax))
    b = create_local_field_zeros(geometry, jnp.float64)
    s0 = initial_condition(geometry)

    token = jnp.empty((1,))


    # save_global_state_domain_on_root(s0, geometry, root, "step-0.png", "Saved initial condition.")
    
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


    # save_global_state_domain_on_root(sN, geometry, root, f"step-{num_steps}.png", "Saved final condition.")
    
