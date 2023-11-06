
from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import FortranFile

from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
# Exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType

from shallow_water.geometry import Vec2, create_domain_par_geometry, add_halo_geometry, add_ghost_geometry, RectangularGrid, at_locally_owned, at_local_domain
from shallow_water.state import create_local_field_zeros, gather_global_field, create_local_field_tsunami_height
from shallow_water.model import advance_model_w_padding_n_steps
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


def save_field_figure(field, filename):
    # make a color map of fixed colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'red'], 256)

    field = np.rot90(field, k=3)
    field = np.fliplr(field)

    # tell imshow about color map so that only set colors are used
    img = plt.imshow(field, interpolation='nearest', cmap = cmap,origin='lower')

    # make a color bar
    plt.colorbar(img,cmap=cmap)
    plt.grid(True,color='black')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    
    xmax = ymax = 100000.0
    nx = ny = 101
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    num_steps = 500

    grid = RectangularGrid(nx, ny)
    geometry = create_par_geometry(rank, size, grid, Vec2(xmax, ymax))
    b = create_local_field_zeros(geometry, jnp.float64)
    s = initial_condition(geometry)


    s_global = gather_global_state_domain(s, geometry, root)
    if rank == root:
        save_field_figure(s_global.h, "step-0.png")
        

    s = advance_model_w_padding_n_steps(s, geometry, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy)


    s_global = gather_global_state_domain(s, geometry, root)
    if rank == root:
        save_field_figure(s_global.h, f"step-{num_steps}.png")
    
