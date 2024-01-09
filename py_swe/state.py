from math import floor, exp
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax.lax import fori_loop
from jax.tree_util import register_pytree_node

from .geometry import ParGeometry, Vec2, coord_to_index_xy_order, get_locally_active_shape, at_locally_owned


State = namedtuple('State', 'u v h')

register_pytree_node(
    State,
    lambda state: ([state.u, state.v, state.h], None),
    lambda aux_data, flat_state: State(*flat_state)
)


def create_local_field_empty(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.empty(shape, dtype=dtype)

def create_local_field_zeros(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.zeros(shape, dtype=dtype)

def create_local_field_ones(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.ones(shape, dtype=dtype)

def create_local_field_random(geometry: ParGeometry, dtype, rng=np.random.default_rng()):
    shape = get_locally_active_shape(geometry)
    # return rng.random(shape, dtype=dtype)
    return np.array(rng.normal(0.0, 1.0, shape), dtype=dtype)

def create_local_field_unit_random(geometry: ParGeometry, dtype, rng=np.random.default_rng()):
    field = create_local_field_random(geometry, dtype, rng=rng)
    norm = jnp.linalg.norm(field)
    if norm != 0:
        field /= norm
    return field

def create_local_field_tsunami_height(geometry: ParGeometry, dtype):
    # The global domain and grid must be square
    assert geometry.global_domain.extent.x == geometry.global_domain.extent.y
    assert geometry.global_domain.grid_extent.x == geometry.global_domain.grid_extent.y

    ymax = xmax = geometry.global_domain.extent.x
    ny   = nx   = geometry.global_domain.grid_extent.x
    dy   = dx   = xmax / (nx - 1)

    h = create_local_field_zeros(geometry, dtype)
    xmid = (xmax / 2.0) + geometry.global_domain.origin.x
    ymid = (ymax / 2.0) + geometry.global_domain.origin.y
    sigma = floor((xmax + 2 * dx) / 20.0)

    # Create a height field with a tsunami pulse
    local_origin_x = geometry.local_domain.grid_origin.x
    local_origin_y = geometry.local_domain.grid_origin.y
    x_slice, y_slice = at_locally_owned(geometry)
  
    def j_loop(j, ia):
        i, a = ia
        dsqr = ((i + local_origin_x) * dx - xmid) ** 2 + ((j + local_origin_y) * dy - ymid) ** 2
        a = a.at[i,j].set(5000.0 + 30.0 * jnp.exp(-dsqr / sigma ** 2))
        return i, a

    def i_loop(i, a):
        i, a = fori_loop(y_slice.start, y_slice.stop, j_loop, (i, a))
        return a

    h = fori_loop(x_slice.start, x_slice.stop, i_loop, h)

    return h

def gather_global_field(locally_owned_field, nxprocs, nyprocs, root, rank, mpi4py_comm):
    '''Gather the distributed blocks of a field into a single 2D array on `rank == root`.
    Warning: one must ensure that the communicator argument, `mpi4py_comm`, is a communicator that is not used with any mpi4jax routines, 
    but strictly only with mpi4py routines.'''

    assert locally_owned_field.ndim == 2

    if nxprocs * nyprocs == 1:
        return locally_owned_field

    # Collect local array shapes
    sendshapes = np.array(mpi4py_comm.gather(np.shape(locally_owned_field), root))

    # Compute size of each shape
    if rank == root:
        assert np.size(sendshapes, axis=0) == (nxprocs * nyprocs)
        shapesize = list(map(lambda shape: shape[0] * shape[1], [sendshapes[i,:] for i in range(np.size(sendshapes, axis=0))]))
        sendcounts = np.array(shapesize)
        sendcounts_prefix_sum = np.concatenate(([0], np.cumsum(sendcounts)))
    else:
        sendcounts = 0

    # Prepare the recieve buffer
    if rank == root:
        global_field = np.empty(sum(sendcounts), dtype=locally_owned_field.dtype)
    else:
        global_field = None

    # Gather the data
    mpi4py_comm.Gatherv(sendbuf=locally_owned_field, recvbuf=(global_field, sendcounts), root=root)
    if rank == root:
        blocks = []
        for i in range(nxprocs):
            block_column = []
            for j in range(nyprocs):
                index = coord_to_index_xy_order(Vec2(nxprocs, nyprocs), Vec2(i, j))
                block_start = sendcounts_prefix_sum[index]
                block_end = sendcounts_prefix_sum[index + 1]
                block_column.append(np.reshape(global_field[block_start:block_end], (sendshapes[index,0], sendshapes[index,1])))
            blocks.append(block_column)

        return np.block(blocks)

    return np.empty((1,))
