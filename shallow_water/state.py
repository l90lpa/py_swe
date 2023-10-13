
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from .geometry import ParGeometry, Vec2, coord_to_index_xy_order, get_locally_active_shape


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
    return rng.random(shape, dtype=dtype)

def gather_global_field(locally_owned_field, nxprocs, nyprocs, root, rank, mpi4py_comm):
    '''Gather the distributed blocks of a field into a single 2D array on `rank == root`.
    Warning: one must ensure that the communicator argument, `mpi4py_comm`, is a communicator that is not used with any mpi4jax routines, 
    but strictly only with mpi4py routines.'''

    assert locally_owned_field.ndim == 2

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
