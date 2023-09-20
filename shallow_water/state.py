from dataclasses import dataclass
from typing import TypeVar

from numpy.typing import NDArray
import numpy as np
import jax.numpy as jnp
# from jax.tree_util import register_pytree_node

from .geometry import ParGeometry, Vec2, coord_to_index_xy_order, get_locally_owned_range, get_locally_active_shape


@dataclass
class State:
    u: NDArray[TypeVar("NpFloat", bound=np.floating)]  #: dimension(:,:), real(r8kind)  # Maximum extent of the domain in the x direction
    v: NDArray[TypeVar("NpFloat", bound=np.floating)]  #: dimension(:,:), real(r8kind)  # Maximum extent of the domain in the y direction
    h: NDArray[TypeVar("NpFloat", bound=np.floating)]  #: dimension(:,:), real(r8kind)  # Maximum extent of the domain in the y direction
    max_wavespeed: float
    geometry: ParGeometry


def create_local_field_empty(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.empty(shape, dtype=dtype)

def create_local_field_zeros(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.zeros(shape, dtype=dtype)

def create_local_field_ones(geometry: ParGeometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.ones(shape, dtype=dtype)

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

# def flatten_geometry(geometry):
#   flat_content = (geometry.nx, geometry.ny,
#                   geometry.xmax, geometry.ymax,
#                   geometry.dx, geometry.dy,
#                   geometry.mpi_comm,
#                   geometry.nranks, geometry.rank,
#                   geometry.nxprocs, geometry.nyprocs,
#                   geometry.xproc, geometry.yproc,
#                   geometry.north, geometry.south,
#                   geometry.west, geometry.east,
#                   geometry.npx, geometry.npy,
#                   geometry.xps, geometry.xpe,
#                   geometry.yps, geometry.ype,
#                   geometry.xts, geometry.xte,
#                   geometry.yts, geometry.yte,
#                   geometry.xms, geometry.xme,
#                   geometry.yms, geometry.yme)
#   aux_data = None
#   return (flat_content, aux_data)

# def unflatten_geometry(aux_data, flat_content):
#   v = Geometry(*flat_content)
#   return v

# def flatten_state(state):
#   flat_content = (state.u, state.v, state.h, state.geometry, state.max_wavespeed)
#   aux_data = None
#   return (flat_content, aux_data)

# def unflatten_state(aux_data, flat_content):
#   v = State(*flat_content)
#   return v


# register_pytree_node(Geometry, flatten_geometry, unflatten_geometry)
# register_pytree_node(State, flatten_state, unflatten_state)
