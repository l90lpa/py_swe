from math import floor, exp
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax.lax import fori_loop
from jax.tree_util import register_pytree_node, tree_map

from .geometry import Geometry, Vec2, coord_to_index_xy_order, get_locally_active_shape, at_locally_owned, at_local_domain


State = namedtuple('State', 'u v h')

register_pytree_node(
    State,
    lambda state: ([state.u, state.v, state.h], None),
    lambda aux_data, flat_state: State(*flat_state)
)


def create_local_field_empty(geometry: Geometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.empty(shape, dtype=dtype)

def create_local_field_zeros(geometry: Geometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.zeros(shape, dtype=dtype)

def create_local_field_ones(geometry: Geometry, dtype):
    shape = get_locally_active_shape(geometry)
    return jnp.ones(shape, dtype=dtype)

def create_local_field_random(geometry: Geometry, dtype, rng=np.random.default_rng()):
    shape = get_locally_active_shape(geometry)
    return np.array(rng.normal(0.0, 1.0, shape), dtype=dtype)

def create_local_field_unit_random(geometry: Geometry, dtype, rng=np.random.default_rng()):
    field = create_local_field_random(geometry, dtype, rng=rng)
    norm = jnp.linalg.norm(field)
    if norm != 0:
        field /= norm
    return field

def create_local_field_tsunami_height(geometry: Geometry, dtype):
    # The global domain and grid must be square
    assert geometry.extent.x == geometry.extent.y
    assert geometry.grid_extent.x == geometry.grid_extent.y

    ymax = xmax = geometry.extent.x
    ny   = nx   = geometry.grid_extent.x
    dy   = dx   = xmax / (nx - 1)

    h = create_local_field_zeros(geometry, dtype)
    xmid = (xmax / 2.0) + geometry.origin.x
    ymid = (ymax / 2.0) + geometry.origin.y
    sigma = floor((xmax + 2 * dx) / 20.0)

    # Create a height field with a tsunami pulse
    local_origin_x = geometry.local_grid_origin.x
    local_origin_y = geometry.local_grid_origin.y
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

def create_local_state_tsunami_pulse(geometry, dtype):
    zero_field = create_local_field_zeros(geometry, dtype)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = create_local_field_tsunami_height(geometry, dtype)

    print(jnp.min(h), jnp.max(h))

    return State(u, v, h)

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


def pad_field(f, geometry_padded):
    from .state import create_local_field_zeros

    zeros_field = create_local_field_zeros(geometry_padded, jnp.float64)

    f_padded = zeros_field.at[at_local_domain(geometry_padded)].set(f)
    
    return f_padded


def pad_state(s, geometry_padded):   
    return State(pad_field(s.u, geometry_padded),
                 pad_field(s.v, geometry_padded),
                 pad_field(s.h, geometry_padded))


def unpad_field(f_padded, geometry_padded):
    return f_padded[at_local_domain(geometry_padded)]


def unpad_state(s_padded, geometry_padded):
    return State(unpad_field(s_padded.u, geometry_padded),
                 unpad_field(s_padded.v, geometry_padded),
                 unpad_field(s_padded.h, geometry_padded))


def gather_global_state_domain(s, geometry, root, mpi4py_comm):
    s_local_domain = tree_map(lambda x: np.array(x[at_local_domain(geometry)]), s)
    s_global = tree_map(lambda x: gather_global_field(x, geometry.nxprocs, geometry.nyprocs, root, geometry.local_rank, mpi4py_comm), s_local_domain)
    return s_global
