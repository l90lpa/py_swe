import pytest

import jax.numpy as jnp
import numpy as np

from shallow_water.geometry import create_par_geometry, RectangularDomain, get_locally_owned_range, at_locally_owned
from shallow_water.state import create_local_field_ones, gather_global_field
from shallow_water.runtime_context import mpi4py_comm

rank = mpi4py_comm.Get_rank()
size = mpi4py_comm.Get_size()
root = 0


def test_gather_global_field():
    assert size == 4
    
    domain = RectangularDomain(9,9)

    geometry = create_par_geometry(rank, size, domain)
    field = (rank + 1) * create_local_field_ones(geometry, jnp.float32)
    start, end = get_locally_owned_range(geometry)

    if rank == 0:
        field = field.at[start.x:start.x+2, start.y].set(-2)
        field = field.at[start.x+2:start.x+4, start.y].set(-1)
    if rank == 2:
        field = field.at[start.x:start.x+2, start.y].set(-4)
        field = field.at[start.x+2:start.x+4, start.y].set(-3)
                                         
    locally_owned_field = np.array(field[at_locally_owned(geometry)])
    
    global_field = gather_global_field(locally_owned_field, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, mpi4py_comm)

    if rank == root:
        if size == 4:
            hypo_block_0 = 1 * jnp.ones((4,4), dtype=jnp.float32)
            hypo_block_0 = hypo_block_0.at[0:2, 0].set(-2)
            hypo_block_0 = hypo_block_0.at[2:4, 0].set(-1)
            hypo_block_1 = 2 * jnp.ones((5,4), dtype=jnp.float32)
            hypo_block_2 = 3 * jnp.ones((4,5), dtype=jnp.float32)
            hypo_block_2 = hypo_block_2.at[0:2, 0].set(-4)
            hypo_block_2 = hypo_block_2.at[2:4, 0].set(-3)
            hypo_block_3 = 4 * jnp.ones((5,5), dtype=jnp.float32)

            assert np.array_equal(global_field[0:4,0:4], hypo_block_0)
            assert np.array_equal(global_field[4:9,0:4], hypo_block_1)
            assert np.array_equal(global_field[0:4,4:9], hypo_block_2)
            assert np.array_equal(global_field[4:9,4:9], hypo_block_3)