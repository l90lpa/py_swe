import pytest

import jax.numpy as jnp
import numpy as np

from shallow_water.geometry import create_par_geometry, RectangularDomain, get_locally_owned_range
from shallow_water.state import create_par_field, gather_global_field
from shallow_water.runtime_context import mpi4py_comm

rank = mpi4py_comm.Get_rank()
size = mpi4py_comm.Get_size()
root = 0


def test_gather_global_field():
    assert size == 4
    
    domain = RectangularDomain(9,9)

    geometry = create_par_geometry(rank, size, domain)
    locally_owned_field = (rank + 1) * jnp.ones((geometry.locally_owned_extent_x,geometry.locally_owned_extent_y), dtype=jnp.float32)

    if rank == 0:
        locally_owned_field = locally_owned_field.at[0:2, 0].set(-2)
        locally_owned_field = locally_owned_field.at[2:4, 0].set(-1)
    if rank == 2:
        locally_owned_field = locally_owned_field.at[0:2, 0].set(-4)
        locally_owned_field = locally_owned_field.at[2:4, 0].set(-3)

    field = create_par_field(locally_owned_field, geometry)

    start, end = get_locally_owned_range(geometry)
    locally_owned_field = np.array(field.value[start.x:end.x,start.y:end.y])
    
    global_field = gather_global_field(locally_owned_field, geometry.pg_info.nxprocs, geometry.pg_info.nyprocs, root, rank, comm)

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