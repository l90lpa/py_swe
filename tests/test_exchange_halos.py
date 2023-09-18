import pytest

import jax.numpy as jnp
from mpi4py import MPI

from shallow_water.geometry import create_par_geometry, RectangularDomain
from shallow_water.state import create_par_field
from shallow_water.exchange_halos import exchange_halos

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_exchange_halos():

    if size == 1:
        domain = RectangularDomain(1,1)
    elif size == 2:
        domain = RectangularDomain(2,1)
    elif size == 3:
        domain = RectangularDomain(3,1)
    elif size == 4:
        domain = RectangularDomain(2,2)
    else:
        assert False


    geometry = create_par_geometry(comm, rank, size, domain)
    locally_owned_field = (rank + 1) * jnp.ones((geometry.locally_owned_extent_x,geometry.locally_owned_extent_y), dtype=jnp.float32)

    field = create_par_field(locally_owned_field, geometry)
    
    new_field, _ = exchange_halos(field)


    if size == 1:
        hypothesis = jnp.array([[1]], dtype=jnp.float32)
        assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 2:
        hypothesis = jnp.array([[1],[2]], dtype=jnp.float32)
        if rank == 0:
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 3:
        if rank == 0:
            hypothesis = jnp.array([[1],[2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1],[2],[3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            hypothesis = jnp.array([[2],[3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 4:
        if rank == 0:
            hypothesis = jnp.array([[1,3], [2,0]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,0],[2,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 2:
            hypothesis = jnp.array([[1,3],[0,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            hypothesis = jnp.array([[0,3],[2,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
    else:
        assert False

def test_exchange_halos_2():
    assert size <= 4

    domain = RectangularDomain(1,4)


    geometry = create_par_geometry(comm, rank, size, domain)
    locally_owned_field = (rank + 1) * jnp.ones((geometry.locally_owned_extent_x,geometry.locally_owned_extent_y), dtype=jnp.float32)

    field = create_par_field(locally_owned_field, geometry)

    new_field, _ = exchange_halos(field)


    if size == 1:
        hypothesis = jnp.array([[1,1,1,1]], dtype=jnp.float32)
        assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 2:
        if rank == 0:
            hypothesis = jnp.array([[1,1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            hypothesis = jnp.array([[1,2,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 3:
        if rank == 0:
            hypothesis = jnp.array([[1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,2,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            hypothesis = jnp.array([[2,3,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
    elif size == 4:
        if rank == 0:
            hypothesis = jnp.array([[1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,2,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        elif rank == 2:
            hypothesis = jnp.array([[2,3,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
        else:
            hypothesis = jnp.array([[3,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field.value, hypothesis)
    else:
        assert False