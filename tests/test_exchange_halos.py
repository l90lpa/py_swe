import pytest

import jax.numpy as jnp
from jax import jvp, vjp, jit

from mpi4py import MPI
# Abusing mpi4jax by exposing HashableMPIType, which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import HashableMPIType


from shallow_water.geometry import create_domain_par_geometry, add_halo_geometry, RectangularGrid, at_locally_owned
from shallow_water.state import create_local_field_zeros
from shallow_water.exchange_halos import exchange_field_halos

mpi4jax_comm = MPI.COMM_WORLD
rank = mpi4jax_comm.Get_rank()
size = mpi4jax_comm.Get_size()

def test_exchange_halos():
    assert size <= 4

    if size == 1:
        grid = RectangularGrid(1,1)
    elif size == 2:
        grid = RectangularGrid(2,1)
    elif size == 3:
        grid = RectangularGrid(3,1)
    elif size == 4:
        grid = RectangularGrid(2,2)
    else:
        assert False


    geometry = create_domain_par_geometry(rank, size, grid)
    geometry = add_halo_geometry(geometry, 1)
    
    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    token = jnp.empty((1,))
    new_field, _ = exchange_field_halos(field, geometry, HashableMPIType(mpi4jax_comm), token)


    if size == 1:
        hypothesis = jnp.array([[1]], dtype=jnp.float32)
        assert jnp.array_equal(new_field, hypothesis)
    elif size == 2:
        hypothesis = jnp.array([[1],[2]], dtype=jnp.float32)
        if rank == 0:
            assert jnp.array_equal(new_field, hypothesis)
        else:
            assert jnp.array_equal(new_field, hypothesis)
    elif size == 3:
        if rank == 0:
            hypothesis = jnp.array([[1],[2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1],[2],[3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        else:
            hypothesis = jnp.array([[2],[3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
    elif size == 4:
        if rank == 0:
            hypothesis = jnp.array([[1,3], [2,0]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,0],[2,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 2:
            hypothesis = jnp.array([[1,3],[0,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        else:
            hypothesis = jnp.array([[0,3],[2,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
    else:
        assert False


def test_exchange_halos_2():
    assert size <= 4

    grid = RectangularGrid(1,4)

    geometry = create_domain_par_geometry(rank, size, grid)
    geometry = add_halo_geometry(geometry, 1)

    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    token = jnp.empty((1,))
    new_field, _ = exchange_field_halos(field, geometry, HashableMPIType(mpi4jax_comm), token)


    if size == 1:
        hypothesis = jnp.array([[1,1,1,1]], dtype=jnp.float32)
        assert jnp.array_equal(new_field, hypothesis)
    elif size == 2:
        if rank == 0:
            hypothesis = jnp.array([[1,1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        else:
            hypothesis = jnp.array([[1,2,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
    elif size == 3:
        if rank == 0:
            hypothesis = jnp.array([[1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,2,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        else:
            hypothesis = jnp.array([[2,3,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
    elif size == 4:
        if rank == 0:
            hypothesis = jnp.array([[1,2]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 1:
            hypothesis = jnp.array([[1,2,3]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        elif rank == 2:
            hypothesis = jnp.array([[2,3,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
        else:
            hypothesis = jnp.array([[3,4]], dtype=jnp.float32)
            assert jnp.array_equal(new_field, hypothesis)
    else:
        assert False


def test_exchange_halos_jvp():
    assert size <= 4

    if size == 1:
        grid = RectangularGrid(1,1)
    elif size == 2:
        grid = RectangularGrid(2,1)
    elif size == 3:
        grid = RectangularGrid(3,1)
    elif size == 4:
        grid = RectangularGrid(2,2)
    else:
        assert False


    geometry = create_domain_par_geometry(rank, size, grid)
    geometry = add_halo_geometry(geometry, 1)

    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    def exchange_field_halos_jvp(field, dfield, geometry):
        token = jnp.empty((1,))
        def exchange_field_halos_wrapper(field, token):
            new_field, _ = exchange_field_halos(field, geometry, HashableMPIType(mpi4jax_comm), token)
            return new_field
        primals, tangents = jvp(exchange_field_halos_wrapper, (field, token), (dfield, token))
        return primals, tangents
    
    dfield = create_local_field_zeros(geometry, jnp.float32)
    if rank == 0:
        dfield = field

    primals, tangents = exchange_field_halos_jvp(field, dfield, geometry)


    if size == 1:
        assert jnp.array_equal(primals, jnp.array([[1]], dtype=jnp.float32))
        assert jnp.array_equal(tangents, jnp.array([[1]], dtype=jnp.float32))
    elif size == 2:
        assert jnp.array_equal(primals, jnp.array([[1],[2]], dtype=jnp.float32))
        assert jnp.array_equal(tangents, jnp.array([[1],[0]], dtype=jnp.float32))
    elif size == 3:
        if rank == 0:
            assert jnp.array_equal(primals, jnp.array([[1],[2]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[1],[0]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals, jnp.array([[1],[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[1],[0],[0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals, jnp.array([[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[0],[0]], dtype=jnp.float32))
    elif size == 4:
        if rank == 0:
            assert jnp.array_equal(primals, jnp.array([[1,3], [2,0]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[1,0], [0,0]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals, jnp.array([[1,0],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[1,0], [0,0]], dtype=jnp.float32))
        elif rank == 2:
            assert jnp.array_equal(primals, jnp.array([[1,3],[0,4]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[1,0], [0,0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals, jnp.array([[0,3],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(tangents, jnp.array([[0,0], [0,0]], dtype=jnp.float32))
    else:
        assert False


def test_exchange_halos_vjp():
    assert size <= 4

    if size == 1:
        grid = RectangularGrid(1,1)
    elif size == 2:
        grid = RectangularGrid(2,1)
    elif size == 3:
        grid = RectangularGrid(3,1)
    elif size == 4:
        grid = RectangularGrid(2,2)
    else:
        assert False


    geometry = create_domain_par_geometry(rank, size, grid)
    geometry = add_halo_geometry(geometry, 1)

    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)


    def exchange_field_halos_vjp(field, Dfield, geometry):
        token = jnp.empty((1,))
        Dtoken = jnp.empty((1,))
        def exchange_field_halos_wrapper(f, tok):
            new_f, tok = exchange_field_halos(f, geometry, HashableMPIType(mpi4jax_comm), tok)
            return new_f, tok
        primals, exchange_vjp = vjp(exchange_field_halos_wrapper, field, token)
        cotangents = exchange_vjp((Dfield, Dtoken))
        return primals[0], cotangents[0]
    
    zeros_field = create_local_field_zeros(geometry, jnp.float32)
    ones_field = zeros_field.at[:,:].set(1)
    
    if rank == 0:
        Dfield = ones_field
    else:
        Dfield = zeros_field

    primals, cotangents = exchange_field_halos_vjp(field, Dfield, geometry)


    if size == 1:
        assert jnp.array_equal(primals, jnp.array([[1]], dtype=jnp.float32))
        assert jnp.array_equal(cotangents, jnp.array([[1]], dtype=jnp.float32))
    elif size == 2:
        assert jnp.array_equal(primals, jnp.array([[1],[2]], dtype=jnp.float32))
        if rank == 0:
            assert jnp.array_equal(cotangents, jnp.array([[1],[0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(cotangents, jnp.array([[0],[1]], dtype=jnp.float32))
    elif size == 3:
        if rank == 0:
            assert jnp.array_equal(primals, jnp.array([[1],[2]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[1],[0]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals, jnp.array([[1],[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[0],[1],[0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals, jnp.array([[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[0],[0]], dtype=jnp.float32))
    elif size == 4:
        if rank == 0:
            assert jnp.array_equal(primals, jnp.array([[1,3], [2,0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[1,0], [0,1]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals, jnp.array([[1,0],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[0,0], [1,0]], dtype=jnp.float32))
        elif rank == 2:
            assert jnp.array_equal(primals, jnp.array([[1,3],[0,4]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[0,1], [0,0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals, jnp.array([[0,3],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents, jnp.array([[0,0], [0,0]], dtype=jnp.float32))
    else:
        assert False

