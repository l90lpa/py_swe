import pytest

import jax.numpy as jnp
from jax import jvp, vjp

from shallow_water.geometry import create_par_geometry, RectangularDomain, at_locally_owned
from shallow_water.state import create_local_field_zeros
from shallow_water.exchange_halos import exchange_field_halos
from shallow_water.runtime_context import mpi4py_comm

rank = mpi4py_comm.Get_rank()
size = mpi4py_comm.Get_size()

def test_exchange_halos():
    assert size <= 4

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


    geometry = create_par_geometry(rank, size, domain)
    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    new_field, _, _ = exchange_field_halos(field, geometry)


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

    domain = RectangularDomain(1,4)

    geometry = create_par_geometry(rank, size, domain)
    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    new_field, _, _ = exchange_field_halos(field, geometry)


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
        domain = RectangularDomain(1,1)
    elif size == 2:
        domain = RectangularDomain(2,1)
    elif size == 3:
        domain = RectangularDomain(3,1)
    elif size == 4:
        domain = RectangularDomain(2,2)
    else:
        assert False


    geometry = create_par_geometry(rank, size, domain)
    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    def exchange_field_halos_jvp(field, dfield, geometry):
        def exchange_field_halos_wrapper(field):
            new_field, _, _ = exchange_field_halos(field, geometry)
            return new_field
        primals, tangents = jvp(exchange_field_halos_wrapper, (field,), (dfield,))
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
        domain = RectangularDomain(1,1)
    elif size == 2:
        domain = RectangularDomain(2,1)
    elif size == 3:
        domain = RectangularDomain(3,1)
    elif size == 4:
        domain = RectangularDomain(2,2)
    else:
        assert False


    geometry = create_par_geometry(rank, size, domain)
    field = create_local_field_zeros(geometry, jnp.float32)
    field = field.at[at_locally_owned(geometry)].set(rank + 1)

    def exchange_field_halos_vjp(field, Dfield, geometry):
        def exchange_field_halos_wrapper(field):
            new_field, field, _ = exchange_field_halos(field, geometry)
            return new_field, field
        primals, exchange_vjp = vjp(exchange_field_halos_wrapper, field)
        cotangents = exchange_vjp(Dfield)
        return primals, cotangents
    
    zeros_field = create_local_field_zeros(geometry, jnp.float32)
    ones_field = zeros_field.at[:,:].set(1)
    
    if rank == 0:
        Dfield = (ones_field, zeros_field)
    else:
        Dfield = (zeros_field, zeros_field)

    primals, cotangents = exchange_field_halos_vjp(field, Dfield, geometry)


    if size == 1:
        assert jnp.array_equal(primals[0], jnp.array([[1]], dtype=jnp.float32))
        assert jnp.array_equal(primals[1], jnp.array([[1]], dtype=jnp.float32))
        assert jnp.array_equal(cotangents[0], jnp.array([[1]], dtype=jnp.float32))
    elif size == 2:
        assert jnp.array_equal(primals[0], jnp.array([[1],[2]], dtype=jnp.float32))
        if rank == 0:
            assert jnp.array_equal(primals[1], jnp.array([[1],[0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[1],[0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals[1], jnp.array([[0],[2]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0],[1]], dtype=jnp.float32))
    elif size == 3:
        if rank == 0:
            assert jnp.array_equal(primals[0], jnp.array([[1],[2]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[1],[0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[1],[0]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals[0], jnp.array([[1],[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[0],[2],[0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0],[1],[0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals[0], jnp.array([[2],[3]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[0],[3]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0],[0]], dtype=jnp.float32))
    elif size == 4:
        if rank == 0:
            assert jnp.array_equal(primals[0], jnp.array([[1,3], [2,0]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[1,0], [0,0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[1,0], [0,1]], dtype=jnp.float32))
        elif rank == 1:
            assert jnp.array_equal(primals[0], jnp.array([[1,0],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[0,0],[2,0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0,0], [1,0]], dtype=jnp.float32))
        elif rank == 2:
            assert jnp.array_equal(primals[0], jnp.array([[1,3],[0,4]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[0,3],[0,0]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0,1], [0,0]], dtype=jnp.float32))
        else:
            assert jnp.array_equal(primals[0], jnp.array([[0,3],[2,4]], dtype=jnp.float32))
            assert jnp.array_equal(primals[1], jnp.array([[0,0],[0,4]], dtype=jnp.float32))
            assert jnp.array_equal(cotangents[0], jnp.array([[0,0], [0,0]], dtype=jnp.float32))
    else:
        assert False

