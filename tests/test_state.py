import pytest

import jax.numpy as jnp

from shallow_water.geometry import create_par_geometry, RectangularDomain
from shallow_water.state import create_par_field

def test_create_par_field():

    comm = 0
    rank = 4
    size = 9
    domain = RectangularDomain(3,3)

    geometry = create_par_geometry(comm, rank, size, domain)
    locally_owned_field = (rank + 1) * jnp.ones((geometry.locally_owned_extent_x,geometry.locally_owned_extent_y), dtype=jnp.float32)

    field = create_par_field(locally_owned_field, geometry)
    
    hypothesis = jnp.array([[0,0,0],[0,5,0],[0,0,0]], dtype=jnp.float32)
    assert jnp.array_equal(field.u, hypothesis)