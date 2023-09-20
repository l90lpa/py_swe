import pytest

from shallow_water.geometry import RectangularDomain, Vec2, coord_to_index_xy_order, index_to_coord_xy_order, partition_rectangular_domain, create_par_geometry

def test_process_coord_index_conversion():

    bounds = Vec2(2, 2)

    proc_index = 0
    proc_coord = index_to_coord_xy_order(bounds, proc_index)
    assert proc_coord == Vec2(0, 0)

    proc_coord = Vec2(0, 0)
    proc_index = coord_to_index_xy_order(bounds, proc_coord)
    assert proc_index == 0

    proc_index = 1
    proc_coord = index_to_coord_xy_order(bounds, proc_index)
    assert proc_coord == Vec2(1, 0)

    proc_coord = Vec2(1, 0)
    proc_index = coord_to_index_xy_order(bounds, proc_coord)
    assert proc_index == 1

    proc_index = 2
    proc_coord = index_to_coord_xy_order(bounds, proc_index)
    assert proc_coord == Vec2(0, 1)

    proc_coord = Vec2(0, 1)
    proc_index = coord_to_index_xy_order(bounds, proc_coord)
    assert proc_index == 2

def test_process_coord_index_conversion_2():

    bounds = Vec2(3, 3)

    proc_index = 4
    proc_coord = index_to_coord_xy_order(bounds, proc_index)
    assert proc_coord == Vec2(1, 1)

    proc_coord = Vec2(1, 1)
    proc_index = coord_to_index_xy_order(bounds, proc_coord)
    assert proc_index == 4

def test_partition_rectangular_domain():
    domain = RectangularDomain(17,17)
    num_subdomains = 4

    subdomains = partition_rectangular_domain(domain, num_subdomains).subdomains

    assert len(subdomains) == 4

    assert subdomains[0].start_x == 0 and subdomains[0].start_y == 0
    assert subdomains[1].start_x > 0 and subdomains[1].start_y == 0 
    assert subdomains[2].start_x == 0 and subdomains[2].start_y > 0
    assert subdomains[3].start_x > 0 and subdomains[3].start_y > 0

    assert subdomains[3].start_x == subdomains[0].local_nx and subdomains[3].start_y == subdomains[0].local_ny

def test_create_par_geometry():
    
    rank = 3
    size = 4
    domain = RectangularDomain(17,17)

    geometry = create_par_geometry(rank, size, domain)

    assert geometry.pg_info.nxprocs == 2 and geometry.pg_info.nyprocs == 2

    assert geometry.locally_owned_extent_x == 9 and geometry.locally_owned_extent_y == 9

    assert geometry.halo_depth.north == 0
    assert geometry.halo_depth.south == 1
    assert geometry.halo_depth.east == 0
    assert geometry.halo_depth.west == 1

    assert geometry.pg_local_topology.north == -1
    assert geometry.pg_local_topology.south == 1
    assert geometry.pg_local_topology.east == -1
    assert geometry.pg_local_topology.west == 2

def test_create_par_geometry_2():
    
    rank = 4
    size = 9
    domain = RectangularDomain(3,3)

    geometry = create_par_geometry(rank, size, domain)

    assert geometry.pg_info.nxprocs == 3 and geometry.pg_info.nyprocs == 3

    assert geometry.locally_owned_extent_x == 1 and geometry.locally_owned_extent_y == 1

    assert geometry.halo_depth.north == 1
    assert geometry.halo_depth.south == 1
    assert geometry.halo_depth.east == 1
    assert geometry.halo_depth.west == 1

    assert geometry.pg_local_topology.north == 7
    assert geometry.pg_local_topology.south == 1
    assert geometry.pg_local_topology.east == 5
    assert geometry.pg_local_topology.west == 3