import pytest

from py_swe.geometry import RectangularGrid, Vec2, coord_to_index_xy_order, index_to_coord_xy_order, partition_rectangular_grid, create_geometry

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
    domain = RectangularGrid(17,17)
    num_subdomains = 4

    subgrids = partition_rectangular_grid(domain, num_subdomains).subgrids

    assert len(subgrids) == 4

    assert subgrids[0].start_x == 0 and subgrids[0].start_y == 0
    assert subgrids[1].start_x > 0 and subgrids[1].start_y == 0 
    assert subgrids[2].start_x == 0 and subgrids[2].start_y > 0
    assert subgrids[3].start_x > 0 and subgrids[3].start_y > 0

    assert subgrids[3].start_x == subgrids[0].local_nx and subgrids[3].start_y == subgrids[0].local_ny

def test_create_par_geometry():
    
    rank = 3
    size = 4
    grid = RectangularGrid(17,17)

    geometry = create_geometry(rank, size, grid, 1, 1)

    assert geometry.nxprocs == 2 and geometry.nyprocs == 2

    assert geometry.local_grid_extent.x == 9 and geometry.local_grid_extent.y == 9

    assert geometry.local_halo_depth.north == 0
    assert geometry.local_halo_depth.south == 1
    assert geometry.local_halo_depth.east == 0
    assert geometry.local_halo_depth.west == 1

    assert geometry.local_ghost_depth.north == 1
    assert geometry.local_ghost_depth.south == 0
    assert geometry.local_ghost_depth.east == 1
    assert geometry.local_ghost_depth.west == 0

    assert geometry.local_topology.north == -1
    assert geometry.local_topology.south == 1
    assert geometry.local_topology.east == -1
    assert geometry.local_topology.west == 2

def test_create_par_geometry_2():
    
    rank = 4
    size = 9
    grid = RectangularGrid(3,3)

    geometry = create_geometry(rank, size, grid, 1, 1)

    assert geometry.nxprocs == 3 and geometry.nyprocs == 3

    assert geometry.local_grid_extent.x == 1 and geometry.local_grid_extent.y == 1

    assert geometry.local_halo_depth.north == 1
    assert geometry.local_halo_depth.south == 1
    assert geometry.local_halo_depth.east == 1
    assert geometry.local_halo_depth.west == 1

    assert geometry.local_ghost_depth.north == 0
    assert geometry.local_ghost_depth.south == 0
    assert geometry.local_ghost_depth.east == 0
    assert geometry.local_ghost_depth.west == 0

    assert geometry.local_topology.north == 7
    assert geometry.local_topology.south == 1
    assert geometry.local_topology.east == 5
    assert geometry.local_topology.west == 3