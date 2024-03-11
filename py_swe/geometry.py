from math import sqrt, floor
from collections import namedtuple
from dataclasses import dataclass
from itertools import product, accumulate
import numpy as np

@dataclass
class RectangularGrid:
    nx: int
    ny: int

@dataclass
class RectangularSubdomain:
    partition_nx: int
    partition_ny: int
    start_x: int
    start_y: int
    local_nx: int
    local_ny: int

@dataclass(eq=True, frozen=True)
class Vec2:
    x: [int|float]
    y: [int|float]

    def __add__(self, vec):
        assert isinstance(vec, type(self))
        return Vec2(self.x + vec.x, self.y + vec.y)
    
    def __sub__(self, vec):
        assert isinstance(vec, type(self))
        return Vec2(self.x - vec.x, self.y - vec.y)
    
    def __mul__(self, scalar):
        assert isinstance(scalar, int) or isinstance(scalar, float)
        return Vec2(scalar * self.x, scalar * self.y)
    
    def __rmul__(self, scalar):
        assert isinstance(scalar, int) or isinstance(scalar, float)
        return Vec2(scalar * self.x, scalar * self.y)

CardinalTuple = namedtuple('CardinalTuple', 'north south east west')

@dataclass(eq=True, frozen=True)
class Geometry:
    # Global PE Grid Info:
    size:      int    # Total number of MPI ranks (convience member, size == nxprocs * nyprocs)
    nxprocs:   int    # Size of the processor grid in the x direction
    nyprocs:   int    # Size of the processor grid in the y direction

    # Local PE Grid Info:
    local_rank:      int    # MPI ranks of this process
    local_topology:  CardinalTuple # Local process grid topology
    
    # Global Domain Info
    origin: Vec2      # The position of the global domain origin in real space
    extent: Vec2      # The extent of the global domain in real space
    grid_extent: Vec2 # The extent of the global domain in grid space
    
    # Local Domain Info
    local_grid_origin: Vec2 # The position of the local domain origin in grid space
    local_grid_extent: Vec2 # The extent of the local domain in grid space
    local_halo_depth: CardinalTuple  # The number of halo nodes in the N,S,E,and W directions
    local_ghost_depth: CardinalTuple # The number of ghost nodes in the N,S,E,and W directions

def get_locally_owned_range(geometry: Geometry):
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth
    start_x = halo_depth.west
    start_y = halo_depth.south
    end_x = (start_x + ghost_depth.west + geometry.local_grid_extent.x + ghost_depth.east)
    end_y = (start_y + ghost_depth.south + geometry.local_grid_extent.y + ghost_depth.north)

    return Vec2(start_x, start_y), Vec2(end_x, end_y)


def at_locally_owned(geometry: Geometry):
    start, end = get_locally_owned_range(geometry)
    return slice(start.x, end.x), slice(start.y, end.y)


def get_local_domain_range(geometry: Geometry):
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth
    start_x = ghost_depth.west + halo_depth.west
    start_y = ghost_depth.south + halo_depth.south
    end_x   = start_x + geometry.local_grid_extent.x
    end_y   = start_y + geometry.local_grid_extent.y

    return Vec2(start_x, start_y), Vec2(end_x, end_y)


def at_local_domain(geometry: Geometry):
    start, end = get_local_domain_range(geometry)
    return slice(start.x, end.x), slice(start.y, end.y)


def get_locally_active_range(geometry: Geometry):
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth

    start_x = 0
    start_y = 0
    
    end_x = (halo_depth.west + ghost_depth.west + 
             geometry.local_grid_extent.x + 
             halo_depth.east + ghost_depth.east)
    
    end_y = (halo_depth.south + ghost_depth.south + 
             geometry.local_grid_extent.y + 
             halo_depth.north + ghost_depth.north)

    return Vec2(start_x, start_y), Vec2(end_x, end_y)

def get_locally_active_shape(geometry: Geometry):
    _, shape = get_locally_active_range(geometry)
    return (shape.x, shape.y)


def at_locally_active(geometry: Geometry):
    start, end = get_locally_active_range(geometry)
    return (slice(start.x, end.x), slice(start.y, end.y))


def coord_to_index_xy_order(bounds: Vec2, coord: Vec2):
    index = coord.x + bounds.x * coord.y
    assert index < (bounds.x * bounds.y)
    return index

def index_to_coord_xy_order(bounds: Vec2, index: int):
    assert index < (bounds.x * bounds.y)
    y = index // bounds.x
    x = index - (bounds.x * y)
    return Vec2(x, y)

def split(x, n):
    assert x >= n
 
    if (x % n == 0):
        return [x // n for i in range(n)]
    else:
        zp = n - (x % n)
        pp = x // n
        return [pp + 1 if i >= zp else pp for i in range(n)]
    
def prefix_sum(lst):
    return list(accumulate(lst))

def partition_rectangular_grid(grid: RectangularGrid, num_subgrids):
    ratio = grid.nx / grid.ny

    if ratio > 1:
        xy_ordered_pair = lambda smaller, larger: (larger, smaller)
    else:
        xy_ordered_pair = lambda smaller, larger: (smaller, larger)

    xy_factor_pairs = [xy_ordered_pair(i, num_subgrids // i) for i in range(1, floor(sqrt(num_subgrids))+1) if num_subgrids % i == 0]

    ratios = [a/b for a,b in xy_factor_pairs]

    ratios = np.asarray(ratios)
    idx = (np.abs(ratios - ratio)).argmin()

    nxprocs, nyprocs = xy_factor_pairs[idx]

    local_nx = split(grid.nx, nxprocs)
    start_x = prefix_sum([0, *local_nx[:-1]])
    local_ny = split(grid.ny, nyprocs)
    start_y = prefix_sum([0, *local_ny[:-1]])
    subgrid_extents = product(zip(start_y, local_ny), zip(start_x, local_nx))
    create_subgrid = lambda extents: RectangularSubdomain(nxprocs, nyprocs, extents[1][0], extents[0][0], extents[1][1], extents[0][1])
    subgrid = list(map(create_subgrid, subgrid_extents))
    Partition = namedtuple("Partition", "subgrids nxprocs nyprocs")
    return Partition(subgrid, nxprocs, nyprocs)

def create_geometry(rank, size, grid: RectangularGrid, halo_depth, ghost_depth, global_origin: Vec2=Vec2(0.0,0.0), global_extent: Vec2=Vec2(1.0,1.0)):

    subgrids, nxprocs, nyprocs = partition_rectangular_grid(grid, size)
    assert size == nxprocs * nyprocs

    local_subgrid = subgrids[rank]

    bounds = Vec2(nxprocs, nyprocs)
    coord = index_to_coord_xy_order(bounds, rank)

    if coord.y != (nyprocs - 1):
        north = coord_to_index_xy_order(bounds, Vec2(coord.x, coord.y + 1)) 
    else:
        north = -1

    if coord.y != 0:
        south = coord_to_index_xy_order(bounds, Vec2(coord.x, coord.y - 1)) 
    else:
        south = -1

    if coord.x != (nxprocs - 1):
        east = coord_to_index_xy_order(bounds, Vec2(coord.x + 1, coord.y)) 
    else:
        east = -1

    if coord.x != 0:
        west = coord_to_index_xy_order(bounds, Vec2(coord.x - 1, coord.y)) 
    else:
        west = -1

    topology = CardinalTuple(north, south, east, west)
    halo = CardinalTuple(*map(lambda neighbor_id: halo_depth if neighbor_id != -1 else 0, topology))
    ghost = CardinalTuple(*map(lambda neighbor_id: ghost_depth if neighbor_id == -1 else 0, topology))
   
    geometry = Geometry(
        size=size,
        nxprocs=nxprocs,
        nyprocs=nyprocs,
        local_rank=rank,
        local_topology=topology,
        origin=global_origin,
        extent=global_extent,
        grid_extent=Vec2(grid.nx, grid.ny),
        local_grid_origin=Vec2(local_subgrid.start_x, local_subgrid.start_y),
        local_grid_extent=Vec2(local_subgrid.local_nx, local_subgrid.local_ny),
        local_halo_depth=halo,
        local_ghost_depth=ghost
        )

    return geometry