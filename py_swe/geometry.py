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

@dataclass(eq=True, frozen=True)
class ProcessGridInfo:
    rank:    int    # Total number of MPI ranks
    nxprocs:   int    # Size of the processor grid in the x direction
    nyprocs:   int    # Size of the processor grid in the y direction

# @dataclass(eq=True, frozen=True)
# class ProcessGridLocalTopology:
#     north: int
#     south: int
#     east: int
#     west: int

ProcessGridLocalTopology = namedtuple('ProcessGridLocalTopology', 'north south east west')

@dataclass(eq=True, frozen=True)
class HaloDepth:
    north: int
    south: int
    east: int
    west: int

@dataclass(eq=True, frozen=True)
class GhostDepth:
    north: int
    south: int
    east: int
    west: int

@dataclass(eq=True, frozen=True)
class ProcessGridGlobalInfo:
    size:      int    # Total number of MPI ranks (convience member, size == nxprocs * nyprocs)
    nxprocs:   int    # Size of the processor grid in the x direction
    nyprocs:   int    # Size of the processor grid in the y direction

@dataclass(eq=True, frozen=True)
class ProcessGridLocalInfo:
    rank:      int    # MPI ranks of this process
    topology:  ProcessGridLocalTopology # Local process grid topology

@dataclass(eq=True, frozen=True)
class DomainGlobalInfo:
    origin: Vec2 # The position of the global domain origin in real space
    extent: Vec2 # The extent of the global domain in real space
    grid_extent: Vec2 # The extent of the global domain in grid space

@dataclass(eq=True, frozen=True)
class DomainLocalInfo:
    grid_origin: Vec2 # The position of the local domain origin in grid space
    grid_extent: Vec2 # The extent of the local domain in grid space
    halo_depth:  HaloDepth
    ghost_depth: GhostDepth

@dataclass(eq=True, frozen=True)
class ParGeometry:
    global_pg: ProcessGridGlobalInfo
    local_pg: ProcessGridLocalInfo
    global_domain: DomainGlobalInfo
    local_domain: DomainLocalInfo

def get_locally_owned_range(geometry: ParGeometry):
    
    domain = geometry.local_domain
    start_x = domain.halo_depth.west
    start_y = domain.halo_depth.south
    end_x = (start_x + domain.ghost_depth.west + domain.grid_extent.x + domain.ghost_depth.east)
    end_y = (start_y + domain.ghost_depth.south + domain.grid_extent.y + domain.ghost_depth.north)

    return Vec2(start_x, start_y), Vec2(end_x, end_y)


def at_locally_owned(geometry: ParGeometry):
    start, end = get_locally_owned_range(geometry)
    return slice(start.x, end.x), slice(start.y, end.y)


def get_local_domain_range(geometry: ParGeometry):

    domain = geometry.local_domain
    start_x = domain.ghost_depth.west + domain.halo_depth.west
    start_y = domain.ghost_depth.south + domain.halo_depth.south
    end_x   = start_x + domain.grid_extent.x
    end_y   = start_y + domain.grid_extent.y

    return Vec2(start_x, start_y), Vec2(end_x, end_y)


def at_local_domain(geometry: ParGeometry):
    start, end = get_local_domain_range(geometry)
    return slice(start.x, end.x), slice(start.y, end.y)


def get_locally_active_range(geometry: ParGeometry):

    domain = geometry.local_domain

    start_x = 0
    start_y = 0
    
    end_x = (domain.halo_depth.west + domain.ghost_depth.west + 
             domain.grid_extent.x + 
             domain.halo_depth.east + domain.ghost_depth.east)
    
    end_y = (domain.halo_depth.south + domain.ghost_depth.south + 
             domain.grid_extent.y + 
             domain.halo_depth.north + domain.ghost_depth.north)

    return Vec2(start_x, start_y), Vec2(end_x, end_y)

def get_locally_active_shape(geometry: ParGeometry):
    _, shape = get_locally_active_range(geometry)
    return (shape.x, shape.y)


def at_locally_active(geometry: ParGeometry):
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

def create_geometry(rank, size, grid: RectangularGrid, global_origin: Vec2=Vec2(0.0,0.0), global_extent: Vec2=Vec2(1.0,1.0)):

    global_grid_extent = Vec2(grid.nx, grid.ny)

    subgrids, nxprocs, nyprocs = partition_rectangular_grid(grid, size)
    assert size == nxprocs * nyprocs

    local_subgrid = subgrids[rank]

    pg_global_info = ProcessGridGlobalInfo(size, nxprocs, nyprocs)

    bounds = Vec2(pg_global_info.nxprocs, pg_global_info.nyprocs)
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

    pg_local_topology = ProcessGridLocalTopology(*[north, south, east, west])

    pg_local_info = ProcessGridLocalInfo(rank, pg_local_topology)

    geometry = ParGeometry(
        pg_global_info,
        pg_local_info,
        DomainGlobalInfo(global_origin,
                         global_extent,
                         global_grid_extent),
        DomainLocalInfo(Vec2(local_subgrid.start_x, local_subgrid.start_y),
                        Vec2(local_subgrid.local_nx, local_subgrid.local_ny),
                        HaloDepth(0,0,0,0),
                        GhostDepth(0,0,0,0))
        )

    return geometry

def add_ghost_geometry(geometry: ParGeometry, depth):
    
    ghost_depth = GhostDepth(*map(lambda neighbor_id: depth if neighbor_id == -1 else 0, geometry.local_pg.topology))

    return ParGeometry(geometry.global_pg,
                       geometry.local_pg,
                       geometry.global_domain,
                       DomainLocalInfo(geometry.local_domain.grid_origin,
                                       geometry.local_domain.grid_extent,
                                       geometry.local_domain.halo_depth,
                                       ghost_depth))

def add_halo_geometry(geometry: ParGeometry, depth):
    
    halo_depth = HaloDepth(*map(lambda neighbor_id: depth if neighbor_id != -1 else 0, geometry.local_pg.topology))

    return ParGeometry(geometry.global_pg,
                       geometry.local_pg,
                       geometry.global_domain,
                       DomainLocalInfo(geometry.local_domain.grid_origin,
                                       geometry.local_domain.grid_extent,
                                       halo_depth,
                                       geometry.local_domain.ghost_depth))

def create_geometry_w_padding(rank, size, grid, extent):
    grid = RectangularGrid(grid.nx, grid.ny)
    geometry = create_geometry(rank, size, grid, Vec2(0.0, 0.0), extent)
    geometry = add_ghost_geometry(geometry, 1)
    geometry = add_halo_geometry(geometry, 1)
    return geometry

