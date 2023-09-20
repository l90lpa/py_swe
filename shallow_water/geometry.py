from math import sqrt, floor
from collections import namedtuple
from dataclasses import dataclass
from itertools import product, accumulate
import numpy as np

@dataclass
class RectangularDomain:
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

@dataclass
class Vec2:
    x: int
    y: int

@dataclass
class ProcessGridInfo:
    rank:    int    # Total number of MPI ranks
    nxprocs:   int    # Size of the processor grid in the x direction
    nyprocs:   int    # Size of the processor grid in the y direction

@dataclass
class ProcessGridLocalTopology:
    north: int
    south: int
    east: int
    west: int

@dataclass
class HaloDepth:
    north: int
    south: int
    east: int
    west: int

@dataclass
class ParGeometry:
    pg_info: ProcessGridInfo
    pg_local_topology: ProcessGridLocalTopology
    halo_depth: HaloDepth
    locally_owned_extent_x: int
    locally_owned_extent_y: int

def get_locally_owned_shape(geometry: ParGeometry):
    shape = (
        geometry.locally_owned_extent_x,
        geometry.locally_owned_extent_y,
    )
    return shape

def get_locally_active_shape(geometry: ParGeometry):
    shape = (
        geometry.locally_owned_extent_x + geometry.halo_depth.west + geometry.halo_depth.east,
        geometry.locally_owned_extent_y + geometry.halo_depth.south + geometry.halo_depth.north,
    )
    return shape

def get_locally_owned_range(geometry: ParGeometry):

    start_x = geometry.halo_depth.west
    start_y = geometry.halo_depth.south
    end_x = (start_x + geometry.locally_owned_extent_x)
    end_y = (start_y + geometry.locally_owned_extent_y)

    return Vec2(start_x, start_y), Vec2(end_x, end_y)

def at_locally_owned(geometry: ParGeometry):
    start, end = get_locally_owned_range(geometry)
    return (slice(start.x, end.x), slice(start.y, end.y))

def get_locally_active_range(geometry: ParGeometry):

    start_x = 0
    start_y = 0
    end_x = geometry.halo_depth.west + geometry.locally_owned_extent_x + geometry.halo_depth.east
    end_y = geometry.halo_depth.south + geometry.locally_owned_extent_y + geometry.halo_depth.north

    return Vec2(start_x, start_y), Vec2(end_x, end_y)

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

def partition_rectangular_domain(domain: RectangularDomain, num_subdomains):
    ratio = domain.nx / domain.ny

    if ratio > 1:
        xy_ordered_pair = lambda smaller, larger: (larger, smaller)
    else:
        xy_ordered_pair = lambda smaller, larger: (smaller, larger)

    xy_factor_pairs = [xy_ordered_pair(i, num_subdomains // i) for i in range(1, floor(sqrt(num_subdomains))+1) if num_subdomains % i == 0]

    ratios = [a/b for a,b in xy_factor_pairs]

    ratios = np.asarray(ratios)
    idx = (np.abs(ratios - ratio)).argmin()

    nxprocs, nyprocs = xy_factor_pairs[idx]

    local_nx = split(domain.nx, nxprocs)
    start_x = prefix_sum([0, *local_nx[:-1]])
    local_ny = split(domain.ny, nyprocs)
    start_y = prefix_sum([0, *local_ny[:-1]])
    subdomain_extents = product(zip(start_y, local_ny), zip(start_x, local_nx))
    create_subdomain = lambda extents: RectangularSubdomain(nxprocs, nyprocs, extents[1][0], extents[0][0], extents[1][1], extents[0][1])
    subdomains = list(map(create_subdomain, subdomain_extents))
    Partition = namedtuple("Partition", "subdomains nxprocs nyprocs")
    return Partition(subdomains, nxprocs, nyprocs)

def create_par_geometry(rank, size, domain: RectangularDomain):

    subdomains, nxprocs, nyprocs = partition_rectangular_domain(domain, size)

    local_subdomain = subdomains[rank]

    pg_info = ProcessGridInfo(rank, nxprocs, nyprocs)

    bounds = Vec2(pg_info.nxprocs, pg_info.nyprocs)
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

    neighbor_topology = [north, south, east, west]
    halo_depth = list(map(lambda neighbor_id: 1 if neighbor_id != -1 else 0, neighbor_topology))

    geometry = ParGeometry(pg_info, ProcessGridLocalTopology(*neighbor_topology), HaloDepth(*halo_depth), local_subdomain.local_nx, local_subdomain.local_ny)

    return geometry

