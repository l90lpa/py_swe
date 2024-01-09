import jax.numpy as jnp
from jax import jit
import jax.numpy as jnp
import mpi4jax_ad_wrapper
# Abusing mpi4jax by exposing unpack_hashable, to unpack HashableMPIType which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import unpack_hashable

from .geometry import ParGeometry, get_locally_owned_range
from .state import State

def exchange_field_halos(field, geometry: ParGeometry, comm_wrapped, token):

    if geometry.global_pg.nxprocs * geometry.global_pg.nyprocs == 1:
        return field, token

    comm = unpack_hashable(comm_wrapped)
    local_topology = geometry.local_pg.topology
    halo_depth = geometry.local_domain.halo_depth

    start, end = get_locally_owned_range(geometry)

    # send buffer slices
    halo_source_slices = {
        "north": (slice(start.x, end.x), slice(-2 * halo_depth.north, -halo_depth.north)), 
        "south": (slice(start.x, end.x), slice(halo_depth.south, 2 * halo_depth.south)),    
        "east" : (slice(-2 * halo_depth.east, -halo_depth.east), slice(start.y, end.y)),   
        "west" : (slice(halo_depth.west, 2 * halo_depth.west), slice(start.y, end.y))}     
    
    # recv buffer slices
    halo_slices = {
        "north": (slice(start.x, end.x), slice(-halo_depth.north, None)),
        "south": (slice(start.x, end.x), slice(0,halo_depth.south)),      
        "east" : (slice(-halo_depth.east, None), slice(start.y, end.y)),  
        "west" : (slice(0,halo_depth.west), slice(start.y, end.y))}       
    
    neighbor_ids = {
        "north": local_topology.north,
        "south": local_topology.south,
        "east":  local_topology.east,
        "west":  local_topology.west}
    
    send_recv_pairs = [
        ("south", "north"),
        ("west", "east"),
        ("north", "south"),
        ("east", "west"),
    ]

    new_field = jnp.copy(field)

    for send_name, recv_name in send_recv_pairs:
        send_id = neighbor_ids[send_name]
        recv_id = neighbor_ids[recv_name]
        recv_buf = field[halo_slices[recv_name]]
        send_buf = field[halo_source_slices[send_name]]

        if send_id == -1 and recv_id == -1:
            continue
        elif send_id == -1:
            recv_buf, token = mpi4jax_ad_wrapper.recv(recv_buf, token, recv_id, comm=comm)
            new_field = new_field.at[halo_slices[recv_name]].set(recv_buf)
        elif recv_id == -1:
            token = mpi4jax_ad_wrapper.send(send_buf, token, send_id, comm=comm)
        else:
            recv_buf, token = mpi4jax_ad_wrapper.sendrecv(send_buf, recv_buf, token, recv_id, send_id, comm=comm)
            new_field = new_field.at[halo_slices[recv_name]].set(recv_buf)
        
    return new_field, token

def exchange_state_halos(s, geometry, comm_wrapped, token):

    (u, v, h) = s

    u_new, token = exchange_field_halos(u, geometry, comm_wrapped, token)
    v_new, token = exchange_field_halos(v, geometry, comm_wrapped, token)
    h_new, token = exchange_field_halos(h, geometry, comm_wrapped, token)

    return State(u_new, v_new, h_new), token