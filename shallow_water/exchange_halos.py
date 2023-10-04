import jax.numpy as jnp
from jax import jit
from jax.lax import create_token
import mpi4jax
# Abusing mpi4jax by exposing unpack_hashable, to unpack HashableMPIType which is used in mpi4jax interface, from _src
from mpi4jax._src.utils import unpack_hashable
from functools import partial

from .geometry import ParGeometry, get_locally_owned_range
from .state import State

@partial(jit, static_argnames=['geometry', 'comm_wrapped'])
def exchange_field_halos(field, geometry: ParGeometry, comm_wrapped, token=None):

    if geometry.pg_info.nxprocs * geometry.pg_info.nyprocs == 1:
        return field, field, None

    comm = unpack_hashable(comm_wrapped)
    local_topology = geometry.pg_local_topology
    halo_depth = geometry.halo_depth

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
    if token is None:
        token = create_token()
    for send_name, recv_name in send_recv_pairs:
        send_id = neighbor_ids[send_name]
        recv_id = neighbor_ids[recv_name]
        recv_buf = field[halo_slices[recv_name]]
        send_buf = field[halo_source_slices[send_name]]

        if send_id == -1 and recv_id == -1:
            continue
        elif send_id == -1:
            recv_buf, token = mpi4jax.recv(recv_buf, recv_id, comm=comm, token=token)
            new_field = new_field.at[halo_slices[recv_name]].set(recv_buf)
        elif recv_id == -1:
            send_buf, token = mpi4jax.send(send_buf, send_id, comm=comm, token=token)
            field = field.at[halo_source_slices[send_name]].set(send_buf)
        else:
            recv_buf, token = mpi4jax.sendrecv(send_buf, recv_buf, recv_id, send_id, comm=comm, token=token)
            new_field = new_field.at[halo_slices[recv_name]].set(recv_buf)
        
    return new_field, field, token

@partial(jit, static_argnames=['geometry', 'comm_wrapped'])
def exchange_state_halos(s, geometry, comm_wrapped, token=None):

    (u, v, h) = s

    u_new, u, token = exchange_field_halos(u, geometry, comm_wrapped, token=token)
    v_new, v, token = exchange_field_halos(v, geometry, comm_wrapped, token=token)
    h_new, h, token = exchange_field_halos(h, geometry, comm_wrapped, token=token)  

    return State(u_new, v_new, h_new), State(u, v, h), token