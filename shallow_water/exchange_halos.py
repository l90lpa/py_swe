import numpy as np
import jax.numpy as jnp
from jax.lax import create_token
import mpi4jax

from .geometry import ParGeometry, get_locally_owned_range           
from .runtime_context import mpi4jax_comm

def exchange_field_halos(field, geometry: ParGeometry, token=None):

    if geometry.pg_info.nxprocs * geometry.pg_info.nyprocs == 1:
        return field, None

    comm = mpi4jax_comm
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
        ("north", "south"),
        ("south", "north"),
        ("east", "west"),
        ("west", "east")]

    if token is None:
        token = create_token()
    for send_name, recv_name in send_recv_pairs:
        send_id = neighbor_ids[send_name]
        recv_id = neighbor_ids[recv_name]
        recv_buf = jnp.empty_like(field[halo_slices[recv_name]])

        if send_id == -1 and recv_id == -1:
            continue
        elif send_id == -1:
            # recv_buf = np.empty_like(field[halo_slices[recv_name]])
            recv_buf, token = mpi4jax.recv(recv_buf, recv_id, comm=comm, token=token)
            field = field.at[halo_slices[recv_name]].set(recv_buf)
            # field[halo_slices[recv_name]] = recv_buf
        elif recv_id == -1:
            send_buf = field[halo_source_slices[send_name]]
            token = mpi4jax.send(send_buf, send_id, comm=comm, token=token)
        else:
            # recv_buf = np.empty_like(field[halo_source_slices[recv_name]])
            send_buf = field[halo_source_slices[send_name]]
            recv_buf, token = mpi4jax.sendrecv(send_buf, recv_buf, recv_id, send_id, comm=comm, token=token)
            field = field.at[halo_slices[recv_name]].set(recv_buf)
            # field[halo_slices[recv_name]] = recv_buf
        
    return field, token

def exchange_state_halos(u, v, h, geometry, token=None):

    u_field, token = exchange_field_halos(u, geometry, token=token)
    v_field, token = exchange_field_halos(v, geometry, token=token)
    h_field, token = exchange_field_halos(h, geometry, token=token)  

    return u_field, v_field, h_field, token