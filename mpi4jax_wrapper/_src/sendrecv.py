import numpy as _np
from mpi4py import MPI as _MPI

from jax.core import Primitive
from jax.lax import create_token
from jax.interpreters import mlir, ad

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    wrap_as_hashable,
    ordered_effect,
)
from mpi4jax._src.validation import enforce_types
from mpi4jax._src.comm import get_default_comm
from mpi4jax._src.jax_compat import ShapedArray


# The Jax primitive
mpi_sendrecv_p = Primitive("sendrecv_mpi_wrapped")  # Create the primitive
mpi_sendrecv_impl = default_primitive_impl(mpi_sendrecv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    dest=_np.integer,
    sendtag=_np.integer,
    recvtag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    # token=(type(None), Token, Tracer),
)
def sendrecv(
    sendbuf,
    recvbuf,
    token,
    source,
    dest,
    *,
    sendtag=0,
    recvtag=_MPI.ANY_TAG,
    comm=None,
    status=None,
):
    """Perform a sendrecv operation.

    .. warning::

        Unlike mpi4py's sendrecv, this returns a *new* array with the received data.

    Arguments:
        sendbuf: Array or scalar input to send.
        recvbuf: Array or scalar input with the correct shape and dtype. This can
           contain arbitrary data and will not be overwritten.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        source (int): Rank of the source MPI process.
        dest (int): Rank of the destination MPI process.
        sendtag (int): Tag of this message for sending.
        recvtag (int): Tag of this message for receiving.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        status (mpi4py.MPI.Status): Status object, can be used for introspection.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    # Using a HLO token internally as a work around for issue: https://github.com/google/jax/issues/18068
    hlo_token = create_token()

    return tuple(
        mpi_sendrecv_p.bind(
            sendbuf,
            recvbuf,
            token,
            hlo_token,
            source=source,
            dest=dest,
            sendtag=sendtag,
            recvtag=recvtag,
            comm=comm,
            status=status,
        )
    )


def mpi_sendrecv_xla_encode_cpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
    hlo_token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
):
    from mpi4jax.experimental.notoken.collective_ops.sendrecv import mpi_sendrecv_xla_encode_cpu
    
    return mpi_sendrecv_xla_encode_cpu(ctx, sendbuf, recvbuf, hlo_token, source, dest, sendtag, recvtag, comm, status), token


def mpi_sendrecv_xla_encode_gpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
    hlo_token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
):
    from mpi4jax.experimental.notoken.collective_ops.sendrecv import mpi_sendrecv_xla_encode_gpu
    
    return mpi_sendrecv_xla_encode_gpu(ctx, sendbuf, recvbuf, hlo_token, source, dest, sendtag, recvtag, comm, status), token


def mpi_sendrecv_abstract_eval(
    sendbuf,
    recvbuf,
    token,
    hlo_token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
):
    return (
        ShapedArray(recvbuf.shape, recvbuf.dtype),
        ShapedArray(token.shape, token.dtype),
    ), {ordered_effect}


def mpi_sendrecv_value_and_jvp(
    in_args,
    tan_args,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
):
    sendbuf, recvbuf, token, hlo_token = in_args
    send_tan, recv_tan, token_tan, hlo_token_tan = tan_args

    val, token = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
        hlo_token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
    )

    jvp, token_tan = mpi_sendrecv_p.bind(
        send_tan,
        recv_tan,
        token_tan,
        hlo_token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
    )

    return (val, token), (jvp, token_tan)


def mpi_sendrecv_transpose_rule(
    cot_args, *x_args, source, dest, sendtag, recvtag, comm, status
):
    _, _, token, hlo_token = x_args
    out_cot, token_cot = cot_args

    # swap the sender and receiver
    res, token_tan = mpi_sendrecv_p.bind(
        out_cot,
        out_cot,
        token_cot,
        hlo_token,
        source=dest,
        dest=source,
        sendtag=12345,
        recvtag=12345,
        comm=comm,
        status=status,
    )
    return res, ad.Zero.from_value(res), token_tan, ad.Zero.from_value(hlo_token)


mpi_sendrecv_p.multiple_results = True
mpi_sendrecv_p.def_impl(mpi_sendrecv_impl)
mpi_sendrecv_p.def_effectful_abstract_eval(mpi_sendrecv_abstract_eval)

ad.primitive_jvps[mpi_sendrecv_p] = mpi_sendrecv_value_and_jvp
ad.primitive_transposes[mpi_sendrecv_p] = mpi_sendrecv_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_gpu, platform="cuda")
