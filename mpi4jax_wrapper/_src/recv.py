import numpy as _np
from mpi4py import MPI as _MPI

from jax.lax import create_token
from jax.core import Primitive

from jax.interpreters import mlir, ad
import jax.numpy as jnp

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
mpi_recv_p = Primitive("recv_mpi_wrapped")  # Create the primitive
mpi_recv_impl = default_primitive_impl(mpi_recv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
)
def recv(
    x,
    token,
    source=_MPI.ANY_SOURCE,
    *,
    tag=_MPI.ANY_TAG,
    comm=None,
    status=None,
):
    """Perform a recv (receive) operation.

    .. warning::

        Unlike mpi4py's recv, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. This can contain
           arbitrary data and will not be overwritten.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        source (int): Rank of the source MPI process.
        tag (int): Tag of this message.
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
        mpi_recv_p.bind(x, token, hlo_token, source=source, tag=tag, comm=comm, status=status)
    )


def mpi_recv_xla_encode_cpu(ctx, x, token, hlo_token, source, tag, comm, status):
    from mpi4jax.experimental.notoken.collective_ops.recv import mpi_recv_xla_encode_cpu
    
    del hlo_token

    return mpi_recv_xla_encode_cpu(ctx, x, source, tag, comm, status), token

def mpi_recv_xla_encode_gpu(ctx, x, token, hlo_token, source, tag, comm, status):
    from mpi4jax.experimental.notoken.collective_ops.recv import mpi_recv_xla_encode_gpu

    del hlo_token

    return mpi_recv_xla_encode_gpu(ctx, x, source, tag, comm, status), token


def mpi_recv_abstract_eval(xs, token, hlo_token, source, tag, comm, status):
    return (
        ShapedArray(xs.shape, xs.dtype),
        ShapedArray(token.shape, token.dtype),
    ), {ordered_effect}


def mpi_recv_value_and_jvp(primal_args, tangent_args, source, tag, comm, status):

    recvbuf, token, hlo_token = primal_args
    recvbuf_tan, token_tan, hlo_token_tan = tangent_args

    recvbuf, token = mpi_recv_p.bind(recvbuf, token, hlo_token, source=source, tag=tag, comm=comm, status=status)
    recvbuf_tan, token_tan = mpi_recv_p.bind(recvbuf_tan, token_tan, hlo_token, source=source, tag=tag, comm=comm, status=status)

    return (recvbuf, token), (recvbuf_tan, token_tan)


def mpi_recv_transpose_rule(cotan_args, *primal_args, source, tag, comm, status):
    from .send import mpi_send_p

    recvbuf, token, hlo_token = primal_args
    recvbuf_cot, token_cot = cotan_args

    token_cot = mpi_send_p.bind(recvbuf_cot, token_cot, hlo_token, dest=source, tag=12345, comm=comm)

    return jnp.zeros_like(recvbuf_cot), token_cot, ad.Zero.from_value(hlo_token)


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_effectful_abstract_eval(mpi_recv_abstract_eval)

ad.primitive_jvps[mpi_recv_p] = mpi_recv_value_and_jvp
ad.primitive_transposes[mpi_recv_p] = mpi_recv_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_gpu, platform="cuda")
