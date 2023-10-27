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
mpi_send_p = Primitive("send_mpi_wrapped")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def send(x, token, dest, *, tag=0, comm=None):
    """Perform a send operation.

    Arguments:
        x: Array or scalar input to send.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        dest (int): Rank of the destination MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        Tuple[Token]:
            - A new, modified token, that depends on this operation.

    """

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    # Using a HLO token internally as a work around for issue: https://github.com/google/jax/issues/18068
    hlo_token = create_token()

    return mpi_send_p.bind(x, token, hlo_token, dest=dest, tag=tag, comm=comm)


def mpi_send_xla_encode_cpu(ctx, x, token, hlo_token, dest, tag, comm):
    from mpi4jax.experimental.notoken.collective_ops.send import mpi_send_xla_encode_cpu

    del hlo_token

    mpi_send_xla_encode_cpu(ctx, x, dest, tag, comm)
    return (token,)

def mpi_send_xla_encode_gpu(ctx, x, token, hlo_token, dest, tag, comm):
    from mpi4jax.experimental.notoken.collective_ops.send import mpi_send_xla_encode_gpu

    del hlo_token

    mpi_send_xla_encode_gpu(ctx, x, dest, tag, comm)
    return (token,)


# This function evaluates only the shapes during AST construction
def mpi_send_abstract_eval(xs, token, hlo_token, dest, tag, comm):
    return ShapedArray(token.shape, token.dtype), {ordered_effect}

def mpi_send_value_and_jvp(primal_args, tangent_args, dest, tag, comm):
    
    sendbuf, token, hlo_token = primal_args
    sendbuf_tan, token_tan, hlo_token_tan = tangent_args

    token = mpi_send_p.bind(sendbuf, token, hlo_token, dest=dest, tag=tag, comm=comm)
    token_tan = mpi_send_p.bind(sendbuf_tan, token_tan, hlo_token, dest=dest, tag=tag, comm=comm)

    return token, token_tan


def mpi_send_transpose_rule(cotan_args, *primal_args, dest, tag, comm):
    from .recv import mpi_recv_p

    sendbuf, token, hlo_token = primal_args
    token_cot = cotan_args

    if isinstance(sendbuf, ad.UndefinedPrimal):
        zero_cot = jnp.zeros(sendbuf.aval.shape, sendbuf.aval.dtype)
    else:
        zero_cot = jnp.zeros(sendbuf.shape, sendbuf.dtype)

    sendbuf_cot, token_cot = mpi_recv_p.bind(zero_cot, token_cot, hlo_token, source=dest, tag=12345, comm=comm, status=None)

    return sendbuf_cot, token_cot, ad.Zero.from_value(hlo_token)


mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_effectful_abstract_eval(mpi_send_abstract_eval)

ad.primitive_jvps[mpi_send_p] = mpi_send_value_and_jvp
ad.primitive_transposes[mpi_send_p] = mpi_send_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_gpu, platform="cuda")
