from mpi4py import MPI

mpi4py_comm = MPI.COMM_WORLD
mpi4jax_comm = mpi4py_comm.Clone()