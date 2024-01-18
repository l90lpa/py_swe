
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .state import gather_global_state_domain

def save_state_figure(state, filename):

    def reorientate(x):
        return np.fliplr(np.rot90(x, k=3))
    
    def downsample(x, n):
        nx = np.size(x, axis=0)
        ns = nx // n
        return x[::ns,::ns]

    # make a color map of fixed colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'white', 'red'], 256)

    # modify data layout so that it displays as expected (x horizontal and y vertical, with origin in bottom left corner)
    u = reorientate(state.u)
    v = reorientate(state.v)
    h = reorientate(state.h)

    x = y = np.linspace(0, np.size(u, axis=0)-1, np.size(u, axis=0))
    xx, yy = np.meshgrid(x, y)

    # downsample velocity vector field to make it easier to read
    xx = downsample(xx, 20)
    yy = downsample(yy, 20)
    u = downsample(u, 20)
    v = downsample(v, 20)

    fig, ax = plt.subplots()
    # tell imshow about color map so that only set colors are used
    img = ax.imshow(h, interpolation='nearest', cmap=cmap, origin='lower')
    ax.quiver(xx,yy,u,v)
    plt.colorbar(img,cmap=cmap)
    plt.grid(True,color='black')
    plt.savefig(filename)


def save_global_state_domain_on_root(s, geometry, root, mpi4py_comm, filename, msg):
    s_global = gather_global_state_domain(s, geometry, root, mpi4py_comm)
    if geometry.local_pg.rank == root:
        save_state_figure(s_global, filename)
        print(msg)