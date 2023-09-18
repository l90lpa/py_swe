
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .state import gather_global_field

def visualize_locally_owned_field(file_name: str, locally_owned_field, nxprocs, nyprocs, root, rank, comm):
    global_field = gather_global_field(locally_owned_field, nxprocs, nyprocs, root, rank, comm)

    if rank == root:
        # make a color map of fixed colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                ['blue', 'red'],
                                                256)

        # Modify data layout so that it displays as expected (x horizontal, and y vertical with origin in bottom left corner)
        global_field = np.rot90(global_field, k=3)
        global_field = np.fliplr(global_field)

        fig = plt.figure()

        # tell imshow about color map so that only set colors are used
        img = plt.imshow(global_field,interpolation='nearest', cmap = cmap,origin='lower')

        # make a color bar
        plt.colorbar(img,cmap=cmap)
        plt.grid(True,color='black')

        fig.savefig(file_name)