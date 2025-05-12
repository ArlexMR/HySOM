import matplotlib.pyplot as plt
from hysom.utils.aux_funcs import split_range_auto
from string import ascii_uppercase

def plot_map(prototypes, axs = None, loop_cmap = "inferno", sample_loop_coords = (0,0)):
    """
    Plot Self-Organizing Map grid of prototypes.

    Parameters
    ----------
    prototypes : np.ndarray
        prototypes as given by HSOM.get_prototypes()

    axs : array-like, optional
        array of matplotlib axes to plot on. If None, a new figure and axes are created.

    loop_cmap : str or cmap
        Any matplotlib colormap. Default is "inferno".

    sample_loop_coords: tuple, optional
        coordinates of the sample loop to be plotted in the upper right corner of the figure as a sample loop.
        Default is (0,0).
    """
    
    if axs is None:
        height, width = prototypes.shape[:2]
        fig, axs = _make_figure(height, width, figsize = (width + 1,height))
    else:
        fig = axs[0,0].figure
    
    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            ax = axs[row,col]
            loop = prototypes[row,col]
            _plot_loop(ax, loop, loop_cmap)

    _add_map_coordinates(axs)     

    # Add sample loop 
    sample_loop = prototypes[sample_loop_coords]
    _add_sample_loop(fig, sample_loop, cmap=loop_cmap)  
    
    # if return_axes:     
    return axs


def _make_figure(height, width, figsize = None):
    if figsize is None:
        figsize = (width + 1,height)
    fig, axs = plt.subplots(height,width, figsize = figsize)
    plt.subplots_adjust(wspace = 0.0, right= 0.75, hspace = 0.0)
    return fig, axs

def _plot_loop(ax, loop, cmap):
    ax.scatter( loop[:,0],loop[:,1], c = list(range(len(loop))),s = 2, cmap = cmap)
    _clean_spines_and_ticks(ax)

def _clean_spines_and_ticks(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.axis('equal')

def _add_map_coordinates(axs):
    fig = axs[0,0].figure
    h, v = fig.get_size_inches()

    for row,letter in zip(range(axs.shape[0]), ascii_uppercase):
        ymin,ymax = axs[row, 0].get_ylim()
        axs[row, 0].set_yticks(ticks = [0.5*(ymin + ymax)], labels = [letter])
        axs[row, 0].tick_params(length = h*0.5, labelsize = v)
    for col in range(axs.shape[1]):
        xmin, xmax=axs[0, col].get_xlim() 
        axs[0, col].set_xticks(ticks = [0.5*(xmin + xmax)], labels = [str(col+1)])
        axs[0, col].tick_params(bottom = False, top = True, labeltop=True, labelbottom=False, length = v*0.5, labelsize = v)

def _add_sample_loop(fig, sample_loop, cmap):
    
    ax = fig.add_axes([0.78, 0.76, 0.10, 0.10])
    axcb = fig.add_axes([0.79, 0.87, 0.08, 0.01])
    sc = ax.scatter(sample_loop[:,0], sample_loop[:,1], c = list(range(len(sample_loop))), s = 2, cmap = cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Turbidity", fontsize = 8)
    ax.set_xlabel("Discharge", fontsize = 8)
    plt.colorbar(sc, cax = axcb, orientation = "horizontal")
    axcb.set_xticks([0,100], labels = ["start", "end"], fontsize = 6)
    axcb.tick_params(bottom = False, top = True, labelbottom = False, labeltop = True, pad = 1)
    