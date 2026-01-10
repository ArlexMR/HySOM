import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as mplFigure
from matplotlib.colors import Colormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from string import ascii_uppercase
from typing import Iterable, Tuple, Callable, Any, Literal
from hysom import HSOM
from hysom.utils.aux_funcs import split_range_auto


def plot_map(prototypes, axs = None, loop_cmap = "inferno", sample_loop_coords = (0,0), coordinates_style: Literal["alphanumeric", "matrix"] = "alphanumeric"):
    """
    Plot Self-Organizing Map grid of prototypes.

    Parameters
    ----------
    prototypes : np.ndarray
        prototypes as given by HSOM.get_prototypes()

    axs : array-like, optional
        array of matplotlib axes to plot on. If None (recomended), a new figure and axes are created.

    loop_cmap : str or cmap
        Any matplotlib colormap. Default is "inferno".

    sample_loop_coords: tuple, optional
        coordinates of the sample loop to be plotted in the upper right corner of the figure as a sample loop.
        Coordinates are given in matrix format (row, col). Default is (0,0).
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

    _add_map_coordinates(axs, style = coordinates_style)     

    # Add sample loop 
    sample_loop = prototypes[sample_loop_coords]
    _add_sample_loop(fig, sample_loop, cmap=loop_cmap)  
    
    return axs


def _make_figure(height, width, figsize = None)-> Tuple[mplFigure, np.ndarray]:
    if figsize is None:
        figsize = (width + 1,height)
    fig, axs = plt.subplots(height,width, figsize = figsize, squeeze=False)
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

def _add_map_coordinates(axs, style: Literal["alphanumeric", "matrix"]):
    fig = axs[0,0].figure
    h, v = fig.get_size_inches()
    if style == "alphanumeric":
        row_coords = ascii_uppercase
        col_offset = 1
    else:
        row_coords = range(axs.shape[0])
        col_offset = 0
    for row,letter in zip(range(axs.shape[0]), row_coords):
        ymin,ymax = axs[row, 0].get_ylim()
        axs[row, 0].set_yticks(ticks = [0.5*(ymin + ymax)], labels = [letter])
        axs[row, 0].tick_params(length = h*0.5, labelsize = v)
    for col in range(axs.shape[1]):
        xmin, xmax=axs[0, col].get_xlim() 
        axs[0, col].set_xticks(ticks = [0.5*(xmin + xmax)], labels = [str(col+col_offset)])
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

def heatmap_frequency(som: HSOM, loops, cmap = "Oranges", dots_color = "k",axs = None, coordinates_style: Literal['alphanumeric', 'matrix'] = "alphanumeric"):
    """ Plot frequency distribution
    Parameters
    ----------
    loops : np.ndarray
        Data array. The first dimension corresponds to the number of samples.

    cmap : str | colormap (optional, default = "Oranges")
        The colormap (an instance of matplotlib.colors.Colormap) or registered colormap name used to map 
        frequency counts to colors. See more info at:
                https://matplotlib.org/stable/users/explain/colors/colormaps.html
        
    """
    
    heat_map(som, loops=loops, 
                    values=np.ones(shape = len(loops)),
                    agg_method=len,
                    cmap = cmap,
                    colorbar_label="Count",
                    coordinates_style = coordinates_style
                    )

def heat_map(som: HSOM, loops: Iterable, values: Iterable, 
            axs: np.ndarray | None= None, 
            agg_method: Callable[[Any], float] = np.median ,
            cmap: str | Colormap = "Oranges", 
            minval: float | None = None, 
            maxval: float | None = None, 
            scale: str = "linear",
            colorbar_label: str | None = None,
            coordinates_style: Literal['alphanumeric', 'matrix'] = "alphanumeric"
            ):
    prototypes = som.get_prototypes()
    if prototypes is None:
        raise ValueError("The SOM has no prototypes. Please train the SOM before calling this function.")
    if axs is None:
        height, width = prototypes.shape[:2]
        fig, axs = _make_figure(height, width, figsize = (width + 1,height))
        _ = plot_map(prototypes, axs= axs, coordinates_style = coordinates_style)


    bmu_vals_dict = _groupby_bmu(som, loops, values)
    coloring_vals_dict = _aggregateby_bmu(bmu_vals_dict, agg_method=agg_method)
    _clear_unmatched_bmus(axs, matched_bmus =coloring_vals_dict.keys())
    _set_values_based_background(axs = axs, 
                                    bmus = list(coloring_vals_dict.keys()), 
                                    values=list(coloring_vals_dict.values()), 
                                    cmap = cmap, 
                                    minval=minval, 
                                    maxval=maxval, 
                                    scale=scale,
                                    colorbar_label = colorbar_label
                                    )
    _clear_unmatched_bmus(axs, matched_bmus =coloring_vals_dict.keys())

def _groupby_bmu(som, loops, vals):
    bmus_vals = [(som.get_BMU(loop), val) for loop, val in zip(loops,vals)]
    bmu_vals_dict = defaultdict(list)
    for bmu, val in bmus_vals:
        bmu_vals_dict[bmu].append(val)

    return bmu_vals_dict

def _aggregateby_bmu(bmu_vals_dict: dict, agg_method: Callable[[Any], float]):    
    bmu_stat = {}
    for bmu, vals_list in bmu_vals_dict.items():
        bmu_stat[bmu] = agg_method(vals_list)
    return bmu_stat

def _clear_unmatched_bmus(axs, matched_bmus):
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            if not (i,j) in matched_bmus:
                axs[i,j].collections[0].set_color("grey")
                axs[i,j].collections[0].set_alpha(0.1)

def _set_values_based_background(axs, bmus, values, cmap, minval, maxval, scale, colorbar_label):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    norm = _colorNorm(values, ncolors=cmap.N, minval=minval, maxval=maxval, scale=scale)
    for bmu, val in zip(bmus, values):
        color = cmap(norm(val))
        axs[bmu].set_facecolor(color)
    _make_colorbar(axs, norm, cmap, colorbar_label)

def _make_colorbar(axs, norm, cmap, colorbar_label):
    if colorbar_label is None:
        colorbar_label = "Values"
    scalarmappable = ScalarMappable(norm=norm, cmap=cmap)
    ax_cb = axs[0,0].figure.add_axes([0.77,0.11,0.025,0.5])
    cax = plt.colorbar(scalarmappable, cax = ax_cb)
    ax_cb.set_ylabel(colorbar_label)
    if isinstance(norm, BoundaryNorm): #Discrete colorbar
        _displace_colorbar_ticks(ax_cb, norm)

def _colorNorm(values, ncolors, minval, maxval, scale):
    if minval is None: minval = min(values) 
    if maxval is None: maxval = max(values)
    bounds = _colorbounds(values, minval, maxval, scale)
    if isinstance(values[0], int):
        norm = BoundaryNorm(bounds, ncolors)
    else:
        norm = Normalize(vmin = minval, vmax=maxval)
    return norm

def _colorbounds(values, minval, maxval, scale):
    if minval == maxval:
        maxval =int(maxval) + 0.9999   #get a full (constant) range
        bounds = [minval, maxval]
    elif isinstance(values[0],int): 
        splitted_range = split_range_auto(minval, maxval, max_parts=10)
        bounds = splitted_range + [splitted_range[-1] + 1] # add an additional element so max value is correctly included
    else:
        bounds = np.linspace(minval, maxval, num = 10)
    return bounds

def _displace_colorbar_ticks(ax_cb, norm):
    ylims = ax_cb.get_ylim()
    ax_cb.tick_params(axis='y', which='minor', right=False)
    yticks = norm.boundaries
    displaced_yticks = yticks[:-1] + 0.5 * np.diff(yticks)
    yticklabels = [str(yt) for yt in yticks[:-1]]
    ax_cb.set_yticks(displaced_yticks, yticklabels)
    ax_cb.set_ylim(ylims)
        