import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from string import ascii_uppercase
from math import ceil
from collections import defaultdict
from HySOM.src.hysom.utils.aux_funcs import split_range_auto
from itertools import product
from typing import Literal

def calc_HI(sample):
    x = sample[:,0]
    y = sample[:,1]
    return (0.5*(y[:-1] + y[1:]) * (x[1:] - x[:-1])).sum()


class SOMPlotter:
    def __init__(self, som):
        self.som = som
    
    def som_map(self, axs = None, loop_cmap = "inferno", return_axes = False):
        if axs is None:
            fig, axs = self._make_figure()
        else:
            fig = axs[0,0].figure
        
        for row in range(axs.shape[0]):
            for col in range(axs.shape[1]):
                ax = axs[row,col]
                loop = self.som.get_prototypes()[row,col]
                self._plot_loop(ax, loop, loop_cmap)

        self._add_map_coordinates(axs)     

        # Add sample loop 
        self._add_sample_loop(fig, cmap=loop_cmap)  
        
        # if return_axes:     
        return axs
    
    def _make_figure(self, figsize = None):
        height, width = self.som.get_prototypes().shape[:2]
        if figsize is None:
            figsize = (width + 1,height)
        fig, axs = plt.subplots(height,width, figsize = figsize)
        plt.subplots_adjust(wspace = 0.0, right= 0.75, hspace = 0.0)
        return fig, axs

    def _add_sample_loop(self, fig, cmap):
        ax = fig.add_axes([0.78, 0.76, 0.10, 0.10])
        axcb = fig.add_axes([0.79, 0.87, 0.08, 0.01])
        sample_loop = self.som.get_prototypes()[6,2]
        sc = ax.scatter(sample_loop[:,0], sample_loop[:,1], c = list(range(len(sample_loop))), s = 2, cmap = cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("Turbidity", fontsize = 8)
        ax.set_xlabel("Discharge", fontsize = 8)
        plt.colorbar(sc, cax = axcb, orientation = "horizontal")
        axcb.set_xticks([0,100], labels = ["start", "end"], fontsize = 6)
        axcb.tick_params(bottom = False, top = True, labelbottom = False, labeltop = True, pad = 1)
        
    def _add_map_coordinates(self, axs):
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

    def _plot_loop(self, ax, loop, cmap):
        ax.scatter( loop[:,0],loop[:,1], c = list(range(len(loop))),s = 2, cmap = cmap)
        self._clean_spines_and_ticks(ax)

    def _clean_spines_and_ticks(self, ax):
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(-0.1,1.1)
        ax.axis('equal')

    def quantization_error(self, loops):
        pass

    def topographic_error(self, loops):
        pass

    def Umatrix(self, loops):
        pass

    def heatmap_frequency(self, loops, cmap = "Oranges", dots_color = "k",axs = None):
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
        # if axs is None:
        #     fig, axs = self._make_figure()
        
        self.heat_map(loops=loops, 
                        values=np.ones(shape = len(loops)),
                        # axs = axs,
                        agg_method=len,
                        cmap = cmap,
                        colorbar_label="Count"
                        )
    # def scatter_frequency(self, loops, c = "k", markers = "o"):
    #     fig, axs = self._make_figure()
        

             
    #     bmus = [self.som.get_BMU(loop) for loop in loops]
    #     random_points = [()]
    #     # counter = self._get_freq_counter(loops)

    #     for (bmui,bmuj), freq in counter.items():
    #         coords = np.random.rand(freq, 2)*0.6 + 0.3
    #         axs[bmui, bmuj].scatter(coords[:,0], coords[:,1], c = , alpha = 0.9, s = 20)
    #     for ax in axs.flatten():
    #         ax.collections[0].set_alpha(0.1)


    # def _get_freq_counter(self, loops):
    #     bmu = [self.som.get_BMU(d) for d in loops]
    #     counter = Counter(bmu)
    #     return counter

    def heat_map(self, loops, values, 
                axs = None, 
                agg_method = np.median ,
                cmap = "Oranges", 
                minval = None, 
                maxval = None, 
                scale = "linear",
                colorbar_label = None
                ):
        if axs is None:
            fig, axs = self._make_figure()
            self.som_map(axs)

        bmu_vals_dict = self._groupby_bmu(loops, values)
        coloring_vals_dict = self._aggregateby_bmu(bmu_vals_dict, agg_method=agg_method)
        self._clear_unmatched_bmus(axs, matched_bmus =coloring_vals_dict.keys())
        self._set_values_based_background(axs = axs, 
                                        bmus = list(coloring_vals_dict.keys()), 
                                        values=list(coloring_vals_dict.values()), 
                                        cmap = cmap, 
                                        minval=minval, 
                                        maxval=maxval, 
                                        scale=scale,
                                        colorbar_label = colorbar_label
                                        )
        self._clear_unmatched_bmus(axs, matched_bmus =coloring_vals_dict.keys())
        
    def _clear_unmatched_bmus(self, axs,matched_bmus):
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                if not (i,j) in matched_bmus:
                    axs[i,j].collections[0].set_color("grey")
                    axs[i,j].collections[0].set_alpha(0.1)
        
    def _set_values_based_background(self, axs, bmus, values, cmap, minval, maxval, scale, colorbar_label):
        if isinstance(cmap, str):
            cmap = mpl.colormaps.get_cmap(cmap)
        # cmap.set_under('w')
        norm = self._colorNorm(values, ncolors=cmap.N, minval=minval, maxval=maxval, scale=scale)
        for bmu, val in zip(bmus, values):
            color = cmap(norm(val))
            axs[bmu].set_facecolor(color)
        self._make_colorbar(axs, norm, cmap, colorbar_label)

    def _make_colorbar(self, axs, norm, cmap, colorbar_label):
        if colorbar_label is None:
            colorbar_label = "Values"
        scalarmappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax_cb = axs[0,0].figure.add_axes([0.77,0.11,0.025,0.5])
        cax = plt.colorbar(scalarmappable, cax = ax_cb)
        ax_cb.set_ylabel(colorbar_label)
        if isinstance(norm, mpl.colors.BoundaryNorm): #Discrete colorbar
            self._displace_colorbar_ticks(ax_cb, norm)

    def _displace_colorbar_ticks(self, ax_cb, norm):
        ylims = ax_cb.get_ylim()
        ax_cb.tick_params(axis='y', which='minor', right=False)
        yticks = norm.boundaries
        # yticks = ax_cb.get_yticks()
        displaced_yticks = yticks[:-1] + 0.5 * np.diff(yticks)
        yticklabels = [str(yt) for yt in yticks[:-1]]
        # yticks = [yt + 0.5 for yt in yticks]
        ax_cb.set_yticks(displaced_yticks, yticklabels)
        ax_cb.set_ylim(ylims)
            
    def _colorNorm(self, values, ncolors, minval, maxval, scale):
        if minval is None: minval = min(values) 
        if maxval is None: maxval = max(values)
        bounds = self._colorbounds(values, minval, maxval, scale)
        if isinstance(values[0], int):
            norm = mpl.colors.BoundaryNorm(bounds, ncolors)
        else:
            norm = mpl.colors.Normalize(vmin = minval, vmax=maxval)
        return norm

    def _colorbounds(self,values, minval, maxval, scale):
        if minval == maxval:
            maxval =int(maxval) + 0.9999   #get a full (constant) range
            bounds = [minval, maxval]
        elif isinstance(values[0],int): 
            splitted_range = split_range_auto(minval, maxval, max_parts=10)
            bounds = splitted_range + [splitted_range[-1] + 1] # add an additional element so max value is correctly included
        else:
            bounds = np.linspace(minval, maxval, num = 10)
        return bounds

    def _groupby_bmu(self, loops, vals):
        bmus_vals = [(self.som.get_BMU(loop), val) for loop, val in zip(loops,vals)]
        bmu_vals_dict = defaultdict(list)
        for bmu, val in bmus_vals:
            bmu_vals_dict[bmu].append(val)

        return bmu_vals_dict
    
    def _aggregateby_bmu(self, bmu_vals_dict, agg_method: callable = np.median):    
        bmu_stat = {}
        for bmu, vals_list in bmu_vals_dict.items():
            bmu_stat[bmu] = agg_method(vals_list)
        return bmu_stat
    
    def HI_distr(self, axs = None, cmap = "coolwarm"):
        axs = self.som_map(axs)
        if isinstance(cmap,str):
            cmap = mpl.colormaps.get_cmap(cmap)
        
        loops = self.som.get_prototypes()
        loops = loops.reshape(-1, *loops.shape[-2:])
        his = [calc_HI(loop) for loop in loops]
        self.heat_map(loops, his, axs=axs,agg_method=np.mean, cmap = cmap, colorbar_label="Hysteresis Index")

    def samples(self, loops: np.ndarray, max_samples: int = 20, cmap = "inferno"):

        """ Plot samples according to the SOM prototype distribution
        Parameters
        ----------
        loops : np.ndarray
            Data array. The first dimension corresponds to the number of samples.

        max_samples: int (default = 20)
            Max number of samples per axes. Set max_samples = -1 to include all samples
        """        
        bmu_loops_dict = self._groupby_bmu(loops, loops)
        max_sample_per_bmu = max([len(sequence) for _, sequence in bmu_loops_dict.items()])
        nsamples_ax = min(max_sample_per_bmu, max_samples)
        nrows = round(np.sqrt(nsamples_ax))
        ncols = nsamples_ax // nrows + min(1, nsamples_ax % nrows)
        figsizeh = ncols * self.som.width
        figsizev = nrows * self.som.height
        fig, axs = self._make_figure(figsize = (figsizeh, figsizev))

        for ax in axs.flatten(): 
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines:
                ax.spines[spine].set_color("Gray")


        for (i, j), loops in bmu_loops_dict.items():
            ax = axs[i,j]
            
            for sample, (voffset, hoffset) in zip(loops, product(range(nrows), range(ncols))):
                ax.scatter(sample[:,0] + hoffset*1.2 , sample[:,1] - voffset*1.2 , c = range(len(sample)), cmap = cmap, s = 2)

            ax.set_xlim(-0.4, 1 + (ncols-1)*1.3)
            ax.set_ylim(-(nrows-1)*1.3, 1.1)
            
    
        self._add_map_coordinates(axs)

    def biplot(self, loops: np.ndarray, matrix: np.ndarray, regressor, labels = None, axs = None,
               corrtype: Literal["linear", "rank"] = "linear"):
        
        if axs is None:
            fig, axs = self._make_figure()
            self.heatmap_frequency(loops, axs = axs, type = "dots")
        if labels is None:
            labels = [f"var {i+1}" for i in range(matrix.shape[1])]
        
        for ax in axs.flatten():
            ax.collections[0].set_alpha(0.1)
        
        # set polar axes
        fig = axs[0,0].figure
        rec = self._get_axs_rec(axs)
        ax = fig.add_axes(rec, facecolor="none", polar = True)
        ax.spines["polar"].set_visible(False)
        ax.set_thetagrids([],[])
        
        # Get correlations
        bmus = [self.som.get_BMU(sample) for sample in loops]
        xycoords = np.array(bmus)
        output = []
        for variable, label in zip(matrix.T, labels):
            if corrtype == "rank":
                temp = variable.argsort()
                ranks = np.empty_like(temp)
                ranks[temp] = np.arange(len(variable))
                variable  = ranks 
            regressor.fit(xycoords, variable)
            ypred = regressor.predict(xycoords)
            r2 = (np.corrcoef(variable, ypred)[0,1]) ** 2
            _, m1, m2 = regressor.coefficients
            theta = self._calc_angle(m1, m2)
            output.append((label,theta, r2))
            if r2 > 0:
                ax.annotate("", xy = (theta, r2), xytext= (0,0), textcoords = "data", 
                        arrowprops=dict(facecolor='black', width = 1, headwidth = 5, headlength = 4))
            
                ax.annotate(label, xy = (theta, r2), xytext =  (0,-10), textcoords = "offset points", fontsize = 10)
            print(label, theta, r2)
        ax.set_rlim(0,1.0)
        return output

    def _calc_angle(self, m1, m2):
        theta = np.arctan(abs(m1/m2))

        if m1 >= 0 and m2 >= 0:
            angle = - theta
        elif m1 < 0 and m2 > 0:
            angle = theta
        elif m1 < 0 and m2 < 0:

            angle = np.pi-theta
        elif m1 > 0 and m2 < 0:
            angle = np.pi+theta
        
        return angle

    def _get_axs_rec(self, axs):
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, - np.inf
        for ax in axs.flatten():
            ax_xmin, ax_ymin= ax.get_position().min
            ax_xmax, ax_ymax= ax.get_position().max

            xmin = min(xmin, ax_xmin)
            ymin = min(ymin, ax_ymin)
            xmax = max(xmax, ax_xmax)
            ymax = max(ymax, ax_ymax)
        
        left = xmin
        bottom = ymin
        width = xmax - xmin
        height = ymax - ymin
        rec = left, bottom, width, height
        return rec

            