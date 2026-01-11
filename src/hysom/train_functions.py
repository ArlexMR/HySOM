import numpy as np
import warnings 
import numba as nb
from numba import prange

with warnings.catch_warnings():
     warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")
     from tslearn.metrics import dtw as tslearndtw


#Decay functions
def decay_linear(init_val, iter, max_iter, final_val):
     slope =  (init_val - final_val) / max_iter 
     return init_val - (slope * iter)

def decay_power(init_val, iter, max_iter, final_val):
     min_frac = final_val / init_val
     fraction = min_frac ** (iter / max_iter)
     return init_val * fraction

# Neighborhood functions
def gaussian(grid, center, sigma):
    
    distances = np.sqrt( (grid[0] - center[0])**2 + (grid[1] - center[1])**2 )
    neig_vals = np.exp( - distances ** 2 / (2 * sigma**2))
    return neig_vals

def bubble(grid, center, sigma):
     pass

# Distance Functions

def euclidean(prototypes, sample):
    dif_sqr = (prototypes - sample)**2
    return dif_sqr.sum(axis = (-1,-2))

def dtw_tslearn(prototypes, sample):
    return np.array([[tslearndtw(unit,sample) for unit in row] for row in prototypes])

@nb.njit
def njit_dtw(x, x_prime):
    R = np.zeros(shape = (len(x), len(x_prime)))
    for i in range(len(x)):
        for j in range(len(x_prime)):
            R[i, j] = _njit_local_sqr_dist(x[i], x_prime[j])
            if i > 0 or j > 0:
                R[i, j] += min(
                R[i-1, j  ] if i > 0             else np.inf,
                R[i  , j-1] if j > 0             else np.inf,
                R[i-1, j-1] if (i > 0 and j > 0) else np.inf
                )
    return (R[-1, -1])**(1/2)

@nb.njit
def _njit_local_sqr_dist(x1, x2):
    acum = 0.0
    for i in range(x1.shape[0]):
        acum += (x1[i] - x2[i])**2
    return acum

@nb.njit(parallel = True )
def dtw(prototypes, sample):
    distances = np.empty(prototypes.shape[:2])
    rows, columns = prototypes.shape[:2]
    for i in prange(rows):
        for j in prange(columns):
            distances[i,j] = njit_dtw(prototypes[i,j], sample)
    return distances
