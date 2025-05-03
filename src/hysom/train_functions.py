import numpy as np
import warnings 

with warnings.catch_warnings():
     warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")
     from tslearn.metrics import dtw as tslearndtw


#Decay functions
def decay_linear(init_val, iter, max_iter, min_val):
     slope =  (init_val - min_val) / max_iter 
     return init_val - (slope * iter)

def decay_power(init_val, iter, max_iter, min_val):
     min_frac = min_val / init_val
     fraction = min_frac ** (iter / max_iter)
     return init_val * fraction

def decay_piecewise(init_val, iter, max_iter, min_val):
     pass

# Neighborhood functions
def gaussian(grid, center, sigma):
    
    distances = np.sqrt( (grid[0] - center[0])**2 + (grid[1] - center[1])**2 )
    neig_vals = np.exp( - distances ** 2 / (2 * sigma**2))
    return neig_vals

def bubble(grid, center, sigma):
     pass

def mexican_hat(grid, center, sigma):
     pass

# Distance Functions

def euclidean(prototypes, sample):
    dif_sqr = (prototypes - sample)**2
    return dif_sqr.sum(axis = (-1,-2))
     

def dtw(prototypes, sample):
    #return np.array([[dtw(unit,x) for unit in row] for row in prototypes])
    return np.array([[tslearndtw(unit,sample) for unit in row] for row in prototypes])

