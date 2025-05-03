from typing import Union, Callable
import numpy as np



def validate_train_params(data, epochs, random_order, 
                               track_errors, errors_sampling_rate, 
                               errors_data_fraction, verbose):
        # Validate data type
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")

        # Validate epochs
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        # Validate errors_data_fraction
        if not (0 <= errors_data_fraction <= 1):
            raise ValueError("errors_data_fraction must be between 0 and 1")

        # Validate errors_sampling_rate
        if (not isinstance(errors_sampling_rate, int)) or (errors_sampling_rate <= 0):
            raise ValueError("errors_sampling_rate must be a positive integer")
        
        if (not isinstance(verbose, (bool, int))) or (type(verbose) in (int,float) and verbose <=0):
            raise ValueError(f"verbose must be bool or int > 0, not {type(verbose)} = {verbose}") 
        

def validate_prototypes_initialization(width, height, input_dim, prototypes):
    if not isinstance(prototypes, np.ndarray):
         raise TypeError("prototypes must be a np.ndarray")

    if prototypes.shape != (height, width) + input_dim:
         raise ValueError(f"'prototypes' dimension mismatch. 'prototypes' should be a (height, width, input_dim): ({(height, width) + input_dim}) numpy array instead of {prototypes.shape}")
         