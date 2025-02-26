from typing import Union, Callable
import numpy as np
def validate_constructor_params(hdim, vdim, input_dim, decay_sigma, decay_learning_rate, neighborhood_function, distance_function, min_sigma, min_learning_rate):

        if not isinstance(input_dim, tuple):
             raise TypeError("input_dim must be a tuple")  
           
        # Validate decay functions
        for param, name in [(decay_sigma, 'decay_sigma'), 
                            (decay_learning_rate, 'decay_learning_rate')]:
            if not (isinstance(param, str) or callable(param)):
                raise TypeError(f"{name} must be either a string or a callable")

        # Validate neighborhood_function
        if not (isinstance(neighborhood_function, str) or callable(neighborhood_function)):
            raise TypeError("neighborhood_function must be a string or callable")

        # Validate distance_function
        if not (isinstance(distance_function, str) or callable(distance_function)):
            raise TypeError("distance_function must be a string or callable")


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
        

def validate_codebooks_initialization(width, height, input_dim, codebooks):
    if not isinstance(codebooks, np.ndarray):
         raise TypeError("codebooks must be a np.ndarray")

    if codebooks.shape != (height, width) + input_dim:
         raise ValueError(f"'codebooks' dimension mismatch. 'codebooks' should be a (height, width, input_dim): ({(height, width) + input_dim}) numpy array instead of {codebooks.shape}")
         