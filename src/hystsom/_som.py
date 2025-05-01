import numpy as np
from typing import Union, Callable
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from hystsom._validators import validate_constructor_params, validate_train_params, validate_codebooks_initialization
from hystsom._functions import decay_linear, decay_piecewise, decay_power
from hystsom._functions import gaussian, bubble, mexican_hat
from hystsom._functions import euclidean, dtw

decay_functions = {"power": decay_power,
                         "linear": decay_linear,
                         }

neighborhood_functions = {"gaussian": gaussian,
                          "bubble": bubble,
                          "mexican_hat": mexican_hat
                          }

distance_functions = {"euclidean": euclidean,
                      "dtw": dtw
                      }

class SOM:
    def __init__(self,
                width: int,
                height: int,
                input_dim: tuple,
                random_seed: int= None
                ):
        """
        Initialize Self Organizing Map

        Parameters
        ----------
        width, height : int
            Map dimensions

        input_dim : tuple 
            Shape of the input samples. Typically: (seq_len,2) where seq_len is the number of (x,y) coordinate points representing a loop

        initial_sigma: float, optional (default = sqrt(width * ndim))
            Neighborhood radius at the first iteration

        initial_learning_rate: float, optional (default = 1.0)
            Learning rate at the first iteration

        min_sigma: float, optional (default = 0.3)
            Neighborhood radius at the last iteration

        min_learning_rate: float, optional (default = 0.01)
            Learning rate at the last iteration

        decay_sigma_func: str or callable, optional (default = "power")
            Decay functions for the neighborhood radius. Available options: "power" or "linear"
            If callable, the provided function should receive four arguments: 
                init_val: initial_sigma, 
                iter: current iteration, 
                max_iter: maximum number of iterations, 
                min_val: minimum value 
            and return a number 
        
        decay_learning_rate_func: str or callable, optional (default = "power")
            same as for decay_sigma_func

        neighborhood_function: str or callable, optional (default = gaussian)
            Neighborhood function. Available options: "gaussian", "mexican_hat", "bubble"
            If callable, the provided functions should receive three arguments:
                
                grid: tuple of coordinate matrices as returned by numpy.meshgrid with matrix indexing convention: 
                    grid = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")

                center: tuple defining coordinates of the center. That is, where the neighborhood function
                        returns a value of 1.0 (peak value). Coordinates follow matrix convention (i,j)

                sigma: float defining the neighborhood radius  
            
                Returns: Matrix of neighborhood values of size (width, height)
                     
        """

        
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.random_seed = random_seed
        self._grid = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        self._rng = np.random.default_rng(self.random_seed)
        self._TE = []
        self._QE = []
        self._codebooks = None

    def random_init(self, data):

        """Initialize codebooks randomly from data
        """
        random_sample = self._rng.choice(data,self.width * self.height, replace = False)
        codeboox_dim = (self.height, self.width) + self.input_dim
        init_codebooks = random_sample.reshape(codeboox_dim)
        self.set_init_codebooks(init_codebooks)

    def set_init_codebooks(self, codebooks: np.ndarray):
        """Initialize codebooks
        """
        validate_codebooks_initialization(self.width, self.height, self.input_dim, codebooks)
        self._codebooks = codebooks

    def train(self, data: np.ndarray, 
              epochs: int, 
              random_order: bool = True,
              initial_sigma: float = None,
              initial_learning_rate: float = 1.0,
              min_sigma: float = 0.3,
              min_learning_rate: float = 0.01,
              decay_sigma_func: Union[str, callable]= "power",
              decay_learning_rate_func: Union[str, callable]=  "power",
              neighborhood_function: Union[str, callable]= "gaussian",
              distance_function: Union[str, callable] = "dtw", 
              track_errors: bool = False, 
              errors_sampling_rate: int = 4, 
              errors_data_fraction: float = 1.0,
              verbose: bool = False
              ):
        """
        Trains the Self-Organizing Map (SOM).

        Parameters
        ----------
        data : np.ndarray
            Data array. The first dimension corresponds to the number of samples.

        epochs : int
            The number of training iterations (epochs). Each data sample is fed to the map once every epoch.

        random_order : bool, optional (default=True)
            If True, samples are picked randomly without replacement. If False, they are fed sequentially.

        track_errors : bool, optional (default=False)
            If True, quantization error (QE) and topographic error (TE) will be computed during training. These values can be accessed using `self.get_QE_history()` and `self.get_TE_history()`.

        errors_sampling_rate : int, optional (default=4)
            If `track_errors` is True, this parameter controls how often errors are tracked. Errors will be tracked `errors_sampling_rate` times per epoch.            

        errors_data_fraction : float, optional (default=1.0)
            If `track_errors` is True, this parameter specifies the fraction of the data used to compute errors. 
            It should be between 0 and 1.0 (inclusive). If set to 1.0, all samples are used; if set to a value less than 1.0, the calculation is faster but uses fewer samples.

        verbose : bool or int, optional (default=False)
            If True, the status of the training process will be printed each epoch. 
            If int, this value represents the approximate number of times the status of the training process will be printed each epoch. 

        """
        
        if initial_sigma is None:
            initial_sigma = np.sqrt(self.width * self.height)

        validate_train_params(data, epochs, random_order, track_errors, 
                                    errors_sampling_rate, errors_data_fraction, verbose)
        # Validate decay functions
        for param, name in [(decay_sigma_func, 'decay_sigma'), 
                            (decay_learning_rate_func, 'decay_learning_rate')]:
            if not (isinstance(param, str) or callable(param)):
                raise TypeError(f"{name} must be either a string or a callable")

        # Validate neighborhood_function
        if not (isinstance(neighborhood_function, str) or callable(neighborhood_function)):
            raise TypeError("neighborhood_function must be a string or callable")

        # Validate distance_function
        if not (isinstance(distance_function, str) or callable(distance_function)):
            raise TypeError("distance_function must be a string or callable")

        if isinstance(decay_sigma_func, str):
            decay_sigma_func = decay_functions[decay_sigma_func]

        if isinstance(decay_learning_rate_func, str):
            decay_learning_rate_func = decay_functions[decay_learning_rate_func]
        
        if isinstance(neighborhood_function, str):
            neighborhood_function = neighborhood_functions[neighborhood_function]

        if isinstance(distance_function, str):
            distance_function = distance_functions[distance_function]    
        
        
        self.initial_sigma = initial_sigma
        self.initial_learning_rate = initial_learning_rate
        self.decay_sigma_func = decay_sigma_func
        self.decay_learning_rate_func = decay_learning_rate_func
        self.neighborhood_function = neighborhood_function
        self.distance_function = distance_function
        self.min_sigma = min_sigma
        self.min_learning_rate = min_learning_rate
        
        nsamples = len(data)

        if self._codebooks is None:
            self.random_init(data)

        if track_errors is False:
            samples_per_error = nsamples + 1 # out of reach value
        else:
            samples_per_error = max(1, int(nsamples / errors_sampling_rate))
            nsamples_error = max(1 , int(nsamples * errors_data_fraction))
        
        if verbose is False:
            samples_per_print = nsamples + 1 # out of reach value
        else:
            verbose = int(verbose)
            samples_per_print = max(1, int(nsamples / verbose))

        # Iteration indices
        max_iter, list_idxs = self._get_iteration_indices(epochs, random_order, nsamples)

        # Training loop
        iter = 0
        for epoch, idxs in enumerate(list_idxs):

            if track_errors: # Compute errors before first iteration
                self._track_errors(iter, data, nsamples_error)
            if verbose:
                self._print_epoch_summary(epoch+1, epochs)

            for inner_iter, idx in enumerate(idxs): 
                sample = data[idx]
                learning_rate = self.decay_learning_rate_func(self.initial_learning_rate, iter, max_iter, self.min_learning_rate)
                sigma = self.decay_sigma_func(self.initial_sigma, iter, max_iter, self.min_sigma)
                self.update(sample, learning_rate, sigma)

                if self._is_time_to_track_errors(inner_iter, samples_per_error):
                    self._track_errors(iter, data, nsamples_error)
                
                if self._is_time_to_print_training_status(inner_iter, samples_per_print):
                    self._print_training_status(inner_iter, nsamples)

                iter += 1
        self._print_finish_message()

    def _get_iteration_indices(self, epochs, random_order, nsamples):
        max_iter = nsamples * epochs
        list_idxs = [[i for i in range(nsamples)] for epoch in range(epochs)]
        if random_order:
            [self._rng.shuffle(indices) for indices in list_idxs]

        return max_iter,list_idxs
    
    def update(self, sample, learning_rate, sigma):
        """Update codebooks
        """
        bmu = self.get_BMU(sample)
        neighborhood_vals = self.neighborhood_function(self._grid, bmu, sigma)
        reshaped_nv = neighborhood_vals.repeat(self.input_dim[0] * self.input_dim[1]).reshape(self.height, self.width, self.input_dim[0], self.input_dim[1])
        self._codebooks += learning_rate * reshaped_nv * (sample - self._codebooks)
 
    def get_BMU(self, sample):
        """return BMU coordinates
        """ 
        distances = self.distance_function(self._codebooks, sample)
        return np.unravel_index(distances.argmin(), distances.shape)

    def quantization_error(self, data: np.ndarray):
        """Quantization error for each sample in data 
            Parameters
            ----------
            data : np.ndarray
                Collection of data samples with shape (nsamples, seq_len, 2)

            Returns  
            qe : list
                Quantization error for each data sample
        """
        return [self.distance_function(self._codebooks, sample).min() for sample in data]

    def topographic_error(self, data: np.ndarray):
        """Topographic error for each sample in data 
            Parameters
            ----------
            data : np.ndarray
                Collection of data samples with shape (nsamples, seq_len, 2)

            Returns  
            qe : list
                Topographic error for each data sample. 
        """
        distances = np.array([self.distance_function(self._codebooks, sample) for sample in data])
        xyindexes = [np.unravel_index(np.argpartition(dist_matrix.flatten(), (0,1))[[0,1]], (self.height, self.width)) for dist_matrix in distances]
        bmu_to_nextbmu_dists = [max(abs(X[0] - X[1]), abs(Y[0] - Y[1])) for X,Y in xyindexes]
        return (np.array(bmu_to_nextbmu_dists) > 1).astype(int).tolist()

    def get_QE_history(self):
        if self._QE:
            t, qe = zip(*self._QE)
        else: 
            t = qe = None
        return t, qe

    def get_TE_history(self):
        if self._QE:
            t, te = zip(*self._TE)
        else: 
            t = te = None
        return t, te

    def get_prototypes(self):
        return self._codebooks
    
    def get_frequencies(self, data):
        """
        Return activation frequency for each prototype 
        
        Parameters
        ----------
            data : np.ndarray
                Collection of data samples with shape (nsamples, seq_len, 2)
        
        Returns
        """

    def _track_errors(self, iter, data, nsamples_error):
        subset = self._rng.choice(data, size = nsamples_error, replace=False)
        qe, te = self._compute_errors_fast(subset)
        self._QE.append((iter, qe))
        self._TE.append((iter, te))

    def _compute_errors_fast(self, data):
        distances = np.array([self.distance_function(self._codebooks, sample) for sample in data])
        qe = distances.min(axis = (1,2)).mean() 

        xyindexes = [np.unravel_index(np.argpartition(dist_matrix.flatten(), (0,1))[[0,1]], (self.height, self.width)) for dist_matrix in distances]
        bmu_to_nextbmu_dists = [max(abs(X[0] - X[1]), abs(Y[0] - Y[1])) for X,Y in xyindexes]
        te = (np.array(bmu_to_nextbmu_dists) > 1).sum() / len(data)
        return qe, te 

    def _is_time_to_track_errors(self, inner_iter, samples_per_error):
        return (inner_iter+1) % samples_per_error == 0     
            
    def _is_time_to_print_training_status(self, inner_iter, samples_per_print):
        return (inner_iter+1) % samples_per_print == 0

    def _print_training_status(self, inner_iter, nsamples):
        print(f"[{inner_iter+1}/{nsamples}] {100 * (inner_iter+1) / nsamples:.0f}%")

    def _print_epoch_summary(self, epoch, nepochs):

        qe, te = self._get_last_errors_for_printing()
        print(f"{'='*60}")
        print(f"Epoch: {epoch}/{nepochs} - Quant. Error: {qe} - Topo. Error: {te}")
    
    def _print_finish_message(self):
        qe, te = self._get_last_errors_for_printing()
        print(f"{'='*60}")
        print(f"Training Completed! - Quant. Error: {qe} - Topo. Error: {te}")
        print(f"{'='*60}")        

    def _get_last_errors_for_printing(self):
        _,qe = self.get_QE_history()
        _,te = self.get_TE_history()
        if qe:
            qe = round(qe[-1],2)
        else: 
            qe = "--"
        if te:
            te = round(te[-1],2)
        else:
            te = "--"
        
        return qe, te