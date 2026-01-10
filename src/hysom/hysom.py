import numpy as np
from typing import Union, Tuple, List, Callable
from collections import defaultdict
from hysom.validators import validate_train_params, validate_prototypes_initialization
from hysom.train_functions import decay_linear, decay_power, gaussian, bubble, euclidean, dtw
from hysom.utils.aux_funcs import resolve_function

decay_functions_map = {"power": decay_power,
                    "linear": decay_linear,
                         }

neighborhood_functions_map = {"gaussian": gaussian,
                          "bubble": bubble,
                          }

distance_functions_map = {"euclidean": euclidean,
                      "dtw": dtw
                      }

class HSOM:
    """
    Self-Organizing Map (SOM) for 2D time series data.

    Parameters
    ----------
    width : int
        Number of units along the width of the map.

    height : int
        Number of units along the height of the map.

    input_dim : tuple of int
        Shape of the input samples. Typically `(seq_len, 2)`, where `seq_len` is the number 
        of (x, y) coordinate points representing a sample.

    random_seed : int, optional
        Ensures reproducibility. If None, results may vary each time due to random elements 
        in the training process. Default is None.
    """
    def __init__(self,
                width: int,
                height: int,
                input_dim: tuple,
                random_seed: int | None | None= None
                ):

        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.random_seed = random_seed
        self._grid = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        self._rng = np.random.default_rng(self.random_seed)
        self._TE = []
        self._QE = []
        self._prototypes = None

    def random_init(self, data: np.ndarray):

        """Initialize prototypes randomly from data

        Parameters
        ----------
        data : np.ndarray
            Data
        """
        random_sample = self._rng.choice(data,self.width * self.height, replace = False)
        prototypes_dim = (self.height, self.width) + self.input_dim
        init_prototypes = random_sample.reshape(prototypes_dim)
        self.set_init_prototypes(init_prototypes)

    def set_init_prototypes(self, prototypes: np.ndarray):
        """
        Initialize prototypes.

        Parameters
        ----------
        prototypes : np.ndarray
            The shape of `prototypes` must be consistent with the SOM dimensions and input dimensions (`input_dim`):  
            `prototypes.shape = (height, width, seq_len, 2)`.
        """

        validate_prototypes_initialization(self.width, self.height, self.input_dim, prototypes)
        self._prototypes = prototypes

    def train(self, data: np.ndarray, 
              epochs: int, 
              random_order: bool = True,
              initial_sigma: float | None= None,
              initial_learning_rate: float = 1.0,
              final_sigma: float = 0.3,
              final_learning_rate: float = 0.01,
              decay_sigma_func: Union[str, Callable]= "power",
              decay_learning_rate_func: Union[str, Callable]=  "power",
              neighborhood_function: Union[str, Callable]= "gaussian",
              distance_function: Union[str, Callable] = "dtw", 
              track_errors: bool = False, 
              errors_sampling_rate: int = 4, 
              errors_data_fraction: float = 1.0,
              verbose: bool | int| int= False
              ):
        """
        Trains the Self-Organizing Map (SOM).

        Parameters
        ----------
        data : np.ndarray
            Data array. The first dimension corresponds to the number of samples. Second and third dimensions must be consistent with `input_dim`

        epochs : int
            Defines the number of training iterations (`total_iterations = number_of_samples * epochs`). Each data sample is fed to the map once every epoch.

        random_order : bool, optional (default=True)
            If True, samples are picked randomly without replacement. If False, they are fed sequentially.
        
        initial_sigma : float, optional (default: sqrt(width * height))
            Neighborhood radius at the first iteration.

        initial_learning_rate : float, optional (default: 1.0)
            Learning rate at the first iteration.

        final_sigma : float, optional (default: 0.3)
            Neighborhood radius at the last iteration.

        final_learning_rate : float, optional (default: 0.01)
            Learning rate at the last iteration.

        decay_sigma_func : str or callable, optional (default: "power")
            Decay function for the neighborhood radius. Defines how the neighborhood radius change from `initial_sigma` to `final_sigma`.   
            Available options: `"power"`, `"linear"`.  
            If callable, the function should accept four arguments:
            
                - `init_val` (float): Initial neighborhood radius.
                - `iter` (int): Current iteration.
                - `max_iter` (int): Maximum number of iterations.
                - `final_val` (float): Minimum radius value.  
                
            The function must return a numeric value.  

            Examples
            --------
            >>> def decay_linear(init_val, iteration, max_iter, final_val):
            >>>    slope =  (init_val - final_val) / max_iter 
            >>>    return init_val - (slope * iteration)

            See the Tutorials for additional details

        decay_learning_rate_func : str or callable, optional (default: "power")
            Same format as `decay_sigma_func`, but applied to the learning rate.

        neighborhood_function : str or callable, optional (default: "gaussian")
            Defines the neighborhood function.   

            Available options: `"gaussian"`, `"bubble"`.  
            If callable, the function should accept three arguments:

                - `grid` (tuple of numpy arrays): Coordinate matrices as returned by `numpy.meshgrid` using matrix indexing convention:  
                - `center` (tuple): Coordinates where the function returns 1.0 (peak value), using `(i, j)` matrix convention.
                - `sigma` (float): Defines the neighborhood radius.  

            The function should return a matrix of neighborhood values with shape `(width, height)`.
            See the Tutorials for additional details

        distance_function : str or callable, optional (default: "dtw")
            Defines the distance function used to identify the BMU.  

            Available options: `"dtw"`, `"euclidean"`.   

            If callable, the function should accept two arguments:

                - `prototypes` (np.ndarray): prototypes array as returned by `get_prototypes()`.
                - `sample` (np.ndarray): a sample data of shape `input_dim`.  

            The function should return an `np.array` of shape (`width`, `height`, `seq_len`, 2)  containing the distance from `sample` to each prototype

        track_errors : bool, optional (default=False)
            If True, quantization error (QE) and topographic error (TE) will be computed during training. These values can be accessed using `get_QE_history()` and `get_TE_history()`.

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

        validate_train_params(data, epochs,errors_sampling_rate, errors_data_fraction, verbose)
        
        self.initial_sigma = initial_sigma
        self.initial_learning_rate = initial_learning_rate
        self.final_sigma = final_sigma
        self.final_learning_rate = final_learning_rate
        self.decay_sigma_func = resolve_function(decay_sigma_func, decay_functions_map)
        self.decay_learning_rate_func = resolve_function(decay_learning_rate_func, decay_functions_map)
        self.neighborhood_function = resolve_function(neighborhood_function, neighborhood_functions_map)
        self.distance_function = resolve_function(distance_function, distance_functions_map)
        nsamples = len(data)

        if self._prototypes is None:
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
                learning_rate = self.decay_learning_rate_func(self.initial_learning_rate, iter, max_iter, self.final_learning_rate)
                sigma = self.decay_sigma_func(self.initial_sigma, iter, max_iter, self.final_sigma)
                self._update(sample, learning_rate, sigma)

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
    
    def _update(self, sample, learning_rate, sigma):

        bmu = self.get_BMU(sample)
        neighborhood_vals = self.neighborhood_function(self._grid, bmu, sigma)
        reshaped_nv = neighborhood_vals.repeat(self.input_dim[0] * self.input_dim[1]).reshape(self.height, self.width, self.input_dim[0], self.input_dim[1])
        self._prototypes += learning_rate * reshaped_nv * (sample - self._prototypes)
 
    def get_BMU(self, sample: np.ndarray) -> Tuple:
        """
        Return BMU coordinates for a given `sample`, following matrix notation: `(row, col)`.

        Parameters
        ----------
        sample : np.ndarray
            Input sample with shape `(sequence_length, 2)`.

        Returns
        -------
        Tuple
            Coordinates of the Best Matching Unit `(row, col)`.
        """

        distances = self.distance_function(self._prototypes, sample)
        unraveled = np.unravel_index(distances.argmin(), distances.shape)
        return tuple(int(x) for x in unraveled)
    
    def get_distance_to_bmu(self, sample: np.ndarray) -> float:
        """
        Return the distance to the BMU for a given `sample`.

        Parameters
        ----------
        sample : np.ndarray
            Input sample with shape `(sequence_length, 2)`.

        Returns
        -------
        float
            distance to the BMU.
        """
        return float(self.distance_function(self._prototypes, sample).min() )
    
    def classify(self, samples: np.ndarray) -> dict[tuple, list]:
        """
        Assign each sample in `samples` to its Best Matching Unit (BMU).

        Parameters
        ----------
        samples : np.ndarray
            Array of input samples with shape `(n_samples, seq_len, n_features)`.
            For this SOM implementation, `n_features` is typically 2.

        Returns
        -------
        dict[tuple, list]
            A dictionary mapping BMU coordinates (row, col) to a list of samples
            whose BMU corresponds to that coordinate. Each key is a tuple
            representing the BMU position on the SOM grid, and each value is the
            list of samples assigned to that node.
        """
        out = defaultdict(list)
        for sample in samples:
            bmu = self.get_BMU(sample)
            out[bmu].append(sample)
        return out

    def quantization_error(self, data: np.ndarray) -> List:
        """
        Compute the quantization error for each sample in `data`.

        Parameters
        ----------
        data : np.ndarray
            Collection of data samples with shape `(nsamples, seq_len, 2)`.

        Returns
        -------
        List
            Quantization error for each data sample.
        """
        return [self.distance_function(self._prototypes, sample).min() for sample in data]

    def topographic_error(self, data: np.ndarray) -> List:
        """
        Compute the topographic error for each sample in `data`.

        Parameters
        ----------
        data : np.ndarray
            Collection of data samples with shape `(nsamples, seq_len, 2)`.

        Returns
        -------
        List
            Topographic error for each data sample.
        """
        distances = np.array([self.distance_function(self._prototypes, sample) for sample in data])
        xyindexes = [np.unravel_index(np.argpartition(dist_matrix.flatten(), (0,1))[[0,1]], (self.height, self.width)) for dist_matrix in distances]
        bmu_to_nextbmu_dists = [max(abs(X[0] - X[1]), abs(Y[0] - Y[1])) for X,Y in xyindexes]
        return (np.array(bmu_to_nextbmu_dists) > 1).astype(int).tolist()

    def get_QE_history(self) -> Tuple:
        """
        Get the average quantization error across iterations.

        Only available if `track_errors` is set to `True` during training.

        Returns
        -------
        iteration : List
            Iteration indices.

        QE : List
            Average quantization error values.

        """

        if self._QE:
            t, qe = zip(*self._QE)
        else: 
            t = qe = None
        return t, qe

    def get_TE_history(self)-> Tuple:
        """
        Get the average topographic error across iterations.

        Only available if `track_errors` is set to `True` during training.

        Returns
        -------
        iteration : List
            Iteration indices.

        TE : List
            Average topographic error values.

        """
        if self._QE:
            t, te = zip(*self._TE)
        else: 
            t = te = None
        return t, te

    def get_prototypes(self, bmu:tuple[int,int] | None = None) -> np.ndarray:
        """
        Get the prototypes.

        Returns
        -------
        np.ndarray
            prototypes array.
        """
        
        if self._prototypes is None:
            raise AttributeError("Prototypes haven't been initialized for this HSOM")
        
        if bmu:
            return self._prototypes[bmu]

        return self._prototypes
    
    def attribute_matrix(self, data: np.ndarray,
                      attribute: np.ndarray,
                      agg_method: Callable[[List], float] = np.median) -> np.ndarray:
        """
        Create an attribute matrix based on the provided data and attribute values.

        Parameters
        ----------
        data : np.ndarray
            Collection of data samples with shape `(nsamples, seq_len, 2)`.

        attribute : np.ndarray
            Attribute values corresponding to each sample in `data`.

        agg_method : Callable, optional (default=np.median)
            Aggregation method to apply to the attribute values for each BMU.

        Returns
        -------
        np.ndarray
            Attribute map with shape `(height, width)`.
        """
        
        bmus = [self.get_BMU(sample) for sample in data]
        bmu_to_attr = {bmu: [] for bmu in set(bmus)}
        
        for bmu, attr in zip(bmus, attribute):
            bmu_to_attr[bmu].append(attr)

        attr_map = np.empty((self.height, self.width))
        attr_map.fill(np.nan)

        for bmu, attrs in bmu_to_attr.items():
            attr_map[bmu] = agg_method(attrs)

        return attr_map
    
    def frequency_matrix(self, data: np.ndarray, relative = False) -> np.ndarray:
        """
        Create a frequency matrix based on the provided data.

        Parameters
        ----------
        data : np.ndarray
            Collection of data samples with shape `(nsamples, seq_len, 2)`.

        Returns
        -------
        np.ndarray
            Frequency matrix with shape `(height, width)`.
        """
        freq_matrix = self.attribute_matrix(data = data, attribute= np.ones(len(data)), agg_method=sum)
        freq_matrix[np.isnan(freq_matrix)] = 0  # Replace NaN values with 0
        if relative:
            freq_matrix = freq_matrix / freq_matrix.sum()  # Normalize to [0, 1]
        return freq_matrix

    def _track_errors(self, iter, data, nsamples_error):
        subset = self._rng.choice(data, size = nsamples_error, replace=False)
        qe, te = self._compute_errors_fast(subset)
        self._QE.append((iter, qe))
        self._TE.append((iter, te))

    def _compute_errors_fast(self, data):
        distances = np.array([self.distance_function(self._prototypes, sample) for sample in data])
        qe = distances.min(axis = (1,2)).mean() 

        xyindexes = [np.unravel_index(np.argpartition(dist_matrix.flatten(), (0,1))[[0,1]], (self.height, self.width)) for dist_matrix in distances]
        bmu_to_nextbmu_dists = [max(abs(X[0] - X[1]), abs(Y[0] - Y[1])) for X,Y in xyindexes]
        te = (np.array(bmu_to_nextbmu_dists) > 1).sum() / len(data)
        return float(qe), float(te) 

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