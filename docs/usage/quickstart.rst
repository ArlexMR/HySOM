Quickstart
===========

This section provides tutorials to help you get started with the HySOM package. Designed for easy learning, they offer step-by-step guidance on using the package.  
Below is a minimal example demonstrating how to train a Self-Organizing Map (SOM) with default hyperparameters on a sample dataset and visualize the results.


✨ Quick Example

.. code-block:: python

   from hysom import HSOM
   from hysom.utils.datasets import get_sample_data
   from hysom.utils.plots import plot_map

   # Get sample data
   data = get_sample_data()

   # Train SOM
   som = HSOM(width=8, height=8, input_dim = data.shape[1:])
   som.train(data, epochs = 5)

   # Visualize results
   prototypes = som.get_prototypes() 
   _ = plot_map(prototypes)

For a detailed explanation of the HSOM class, including diagnosing the training process using topographic and quantization errors refer to the following tutorial.

.. toctree::
   :maxdepth: 1

   The HSOM class <../tutorials/thesomclass>

To explore the visualization functions available in the package, refer to the following tutorial.

.. toctree::
   :maxdepth: 1

   Visualize your SOM <../tutorials/vizfunctions>