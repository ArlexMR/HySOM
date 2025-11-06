Quickstart
===========

Here's a minimal working example of using HySOM for training and visualizing an SOM 

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

.. image:: ../images/SOM_example.png
  :width: 400
  :alt: Trained SOM