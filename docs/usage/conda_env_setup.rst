:orphan:

Setting Up a Python Environment with Conda
===========================================

Before installing **HySOM**, it is recommended to set up an isolated Python environment using `conda`. This ensures dependency management and avoids conflicts with other packages.

.. _conda-setup:

Step 1: Install Conda
----------------------
If you haven't installed `conda`, download and install **Miniconda** or **Anaconda** from:

- `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_
- `Anaconda <https://www.anaconda.com/download>`_

Step 2: Create a New Environment
---------------------------------
Once installed, create a new `conda` environment with Python. Open the `Anaconda Powershell Prompt` and run:

.. code-block:: bash

   conda create --name hysom-env python=3.13

.. note::
   Make sure to install a python version <= 3.13 (the previous line installs python 3.13). `HySOM` is not yet compatible with Python 3.14 

Step 3: Activate the Environment
-----------------------------------
Activate the newly created environment:

.. code-block:: bash

   conda activate hysom-env

(optional) Step 4: Install required `conda` libraries  
-------------------------------------------------------
It is discouraged to combine `pip` and `conda` when installing packages in an environment (`see more details here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment>`_). `HySOM` will be installed using `pip`, so, if you need to install any conda package, do it before installing `HySOM` or any other pip package. For example, if you need to install `Pandas` using conda, do it now:  

.. code-block:: bash

   conda install pandas

A better option would be too use `pip` to install all your packages.   

Note that `numpy`, `matplotlib`, `scikit-learn` and `tslearn` will be installed with `HySOM` in the next step, so you don't have to install them here. 

Step 5: Install HySOM
----------------------
Finally, install **HySOM**:

.. code-block:: bash

   pip install hysom

Your environment is now set up and ready to use! 

