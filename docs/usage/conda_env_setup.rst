:orphan:

Setting up a python environment with conda
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

   conda create --name hysom-env python

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

A better option would be too use `pip` to install all your packages. In that case skip this step

Note that `numpy` and `matplotlib` will be installed with `HySOM` in the next step, so you don't have to install them here. 

Step 5: Install HySOM
----------------------
Now, you can install **HySOM**:

.. code-block:: bash

   pip install hysom

`HySOM` is now set up and ready to use in your conda environment `hysom-env`. However, you might want to install additional libraries:`

(optional) Step 6: Install IPykernel and other libraries using pip
-------------------------------------------------------------------
If you want to use jupyter notebooks, you also want to install the IPykernel:

.. code-block:: bash

   pip install ipykernel

Also, if you haven't done it already, install pandas. It is not required for HySOM but you'll likely need it, especially if you want to replicate the tutorials in this documentation:

.. code-block:: bash

   pip install pandas

Your conda environment is now ready!
