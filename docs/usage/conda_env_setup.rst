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

   conda create --name hysom-env python

Step 3: Activate the Environment
---------------------------------
Activate the newly created environment:

.. code-block:: bash

   conda activate hysom-env

Step 4: Install HySOM
----------------------
Finally, install **HySOM**:

.. code-block:: bash

   pip install hysom

Your environment is now set up and ready to use! 

