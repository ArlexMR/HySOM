:orphan:

Setting up a python environment with uv
===========================================

Step 1: Install uv
----------------------
If you haven't installed `uv`, install it following the `official instructions <https://docs.astral.sh/uv/getting-started/installation/>`_

Step 2: Create a new project 
---------------------------------
Open your preferred terminal and run:

.. code-block:: bash

   uv init --python 3.13

.. note::
   Make sure to install a python version <= 3.13 (the previous line installs python 3.13). `HySOM` is not yet compatible with Python 3.14 

Step 3: install hysom
-----------------------------------

.. code-block:: bash

   uv add hysom

Your environment is now set up and ready to use. However, you might want to install additional libraries:`

(optional) Step 4: Install IPykernel and other libraries 
-------------------------------------------------------------------
If you want to use jupyter notebooks (or notebooks within vscode), you also want to install the IPykernel:

.. code-block:: bash

   uv add --dev ipykernel

Also, if you haven't done it already, install pandas. It is not required for HySOM but you'll likely need it, especially if you want to replicate the tutorials in this documentation:

.. code-block:: bash

   pip install pandas

Your virtual environment is now ready to use!
