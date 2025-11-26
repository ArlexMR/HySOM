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

Your environment is now set up and ready to use! 

