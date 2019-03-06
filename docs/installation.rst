Installation
============

Via conda
---------

Precompiled libraries are available for Windows, Linux, and Mac via the conda package manager

Install `anaconda <https://www.anaconda.com/distribution/#download-section>`_ or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, add a couple channels to your config, then install pycudadecon:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda config --add channels talley
    $ conda install pycudadecon


GPU requirements
----------------

This software requires a CUDA-compatible NVIDIA GPU.  
The underlying libraries (llspylibs) have been compiled against different versions of the CUDA toolkit.  The required CUDA libraries are bundled in the conda distributions so you don't need to install the CUDA toolkit separately.  If desired, you can pick which version of CUDA you'd like based on your needs, but please note that different versions of the CUDA toolkit have different GPU driver requirements (the OS X build has only been compiled for CUDA 9.0).  To see which version you have installed currently, use `conda list llspylibs`, and to manually select a specific version of llspylibs:

======  ============  ==========  ============
 CUDA   Linux driver  Win driver  Install With
======  ============  ==========  ============
 10.0   ≥ 410.48      ≥ 411.31    ``conda install llspylibs=<version>=cu10.0``  
  9.0   ≥ 384.81      ≥ 385.54    ``conda install llspylibs=<version>=cu9.0``  
======  ============  ==========  ============

...where ``<version>`` is the version of llspylibs you'd like to install (use ``conda search llspylibs`` to list available versions)

If you run into trouble, feel free to `open an issue on github <https://github.com/tlambert03/pycudadecon/issues>`_ and describe your setup.


For development
---------------

If you'd like to contribute to pycudadecon, pull requests are welcome!  Minimally, you will want to have llspylibs installed in your conda environment.  Here's an example for creating a new environment (here: `pycdenv`), installing some dependencies, cloning the repo from github, then running the tests.

.. code-block:: bash

    $ conda create -n pycdenv -c talley llspylibs tifffile numpy pytest python=3.7
    $ git clone https://github.com/tlambert03/pycudadecon.git
    $ cd pycudadecon
    # run tests with pytest
    $ pytest
    # or with unittest
    $ python -m unittest discover test
