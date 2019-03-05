pyCUDAdecon: C++/CUDA-accelerated deconvolution with a python wrapper
=====================================================================

This package provides a python wrapper and convenience functions for `cudaDeconv <https://github.com/dmilkie/cudaDecon>`_ (which is a CUDA/C++ implementation of an accelerated Richardson Lucy Deconvolution).  cudaDeconv was originally written by `Lin Shao <https://github.com/linshaova>`_ and modified by `Dan Milkie <https://github.com/dmilkie>`_, at Janelia Research campus.  This package makes use of a shared library interface that I wrote for cudaDecon while developing `LLSpy <https://github.com/tlambert03/LLSpy>`_, that adds a couple additional kernels for affine transformations and camera corrections.  The code here is mostly extracted from that package to allow it to be used independently of LLSpy.

The primary function is 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   deconvolution



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
