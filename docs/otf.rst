Creating Optical Transfer Functions
===================================

These are functions for converting a 3D point spread function (PSF) volume into a radially averaged 2D complex Optical Transfer Function (OTF) that can be used for deconvolution.  You can either write the OTF to file, for later use, using the :func:`pycudadecon.makeotf` function, or use the :class:`pycudadecon.TemporaryOTF` context manager to create and delete a temporary OTF from a 3D PSF input.

.. automodule:: pycudadecon
    :members: makeotf, TemporaryOTF