import numpy as np

from . import lib


def quickCamcor(imstack, camparams):
    """Correct Flash residual pixel artifact on GPU"""
    camcor_init(imstack.shape, camparams)
    return camcor(imstack)


def camcor_init(rawdata_shape, camparams):
    """initialize camera correction on GPU.
    shape is nz/ny/nx of the concatenated stacks from a single timepoint
    """
    nz, ny, nx = rawdata_shape
    if not np.issubdtype(camparams.dtype, np.float32):
        camparams = camparams.astype(np.float32)
    lib.camcor_interface_init(nx, ny, nz, camparams)


def camcor(imstack):
    if not np.issubdtype(imstack.dtype, np.uint16):
        imstack = imstack.astype(np.uint16)
    nz, ny, nx = imstack.shape
    result = np.empty_like(imstack)
    lib.camcor_interface(imstack, nx, ny, nz, result)
    return result
