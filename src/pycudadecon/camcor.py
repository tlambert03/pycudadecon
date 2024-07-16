"""Flash residual pixel artifact correction."""

import numpy as np

from . import lib


def quickCamcor(imstack: np.ndarray, camparams: np.ndarray) -> np.ndarray:
    """Correct Flash residual pixel artifact on GPU."""
    if not imstack.ndim == 3:
        raise ValueError("imstack must be 3D")
    camcor_init(imstack.shape, camparams)  # type: ignore
    return camcor(imstack)


def camcor_init(rawdata_shape: tuple[int, int, int], camparams: np.ndarray) -> None:
    """Initialize camera correction on GPU.

    shape is nz/ny/nx of the concatenated stacks from a single timepoint.
    """
    nz, ny, nx = rawdata_shape
    if not np.issubdtype(camparams.dtype, np.float32):
        camparams = camparams.astype(np.float32)
    lib.camcor_interface_init(nx, ny, nz, camparams)


def camcor(imstack: np.ndarray) -> np.ndarray:
    """Perform residual pixel artifact correction on GPU."""
    if not np.issubdtype(imstack.dtype, np.uint16):
        imstack = imstack.astype(np.uint16)
    nz, ny, nx = imstack.shape
    result = np.empty_like(imstack)
    lib.camcor_interface(imstack, nx, ny, nz, result)
    return result
