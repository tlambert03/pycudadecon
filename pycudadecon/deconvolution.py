from .libcudawrapper import RLContext, rl_decon
from .otf import TemporaryOTF
import os
import tifffile as tf
from fnmatch import fnmatch
import numpy as np


def _yield_arrays(images, fpattern='*.tif'):
    """Accepts a numpy array, a filepath, a directory, or a list of these
    and returns a generator that yields numpy arrays.

    fpattern argument is used to filter files in a directory
    """
    if isinstance(images, np.ndarray):
        yield images

    elif isinstance(images, str):
        if os.path.isfile(images):
            yield tf.imread(images)

        elif os.path.isdir(images):
            imfiles = [f for f in os.listdir(images) if fnmatch(f, fpattern)]
            if not len(imfiles):
                raise IOError('No files matching pattern "{}" found in directory: {}'
                              .format(fpattern, images))
            for fpath in imfiles:
                yield tf.imread(os.path.join(images, fpath))

    elif isinstance(images, (list, tuple)):
        for item in images:
            yield from _yield_arrays(item)  # noqa


def decon(images, psf, fpattern='*.tif', **kwargs):
    """Deconvolve an image or images with a PSF or OTF file

    Args:
        images (str, np.ndarray, list, tuple): The array, filepath,
            directory, or list/tuple thereof to deconvolve
        psf (str, np.ndarray): a filepath of a PSF or OTF file, or a 3D numpy
            PSF array.  Function will auto-detect whether the file is a 3D PSF
            or a filepath representing a 2D complex OTF.
        fpattern (str, optional): Defaults to '\*.tif'. Filepattern to use when
            a directory is provided in the ``images`` argument
        ** kwargs: optional keyword arguments listed below
        ** dxdata (float): The xy pixel size of the ``image``. Defaults to 0.1
        ** dzdata (float): The z step size of the ``image``. Defaults to 0.5
        ** dxpsf (float): The xy pixel size of the ``psf``. Defaults to 0.1
        ** dzpsf (float): The z step size of the ``psf``. Defaults to 0.1
        ** background (int): Background to subtract.  use 'auto' to subtract
            median val of last Z plane. Defaults to 80
        ** deskew (float): Angle to deskew data (for stage scanning acquisition).
            Defaults to 0
        ** pad_val (int): Value to pad edges with when deskewing. Should be
            zero when ``background`` is 'auto' Defaults to 0
        ** rotate (float): Degrees to rotate volume in Y axis (to make Z axis orthogonal to coverslip). Defaults to 0
        ** width (int): Width of output image (0 = full). Defaults to 0
        ** n_iters (int): Number of iterations in deconvolution Defaults to 10
        ** save_deskewed (bool): Save raw deskewed files (if deskew > 0). Defaults to False
        ** napodize (int): Number of pixels to soften edge with. Defaults to 15
        ** nz_blend (int): Number of top and bottom sections to blend in to reduce axial ringing. Defaults to 0
        ** dup_rev_z (bool): Duplicate reversed stack prior to decon to reduce axial ringing. Defaults to False
        ** wavelength (int): Wavelength in nanometers (for OTF cleanup). Defaults to 520
        ** fixorigin (int):  For all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0. Defaults to 10
        ** otf_bgrd (None, int): Background to subtract in PSF (None = autodetect). Defaults to None
        ** na (float): Numerical aperture (for OTF cleanup). Defaults to 1.25]
        ** nimm (float): Refractive index of medium (for OTF cleanup). Defaults to 1.3
        ** krmax (int): Pixels outside this limit will be zeroed (overwriting estimated value from ``na`` and ``nimm``). Defaults to 0
        ** cleanup_otf (bool): Clean up OTF outside of OTF support. Defaults to False

    Raises:
        ValueError: If save_deskewed is True and deskew is unset or 0
        IOError: If a directory is provided as input and ``fpattern`` yields no files
        NotImplementedError: If ``psf`` is provided as a complex, 2D numpy array
            (OTFs can only be provided as filenames created with ``makeotf``)

    Returns:
        np.ndarray, list: numpy array or list of arrays (deconvolved images)

        if ``save_deskewed == True``, returns a tuple (decon, deskewed) or a list
        of tuples (if input was iterable)

    """
    if kwargs.get('save_deskewed'):
        if kwargs.get('deskew', 1) == 0:
            raise ValueError('Cannot use save_deskewed=True with deskew=0')
        if kwargs.get('deskew', 0) == 0:
            raise ValueError('Must set deskew != 0 when using save_deskewed=True')

    out = []
    with TemporaryOTF(psf, **kwargs) as otf:
        arraygen = _yield_arrays(images)
        # first, assume that all of the images are the same shape...
        # in which case we can prevent a lot of GPU IO
        # grab and store the shape of the first item in the generator
        next_im = next(arraygen)
        shp = next_im.shape

        with RLContext(shp, otf.path, **kwargs) as ctx:
            while True:
                out.append(rl_decon(next_im, output_shape=ctx.out_shape, **kwargs))
                try:
                    next_im = next(arraygen)
                    # here we check to make sure that the images are still the same
                    # shape... if not, we'll continue below
                    if not next_im.shape == shp:
                        break
                except StopIteration:
                    next_im = None
                    break

        # if we had a shape mismatch, there will still be images left to process
        # process them the slow way here...
        if next_im is not None:
            for imarray in [next_im, *arraygen]:
                with RLContext(imarray.shape, otf.path, **kwargs) as ctx:
                    out.append(rl_decon(imarray, output_shape=ctx.out_shape, **kwargs))

    if isinstance(images, (list, tuple)) and len(images) > 1:
        return out
    else:
        return out[0]
