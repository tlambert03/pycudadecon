import os
from fnmatch import fnmatch

import numpy as np
import tifffile as tf

from ._libwrap import (
    RL_cleanup,
    RL_interface,
    RL_interface_init,
    get_output_nx,
    get_output_ny,
    get_output_nz,
)
from .otf import TemporaryOTF


def rl_init(
    rawdata_shape,
    otfpath,
    dzdata=0.5,
    dxdata=0.1,
    dzpsf=0.1,
    dxpsf=0.1,
    deskew=0,
    rotate=0,
    width=0,
    **kwargs
):
    """Initialize GPU for deconvolution

    prepares cuFFT plan for deconvolution with a given data shape and OTF.
    Must be used prior to :func:`pycudadecon.rl_decon`


    Args:
        rawdata_shape (tuple, list): 3-tuple of data shape [nz, ny, nx]
        otfpath (str): Path to OTF TIF

        dzdata (float): Z-step size of data (default: {0.5})
        dxdata (float): XY pixel size of data (default: {0.1})
        dzpsf (float): Z-step size of the OTF (default: {0.1})
        dxpsf (float): XY pixel size of the OTF (default: {0.1})
        deskew (float): Deskew angle. If not 0.0 then deskewing will be
            performed before deconv (default: {0})
        rotate (float): Rotation angle; if not 0.0 then rotation will be
            performed around Y axis after deconvolution (default: {0})
        width (int): If deskewed, the output image's width (default: do not crop)

    Example:
        >>> rl_init(im.shape, otfpath)
        >>> decon_result = rl_decon(im)
        >>> RL_cleanup()
    """
    nz, ny, nx = rawdata_shape
    RL_interface_init(
        nx,
        ny,
        nz,
        dxdata,
        dzdata,
        dxpsf,
        dzpsf,
        deskew,
        rotate,
        width,
        otfpath.encode(),
    )


def rl_decon(
    im,
    background=80,
    n_iters=10,
    shift=0,
    save_deskewed=False,
    output_shape=None,
    napodize=15,
    nz_blend=0,
    pad_val=0.0,
    dup_rev_z=False,
    **kwargs
):
    """Perform Richardson Lucy Deconvolution

    Performs actual deconvolution, after GPU has been initialized with
    :func:`pycudadecon.rl_init`

    Args:
        im (np.ndarray): 3D image volume to deconvolve

        background: User-supplied background to subtract. If 'auto', the
            median value of the last Z plane will be used as background.
            (default: {80})
        n_iters: Number of decon iterations (default: {10})
        shift: If deskewed, the output image's extra shift in X
            (positive->left). (default: {0})
        save_deskewed: Save deskewed raw data as well as deconvolution
            result (default: {False})
        output_shape: Specify the output shape after deskewing.  Usually this
            is unnecessary and will be autodetected.  Mostly intended for
            use within a :class:`pycudadecon.RLContext` context.
            (default: autodetect)
        napodize: Number of pixels to soften edge with. (default: {15})
        nz_blend: Number of top and bottom sections to blend in to reduce
            axial ringing. (default: {0})
        pad_val: Value to pad image with when deskewing (default: {0.0})
        dup_rev_z: Duplicate reversed stack prior to decon to reduce
            Z ringing (default: {False})
    """
    nz, ny, nx = im.shape
    if output_shape is None:
        output_shape = (get_output_nz(), get_output_ny(), get_output_nx())
    else:
        assert len(output_shape) == 3, "Decon output shape must have length==3"
    decon_result = np.empty(tuple(output_shape), dtype=np.float32)

    if save_deskewed:
        deskew_result = np.empty_like(decon_result)
    else:
        deskew_result = np.empty(1, dtype=np.float32)

    # must be 16 bit going in
    if not np.issubdtype(im.dtype, np.uint16):
        im = im.astype(np.uint16)

    if isinstance(background, str) and background == "auto":
        background = np.median(im[-1])

    rescale = False  # not sure if this works yet...

    if not im.flags["C_CONTIGUOUS"]:
        im = np.ascontiguousarray(im)
    RL_interface(
        im,
        nx,
        ny,
        nz,
        decon_result,
        deskew_result,
        background,
        rescale,
        save_deskewed,
        n_iters,
        shift,
        napodize,
        nz_blend,
        pad_val,
        dup_rev_z,
    )

    if save_deskewed:
        return decon_result, deskew_result
    else:
        return decon_result


def quickDecon(im, otfpath, save_deskewed=False, **kwargs):
    """Perform deconvolution of im with otf at otfpath

    Not currently used...

    kwargs can be:
        dxdata      float
        dzdata      float
        dxpsf       float
        dzpsf       float
        deskew      float  (0 is no deskew)
        n_iters      int
        save_deskewed  bool
        width       int
        shift       int
        rotate      float
    """
    rl_init(im.shape, otfpath, **kwargs)
    if save_deskewed:
        decon_result, deskew_result = rl_decon(im, save_deskewed=True, **kwargs)
        RL_cleanup()
        return decon_result, deskew_result
    else:
        decon_result = rl_decon(im, save_deskewed=False, **kwargs)
        RL_cleanup()
        return decon_result


class RLContext:
    """Context manager to setup the GPU for RL decon

    Takes care of handing the OTF to the GPU, preparing a cuFFT plane,
    and cleaning up after decon.  Internally, this calls :func:`rl_init`,
    stores the shape of the expected output volume after any deskew/decon,
    then calls :func:`rl_cleanup` when exiting the context.

    Args:
        shape (tuple, list): 3-Tuple with the shape of the data volume to
            deconvolve ([nz, ny, nx])
        otfpath (str): path to the OTF TIF file
        **kwargs: optional keyword arguments to pass to :func:`rl_init`

    Example:
        >>> with RLContext(data.shape, otfpath, dz) as ctx:
        ...     result = rl_decon(data, ctx.out_shape)

    """

    def __init__(self, shape, otfpath, **kwargs):
        self.shape = shape
        self.otfpath = otfpath
        self.kwargs = kwargs
        self.out_shape = None
        self.deviceID = 0

    def __enter__(self):
        """Setup the context and return the ZYX shape of the output image"""
        rl_init(self.shape, self.otfpath, **self.kwargs)
        self.out_shape = (get_output_nz(), get_output_ny(), get_output_nx())
        return self

    def __exit__(self, typ, val, traceback):
        # exit receives a tuple with any exceptions raised during processing
        # if __exit__ returns True, exceptions will be supressed
        RL_cleanup()


def _yield_arrays(images, fpattern="*.tif"):
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
                raise OSError(
                    'No files matching pattern "{}" found in directory: {}'.format(
                        fpattern, images
                    )
                )
            for fpath in imfiles:
                yield tf.imread(os.path.join(images, fpath))

    elif isinstance(images, (list, tuple)):
        for item in images:
            yield from _yield_arrays(item)  # noqa


def decon(images, psf, fpattern="*.tif", **kwargs):
    r"""Deconvolve an image or images with a PSF or OTF file

    If `images` is a directory, use the `fpattern` argument to select files
    by filename pattern.

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
        ** rotate (float): Degrees to rotate volume in Y axis (to make Z axis
            orthogonal to coverslip). Defaults to 0
        ** width (int): Width of output image (0 = full). Defaults to 0
        ** n_iters (int): Number of iterations in deconvolution Defaults to 10
        ** save_deskewed (bool): Save raw deskewed files (if deskew > 0).
            Defaults to False
        ** napodize (int): Number of pixels to soften edge with. Defaults to 15
        ** nz_blend (int): Number of top and bottom sections to blend in to reduce
            axial ringing. Defaults to 0
        ** dup_rev_z (bool): Duplicate reversed stack prior to decon to reduce axial
            ringing. Defaults to False
        ** wavelength (int): Wavelength in nanometers (for OTF cleanup). Defaults to 520
        ** fixorigin (int):  For all kz, extrapolate using pixels kr=1 to this pixel to
            get value for kr=0. Defaults to 10
        ** otf_bgrd (None, int): Background to subtract in PSF (None = autodetect).
            Defaults to None
        ** na (float): Numerical aperture (for OTF cleanup). Defaults to 1.25]
        ** nimm (float): Refractive index of medium (for OTF cleanup). Defaults to 1.3
        ** krmax (int): Pixels outside this limit will be zeroed (overwriting estimated
            value from ``na`` and ``nimm``). Defaults to 0
        ** cleanup_otf (bool): Clean up OTF outside of OTF support. Defaults to False

    Raises:
        ValueError: If save_deskewed is True and deskew is unset or 0
        IOError: If a directory is provided as input and ``fpattern`` yields no files
        NotImplementedError: If ``psf`` is provided as a complex, 2D numpy array
            (OTFs can only be provided as filenames created with
            :func:`pycudadecon.make_otf`)

    Returns:
        np.ndarray, list: numpy array or list of arrays (deconvolved images)

        If the input ``images`` is a single array, or filepath, then the returned
        value will be a single 3D image volume.

        If the input is a directory or a list of arrays, then the returned value
        will be a list of 3D image volumes.

        if ``save_deskewed == True``, returns a tuple (decon, deskewed) or a list
        of tuples (if input was iterable)

    Examples:

        deconvolve a 3D TIF volume with a 3D PSF volume (e.g. a single bead stack)

        >>> impath = '/path/to/image.tif'
        >>> psfpath = '/path/to/psf.tif'
        >>> result = decon(impath, psfpath)

        deconvolve all TIF files in a specific directory that match a certain
        `filename pattern <https://docs.python.org/3.6/library/fnmatch.html>`_,
        (in this example, all TIFs with the string '560nm' in their name)

        >>> imdirectory = '/directory/with/images'
        >>> psfpath = '/path/to/psf.tif'
        >>> result = decon(imdirectory, psfpath, fpattern='*560nm*.tif')

        deconvolve a list of images, provided either as np.ndarrays, filepaths,
        or directories

        >>> imdirectory = '/directory/with/images'
        >>> impath = '/path/to/image.tif'
        >>> imarray = tifffile.imread('some_other_image.tif')
        >>> psfpath = '/path/to/psf.tif'
        >>> result = decon([imdirectory, impath, imarray],
        ...                 psfpath, fpattern='*560nm*.tif')



    """
    if kwargs.get("save_deskewed"):
        if kwargs.get("deskew", 1) == 0:
            raise ValueError("Cannot use save_deskewed=True with deskew=0")
        if kwargs.get("deskew", 0) == 0:
            raise ValueError("Must set deskew != 0 when using save_deskewed=True")

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
                    if next_im.shape != shp:
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
