import os
from fnmatch import fnmatch
from typing import Iterator, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
from typing_extensions import Literal

from . import lib
from .otf import TemporaryOTF
from .util import PathOrArray, _kwargs_for, imread


def rl_cleanup() -> None:
    """Release GPU buffer and cleanup after deconvolution.

    Call this before program quits to release global GPUBuffer d_interpOTF.

    - Resets any bleach corrections
    - Removes OTF from GPU buffer
    - Destroys cuFFT plan
    - Releases GPU buffers
    """
    lib.RL_cleanup()


def rl_init(
    rawdata_shape: Tuple[int, int, int],
    otfpath: str,
    dzdata: float = 0.5,
    dxdata: float = 0.1,
    dzpsf: float = 0.1,
    dxpsf: float = 0.1,
    deskew: float = 0,
    rotate: float = 0,
    width: int = 0,
    skewed_decon: bool = False,
) -> None:
    """Initialize GPU for deconvolution.

    Prepares cuFFT plan for deconvolution with a given data shape and OTF.
    Must be used prior to :func:`pycudadecon.rl_decon`

    Parameters
    ----------
    rawdata_shape : Tuple[int, int, int]
        3-tuple of data shape
    otfpath : str
        Path to OTF TIF
    dzdata : float, optional
        Z-step size of data, by default 0.5
    dxdata : float, optional
        XY pixel size of data, by default 0.1
    dzpsf : float, optional
        Z-step size of the OTF, by default 0.1
    dxpsf : float, optional
        XY pixel size of the OTF, by default 0.1
    deskew : float, optional
        Deskew angle. If not 0.0 then deskewing will be performed before
        deconvolution, by default 0
    rotate : float, optional
        Rotation angle; if not 0.0 then rotation will be performed around Y
        axis after deconvolution, by default 0
    width : int, optional
        If deskewed, the output image's width, by default 0 (do not crop)
    skewed_decon : bool, optional
        If True, perform deconvolution in skewed space, by default False.

    Examples
    --------
    >>> rl_init(im.shape, otfpath)
    >>> decon_result = rl_decon(im)
    >>> rl_cleanup()

    """
    nz, ny, nx = rawdata_shape

    args: list = [nx, ny, nz, dxdata, dzdata, dxpsf, dzpsf, deskew, rotate, width]

    if lib.lib.version >= (0, 6):
        args += [skewed_decon]

    lib.RL_interface_init(*args, otfpath.encode())


def rl_decon(
    im: np.ndarray,
    background: Union[int, Literal["auto"]] = 80,
    n_iters: int = 10,
    shift: int = 0,
    save_deskewed: bool = False,
    output_shape: Optional[Tuple[int, int, int]] = None,
    napodize: int = 15,
    nz_blend: int = 0,
    pad_val: float = 0.0,
    dup_rev_z: bool = False,
    skewed_decon: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Perform Richardson Lucy Deconvolution.

    Performs actual deconvolution. GPU must first be initialized with
    :func:`pycudadecon.rl_init`

    Parameters
    ----------
    im : np.ndarray
        3D image volume to deconvolve
    background : int or 'auto'
        User-supplied background to subtract. If 'auto', the median value of the
        last Z plane will be used as background. by default 80
    n_iters : int, optional
        Number of iterations, by default 10
    shift : int, optional
        If deskewed, the output image's extra shift in X (positive->left),
        by default 0
    save_deskewed : bool, optional
        Save deskewed raw data as well as deconvolution result, by default False
    output_shape : tuple of int, optional
        Specify the output shape after deskewing.  Usually this is unnecessary and
        can be autodetected.  Mostly intended for use within a
        :class:`pycudadecon.RLContext` context, by default None
    napodize : int, optional
        Number of pixels to soften edge with, by default 15
    nz_blend : int, optional
        Number of top and bottom sections to blend in to reduce axial ringing,
        by default 0
    pad_val : float, optional
        Value with which to pad image when deskewing, by default 0.0
    dup_rev_z : bool, optional
        Duplicate reversed stack prior to decon to reduce axial ringing,
        by default False
    skewed_decon : bool, optional
        If True, perform deconvolution in skewed space, by default False.

    Returns
    -------
    np.ndarray or 2-tuple of np.ndarray
        The deconvolved result.  If `save_deskewed` is `True`, returns
        `(decon_result, deskew_result)`

    Raises
    ------
    ValueError
        If im.ndim is not 3, or `output_shape` is provided but not length 3
    """
    if im.ndim != 3:
        raise ValueError("Only 3D arrays supported")

    nz, ny, nx = im.shape
    if output_shape is None:
        output_shape = (lib.get_output_nz(), lib.get_output_ny(), lib.get_output_nx())
    elif len(output_shape) != 3:
        raise ValueError("Decon output shape must have length==3")

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

    args = [
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
    ]

    if lib.lib.version >= (0, 6):
        args += [skewed_decon]

    lib.RL_interface(*args)

    if save_deskewed:
        return decon_result, deskew_result
    else:
        return decon_result


def quickDecon(
    image: np.ndarray, otfpath: str, **kwargs: Any
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Perform deconvolution of `image` with otf at `otfpath`.

    Not currently used...
    """
    assert image.ndim == 3, "Only 3D arrays supported"
    rl_init(image.shape, otfpath, **_kwargs_for(rl_init, kwargs))  # type: ignore
    result = rl_decon(image, **_kwargs_for(rl_decon, kwargs))
    lib.RL_cleanup()
    return result


class RLContext:
    """Context manager to setup the GPU for RL decon.

    Takes care of handing the OTF to the GPU, preparing a cuFFT plane,
    and cleaning up after decon.  Internally, this calls :func:`rl_init`,
    stores the shape of the expected output volume after any deskew/decon,
    then calls :func:`rl_cleanup` when exiting the context.

    For parameters, see :func:`rl_init`.

    Examples
    --------
    >>> with RLContext(data.shape, otfpath, dz) as ctx:
    ...     result = rl_decon(data, ctx.out_shape)
    """

    def __init__(
        self,
        rawdata_shape: Tuple[int, int, int],
        otfpath: str,
        dzdata: float = 0.5,
        dxdata: float = 0.1,
        dzpsf: float = 0.1,
        dxpsf: float = 0.1,
        deskew: float = 0,
        rotate: float = 0,
        width: int = 0,
    ):
        self.kwargs = {
            "rawdata_shape": rawdata_shape,
            "otfpath": otfpath,
            "dzdata": dzdata,
            "dxdata": dxdata,
            "dzpsf": dzpsf,
            "dxpsf": dxpsf,
            "deskew": deskew,
            "rotate": rotate,
            "width": width,
        }
        self.out_shape: Optional[Tuple[int, int, int]] = None

    def __enter__(self) -> "RLContext":
        """Setup the context and return the ZYX shape of the output image."""
        rl_init(**self.kwargs)  # type: ignore
        self.out_shape = (lib.get_output_nz(), lib.get_output_ny(), lib.get_output_nx())
        return self

    def __exit__(self, *_: Any) -> None:
        """Cleanup the context."""
        # exit receives a tuple with any exceptions raised during processing
        # if __exit__ returns True, exceptions will be supressed
        lib.RL_cleanup()


# alias
rl_context = RLContext


def _yield_arrays(
    images: Union[PathOrArray, Sequence[PathOrArray]], fpattern: str = "*.tif"
) -> Iterator[np.ndarray]:
    """Yield arrays from an array, path, or sequence of either.

    Parameters
    ----------
    images : Union[PathOrArray, Sequence[PathOrArray]]
        an array, path, or sequence of either
    fpattern : str, optional
        used to filter files in a directory, by default "*.tif"

    Yields
    ------
    Iterator[np.ndarray]
        Arrays (read from paths if necessary)

    Raises
    ------
    OSError
        If a directory is provided and no files match fpattern.
    """
    if isinstance(images, np.ndarray):
        yield images

    elif isinstance(images, str):
        if os.path.isfile(images):
            yield imread(images)

        elif os.path.isdir(images):
            imfiles = [f for f in os.listdir(images) if fnmatch(f, fpattern)]
            if not len(imfiles):
                raise OSError(
                    'No files matching pattern "{}" found in directory: {}'.format(
                        fpattern, images
                    )
                )
            for fpath in imfiles:
                yield imread(os.path.join(images, fpath))

    else:
        for item in images:
            yield from _yield_arrays(item)


def decon(
    images: Union[PathOrArray, Sequence[PathOrArray]],
    psf: PathOrArray,
    fpattern: str = "*.tif",
    **kwargs: Any
) -> Union[np.ndarray, List[np.ndarray]]:
    """Deconvolve an image or images with a PSF or OTF file.

    If `images` is a directory, use the `fpattern` argument to select files
    by filename pattern.

    Parameters
    ----------
    images : str, np.ndarray, or sequence of either
        The array, filepath, directory, or list/tuple thereof to deconvolve
    psf : str or np.ndarray
        a filepath of a PSF or OTF file, or a 3D numpy PSF array.  Function will
        auto-detect whether the file is a 3D PSF or a filepath representing a 2D
        complex OTF.
    fpattern : str, optional
        Filepattern to use when a directory is provided in the `images` argument,
        by default `*.tif`
    ** kwargs
        All other kwargs must be valid for either :func:`rl_init` or :func:`rl_decon`.

    Returns
    -------
    np.ndarray or list of array
        The deconvolved image(s)

    Raises
    ------
    ValueError
        If save_deskewed is True and deskew is unset or 0
    IOError
        If a directory is provided as input and ``fpattern`` yields no files
    NotImplementedError
        If ``psf`` is provided as a complex, 2D numpy array (OTFs can only be
        provided as filenames created with :func:`pycudadecon.make_otf`)

    Examples
    --------
    deconvolve a 3D TIF volume with a 3D PSF volume (e.g. a single bead stack)

    >>> result = decon('/path/to/image.tif', '/path/to/psf.tif')

    deconvolve all TIF files in a specific directory that match a certain
    `filename pattern <https://docs.python.org/3.6/library/fnmatch.html>`_,
    (in this example, all TIFs with the string '560nm' in their name)

    >>> result = decon(
    ...     '/directory/with/images', '/path/to/psf.tif', fpattern='*560nm*.tif'
    ... )

    deconvolve a list of images, provided either as np.ndarrays, filepaths,
    or directories

    >>> imarray = tifffile.imread('some_other_image.tif')
    >>> inputs = ['/directory/with/images', '/path/to/image.tif', imarray]
    >>> result = decon(inputs, '/path/to/psf.tif', fpattern='*560nm*.tif')
    """
    if kwargs.get("save_deskewed"):
        if kwargs.get("deskew", 1) == 0:
            raise ValueError("Cannot use save_deskewed=True with deskew=0")
        if not kwargs.get("deskew"):
            raise ValueError("Must set deskew != 0 when using save_deskewed=True")

    init_kwargs = _kwargs_for(rl_init, kwargs)
    decon_kwargs = _kwargs_for(rl_decon, kwargs)

    out = []
    with TemporaryOTF(psf, **kwargs) as otf:
        arraygen = _yield_arrays(images, fpattern)
        # first, assume that all of the images are the same shape...
        # in which case we can prevent a lot of GPU IO
        # grab and store the shape of the first item in the generator
        next_im: np.ndarray = next(arraygen)
        assert next_im.ndim == 3, "Images must be 3D"
        shp = next_im.shape

        with RLContext(shp, otf.path, **init_kwargs) as ctx:  # type: ignore
            while True:
                out.append(
                    rl_decon(next_im, output_shape=ctx.out_shape, **decon_kwargs)
                )
                try:
                    next_im = next(arraygen)
                    # here we check to make sure that the images are still the same
                    # shape... if not, we'll continue below
                    if next_im.shape != shp:
                        break
                except StopIteration:
                    next_im = None  # type: ignore
                    break

        # if we had a shape mismatch, there will still be images left to process
        # process them the slow way here...
        if next_im is not None:
            for imarray in [next_im, *arraygen]:
                with RLContext(imarray.shape, otf.path, **init_kwargs) as ctx:  # type: ignore # noqa
                    out.append(
                        rl_decon(imarray, output_shape=ctx.out_shape, **decon_kwargs)
                    )

    if isinstance(images, (list, tuple)) and len(images) > 1:
        return out  # type: ignore
    else:
        return out[0]  # type: ignore
