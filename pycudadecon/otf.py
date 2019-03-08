from .util import load_lib, is_otf
import tempfile
import numpy as np
import tifffile as tf
import os
import ctypes
import logging

logger = logging.getLogger(__name__)


otflib = load_lib("libradialft")

if not otflib:
    logger.error("Could not load libradialft!")
else:
    try:
        shared_makeotf = otflib.makeOTF
        shared_makeotf.restype = ctypes.c_int
        shared_makeotf.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_bool,
        ]
    except AttributeError as e:
        logger.warn("Failed to properly import libradialft")
        logger.error(e)


def requireOTFlib(func, *args, **kwargs):
    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not otflib:
                raise Exception(
                    "Could not find libradialft library! OTF generation "
                    "will not be available:"
                )
            else:
                raise e

    return dec


def make_otf(
    psf,
    outpath=None,
    dzpsf=0.1,
    dxpsf=0.1,
    wavelength=520,
    na=1.25,
    nimm=1.3,
    otf_bgrd=None,
    krmax=0,
    fixorigin=10,
    cleanup_otf=False,
    **kwargs
):
    """ Generate a radially averaged OTF file from a PSF file

    Args:
        psf (str): Filepath of 3D PSF TIF
        outpath (str): Destination filepath for the output OTF
            (default: appends "_otf.tif" to filename)
        dzpsf: Z-step size in microns (default: {0.1})
        dxpsf: XY-Pixel size in microns (default: {0.1})
        wavelength: Emission wavelength in nm (default: {520})
        na: Numerical Aperture (default: {1.25})
        nimm: Refractive indez of immersion medium (default: {1.3})
        otf_bgrd: Background to subtract. "None" = autodetect. (default: {None})
        krmax: pixels outside this limit will be zeroed (overwriting
            estimated value from NA and NIMM) (default: {0})
        fixorigin: for all kz, extrapolate using pixels kr=1 to this pixel
            to get value for kr=0 (default: {10})
        cleanup_otf: clean-up outside OTF support (default: {False})

    Returns:
        str: Path of output file
    """
    if outpath is None:
        outpath = psf.replace(".tif", "_otf.tif")

    if otf_bgrd and isinstance(otf_bgrd, (int, float)):
        bUserBackground = True
        background = float(otf_bgrd)
    else:
        bUserBackground = False
        background = 0.0

    shared_makeotf(
        str.encode(psf),
        str.encode(outpath),
        wavelength,
        dzpsf,
        fixorigin,
        bUserBackground,
        background,
        na,
        nimm,
        dxpsf,
        krmax,
        cleanup_otf,
    )
    return outpath


class TemporaryOTF(object):
    """Context manager to read OTF file or generate a temporary OTF from a PSF.

    Normalizes the input PSF to always provide the path to an OTF file,
    converting the PSF to a temporary file if necessary.

    ``self.path`` can be used within the context to get the filepath to
    the temporary OTF filepath.

    Args:
        psf (str, np.ndarray): 3D PSF numpy array, or a filepath to a 3D PSF
            or 2D complex OTF file.
        **kwargs: optional keyword arguments will be passed to the :func:`pycudadecon.otf.make_otf` function

    Note:
        OTF files cannot currently be provided directly as 2D complex np.ndarrays

    Raises:
        ValueError: If the PSF/OTF is an unexpected type
        NotImplementedError: if the PSF/OTF is a complex 2D numpy array

    Example:
        >>> with TemporaryOTF(psf, **kwargs) as otf:
                print(otf.path)
        /tmp/...
    """

    def __init__(self, psf, **kwargs):
        self.psf = psf
        self.kwargs = kwargs

    def __enter__(self):
        if not is_otf(self.psf):
            self.tempotf = tempfile.NamedTemporaryFile(suffix=".tif")
            if isinstance(self.psf, np.ndarray):
                with tempfile.NamedTemporaryFile(suffix=".tif") as temp_psf:
                    tf.imsave(temp_psf.name, self.psf)
                    make_otf(temp_psf.name, self.tempotf.name, **self.kwargs)
            elif isinstance(self.psf, str) and os.path.isfile(self.psf):
                make_otf(self.psf, self.tempotf.name, **self.kwargs)
            else:
                raise ValueError("Did not expect PSF file as {}".format(type(self.psf)))
            self.path = self.tempotf.name
        elif is_otf(self.psf) and os.path.isfile(self.psf):
            self.path = self.psf
        elif is_otf(self.psf) and isinstance(self.psf, np.ndarray):
            raise NotImplementedError("cannot yet handle OTFs as numpy arrays")
        else:
            raise ValueError("Unrecognized input for otf")
        return self

    def __exit__(self, typ, val, traceback):
        try:
            self.tempotf.close()
        except Exception:
            pass

