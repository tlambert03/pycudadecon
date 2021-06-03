import os
import tempfile
from typing import Optional

import numpy as np
import tifffile as tf

from . import lib
from .util import imread, is_otf


def predict_otf_size(psf):
    """Calculate the file size of the OTF that would result from this psf.

    Note: this does not actually generate an OTF

    Args:
        psf (str, np.ndarray): psf that would be used to make the otf

    Returns:
        int: number of bytes that the OTF file would be

    Raises:
        ValueError: if the input is neither an existing filepath or numpy array
    """
    if isinstance(psf, str) and os.path.isfile(psf):
        with tf.TiffFile(psf) as file:
            nz, _, nx = file.series[0].shape
    elif isinstance(psf, np.ndarray):
        nz, _, nx = psf.shape
    else:
        raise ValueError("psf argument must be filepath or numpy array")
    # the radially averaged OTF only cares about the x
    otfpix = nz * 2 * (nx // 2 + 1)
    # header size adds up to around 251 bytes
    return 251 + otfpix * 32 // 8


def cap_psf_size(psf, max_otf_size=60000, min_xy=200, min_nz=20):
    """crop PSF to a size that will yield an OTF with a maximum specified size

    Args:
        psf (np.ndarray): 3D PSF to be turned into an OTF
        max_otf_size (int, optional): Maximum output OTF size. Defaults to 60000.
    """
    # output_otf_size = (1 + nx // 2) * (nz * 2) * 4
    if not max_otf_size:
        max_otf_size = np.inf

    if predict_otf_size(psf) <= max_otf_size:
        return psf
    _nz, _ny, _nx = psf.shape

    # figure out how close the PSF maximum is to the edge of the stack
    z_center, y_center, x_center = np.unravel_index(psf.argmax(), psf.shape)
    outnz = 2 * (_nz // 2 - np.abs(z_center - _nz // 2))

    # the maximum nx that would work given maximum z size
    outnx = max_otf_size // (outnz * 4) - 2
    # if it's less than the specified minimum
    # then figure out how much Z we can allow
    if outnx < min_xy:
        outnx = min_xy
        outnz = max_otf_size // ((1 + min_xy // 2) * 8)

    idx = tuple(
        slice(np.maximum(0, i - outnz // 2), i + 1 * outnz // 2)
        for i in (z_center, y_center, x_center)
    )
    psf = psf[idx]

    _nz, _ny, _nx = psf.shape
    assert (_nx // 2 + 1) * 2 * _nz * 4 <= max_otf_size, "Cap PSF failed...{}".format(
        psf.shape
    )
    return psf


class CappedPSF:
    """Context manager that provides the path to a 3D PSF that is guaranteed to
    yield an OTF that is smaller than the specified value.

    Args:
        psf (str, np.ndarray): Path to a PSF or a numpy array with a 3D PSF
        max_otf_size (int, None): maximum allowable size in bytes of the OTF.  If None,
            do not restrict size of OTF.

    Returns:
        str: the ``self.path`` attribute may be used within the context to retrieve
            a filepath with (if necessary) a temporary path to a cropped psf.  Or, if
            if the PSF was already small enough, simply returns the original PSF, as
            a temporary pathname if the psf was provided as a numpy array
    """

    def __init__(self, psf, max_otf_size=None):
        self.psf = psf
        self.max_otf_size = max_otf_size or np.inf
        self.temp_psf = None
        self.path = None

    def __enter__(self):
        if isinstance(self.psf, str) and os.path.isfile(self.psf):
            if predict_otf_size(self.psf) <= self.max_otf_size:
                self.path = self.psf
            else:
                self.psf = imread(self.psf)
        if isinstance(self.psf, np.ndarray):
            self.temp_psf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tf.imsave(self.temp_psf.name, cap_psf_size(self.psf, self.max_otf_size))
            self.path = self.temp_psf.name
        return self

    def __exit__(self, typ, val, traceback):
        if self.temp_psf is not None:
            try:
                self.temp_psf.close()
                os.remove(self.temp_psf)
            except Exception:
                pass


def make_otf(
    psf: str,
    outpath: Optional[str] = None,
    dzpsf: float = 0.1,
    dxpsf: float = 0.1,
    wavelength: int = 520,
    na: float = 1.25,
    nimm: float = 1.3,
    otf_bgrd: Optional[int] = None,
    krmax: int = 0,
    fixorigin: int = 10,
    cleanup_otf: bool = False,
    max_otf_size: int = 60000,
    **kwargs,
):
    """Generate a radially averaged OTF file from a PSF file

    Args:
        psf (str): Filepath of 3D PSF TIF
        outpath (str): Destination filepath for the output OTF
            (default: appends "_otf.tif" to filename)
        dzpsf (float): Z-step size in microns (default: {0.1})
        dxpsf (float): XY-Pixel size in microns (default: {0.1})
        wavelength (int): Emission wavelength in nm (default: {520})
        na (float): Numerical Aperture (default: {1.25})
        nimm (float): Refractive indez of immersion medium (default: {1.3})
        otf_bgrd (int, None): Background to subtract. "None" = autodetect.
            (default: {None})
        krmax (int): pixels outside this limit will be zeroed (overwriting
            estimated value from NA and NIMM) (default: {0})
        fixorigin (int): for all kz, extrapolate using pixels kr=1 to this pixel
            to get value for kr=0 (default: {10})
        cleanup_otf (bool): clean-up outside OTF support (default: {False})
        max_otf_size (int): make sure OTF is smaller than this many bytes. Deconvolution
            may fail if the OTF is larger than 60KB (default: 60000)

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

    with CappedPSF(psf, max_otf_size) as _psf:
        lib.makeOTF(
            str.encode(_psf.path),
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


class TemporaryOTF:
    """Context manager to read OTF file or generate a temporary OTF from a PSF.

    Normalizes the input PSF to always provide the path to an OTF file,
    converting the PSF to a temporary file if necessary.

    ``self.path`` can be used within the context to get the filepath to
    the temporary OTF filepath.

    Args:
        psf (str, np.ndarray): 3D PSF numpy array, or a filepath to a 3D PSF
            or 2D complex OTF file.
        **kwargs: optional keyword arguments will be passed to the
            :func:`pycudadecon.otf.make_otf` function

    Note:
        OTF files cannot currently be provided directly as 2D complex np.ndarrays

    Raises:
        ValueError: If the PSF/OTF is an unexpected type
        NotImplementedError: if the PSF/OTF is a complex 2D numpy array

    Example:
        >>> with TemporaryOTF(psf, **kwargs) as otf:
        ...     print(otf.path)

    """

    def __init__(self, psf, **kwargs):
        self.psf = psf
        self.kwargs = kwargs

    def __enter__(self):
        if not is_otf(self.psf):
            self.tempotf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            if isinstance(self.psf, np.ndarray):
                temp_psf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                tf.imsave(temp_psf.name, self.psf)
                make_otf(temp_psf.name, self.tempotf.name, **self.kwargs)
                try:
                    temp_psf.close()
                except Exception:
                    pass
            elif isinstance(self.psf, str) and os.path.isfile(self.psf):
                make_otf(self.psf, self.tempotf.name, **self.kwargs)
            else:
                raise ValueError(f"Did not expect PSF file as {type(self.psf)}")
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
            os.remove(self.tempotf.name)
        except Exception:
            pass
