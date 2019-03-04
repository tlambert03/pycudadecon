from .util import load_lib, imread
import ctypes
import logging
logger = logging.getLogger(__name__)


try:
    import pathlib as plib
    plib.Path()
except (ImportError, AttributeError):
    import pathlib2 as plib
except (ImportError, AttributeError):
    raise ImportError('no pathlib detected. For python2: pip install pathlib2')


otflib = load_lib('libradialft')

if not otflib:
    logger.error('Could not load libradialft!')
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
            ctypes.c_bool
        ]
    except AttributeError as e:
        logger.warn('Failed to properly import libradialft')
        logger.error(e)


def requireOTFlib(func, *args, **kwargs):
    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not otflib:
                raise Exception("Could not find libradialft library! OTF generation "
                                "will not be available:")
            else:
                raise e
    return dec


def makeotf(psf, outpath=None, wavelength=520, dxpsf=0.1, dzpsf=0.1, na=1.25,
            nimm=1.3, otf_bgrd=None, krmax=0, fixorigin=10, cleanup_otf=False,
            **kwargs):
    """
    Generate a radially averaged OTF file from a PSF file, at outpath
    """
    # krmax => "pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM)")
    if outpath is None:
        outpath = psf.replace('.tif', '_otf.tif')

    if otf_bgrd and isinstance(otf_bgrd, (int, float)):
        bUserBackground = True
        background = float(otf_bgrd)
    else:
        bUserBackground = False
        background = 0.0

    shared_makeotf(str.encode(psf), str.encode(outpath), wavelength, dzpsf,
                   fixorigin, bUserBackground, background, na, nimm, dxpsf,
                   krmax, cleanup_otf)
    return outpath
