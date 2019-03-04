import os
import sys
import ctypes
import tifffile as tf
import numpy as np
import warnings


PLAT = sys.platform
if PLAT == 'linux2':
    PLAT = 'linux'
elif PLAT == 'cygwin':
    PLAT = 'win32'


def imread(fpath, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.imread(fpath, **kwargs)


def array_is_otf(arr):
    """Check whether a numpy array is likely an OTF file

    tifffile reads the otf file as a real-valued float32, instead
    of complex valued... so the second column is almost always 0
    Does this always work?
    """
    if arr.dtype != 'float32':
        return False
    if arr.shape[0] > arr.shape[1]:
        return False
    if not arr[:, 1].any():
        return True


def path_is_otf(fpath):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tf.TiffFile(fpath) as tif:
            if (not len(tif.series)) or (tif.series[0].ndim != 2):
                return False
            return array_is_otf(tif.series[0].asarray())


def is_otf(arr_or_fpath):
    """
    accepts either a numpy array or a string representing a filepath
    and returns True if the array or path represents an OTF file
    """
    if isinstance(arr_or_fpath, str):
        # assume it's a filepath
        if os.path.isfile(arr_or_fpath):
            return path_is_otf(arr_or_fpath)
        else:
            raise FileNotFoundError('file path does not exist: {}'  # noqa
                                    .format(arr_or_fpath))
    elif isinstance(arr_or_fpath, np.ndarray):
        return array_is_otf(arr_or_fpath)
    return False


def getAbsoluteResourcePath(relativePath):
    """ Load relative path, in an environment agnostic way"""

    try:
        # PyInstaller stores data files in a tmp folder refered to as _MEIPASS
        basePath = sys._MEIPASS
    except Exception:
        # If not running as a PyInstaller created binary, try to find the data file as
        # an installed Python egg
        try:
            basePath = os.path.dirname(sys.modules['llspy'].__file__)
        except Exception:
            basePath = ''

        # If the egg path does not exist, assume we're running as non-packaged
        if not os.path.exists(os.path.join(basePath, relativePath)):
            basePath = 'llspy'

    path = os.path.join(basePath, relativePath)
    # If the path still doesn't exist, this function won't help you
    if not os.path.exists(path):
        return None

    return path


def load_lib(libname):
    """load shared library, searching a number of likely paths
    """
    # first just try to find it on the search path

    searchlist = [os.path.join(os.environ.get('CONDA_PREFIX', '.'), 'Library', 'bin'),
                  os.path.join(os.environ.get('CONDA_PREFIX', '.'), 'lib'),
                  './lib',
                  '.']

    ext = {'linux': '.so',
           'win32': '.dll',
           'darwin': '.dylib'}

    if not libname.endswith(('.so', '.dll', '.dylib')):
        libname += ext[PLAT]

    for f in searchlist:
        try:
            d = getAbsoluteResourcePath(f)
            return ctypes.CDLL(os.path.abspath(os.path.join(d, libname)))
        except Exception:
            continue

    #last resort, chdir into each dir
    curdir = os.path.abspath(os.curdir)
    for f in searchlist:
        try:
            d = os.path.abspath(getAbsoluteResourcePath(f))
            if os.path.isdir(d):
                os.chdir(d)
                lib = ctypes.CDLL(libname)
                os.chdir(curdir)
                return lib
            raise Exception('didn\'t find it')
        except Exception:
            continue
