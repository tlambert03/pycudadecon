import logging
import os
import sys
import warnings
from contextlib import contextmanager
from inspect import signature
from typing import Callable, Union

import numpy as np
import tifffile as tf

logging.getLogger("tifffile").setLevel(logging.ERROR)

PLAT = sys.platform
if PLAT == "linux2":
    PLAT = "linux"
elif PLAT == "cygwin":
    PLAT = "win32"

PathOrArray = Union[str, np.ndarray]


def _kwargs_for(function: Callable, kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def imread(fpath: str, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.imread(fpath, **kwargs)


def array_is_otf(arr: np.ndarray) -> bool:
    """Check whether a numpy array is likely an OTF file

    tifffile reads the otf file as a real-valued float32, instead
    of complex valued... so the second column is almost always 0
    Does this always work?
    """
    # the first pixel of an OTF will always be 1.0 and the second column 0s
    # too strict? arr[0, 0] == 1
    return False if arr.dtype != "float32" else not arr[:, 1].any()


def path_is_otf(fpath: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tf.TiffFile(fpath) as tif:
            if (not len(tif.series)) or (tif.series[0].ndim != 2):
                return False
            return array_is_otf(tif.series[0].asarray())


def is_otf(arr_or_fpath: PathOrArray) -> bool:
    """
    accepts either a numpy array or a string representing a filepath
    and returns True if the array or path represents an OTF file
    """
    if isinstance(arr_or_fpath, str):
        # assume it's a filepath
        if os.path.isfile(arr_or_fpath):
            return path_is_otf(arr_or_fpath)
        else:
            raise FileNotFoundError(f"file path does not exist: {arr_or_fpath}")  # noqa
    elif isinstance(arr_or_fpath, np.ndarray):
        return array_is_otf(arr_or_fpath)
    return False


# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769
@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
