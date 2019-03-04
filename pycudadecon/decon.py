from .util import is_otf
from .libcudawrapper import RLContext, rl_decon
from .otf import makeotf
import os
import tifffile as tf
from fnmatch import fnmatch
import tempfile
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


class TemporaryOTF(object):
    """ normalizes the input PSF to always provide the path to an OTF
    file ... converting the PSF to a temporary file if necessary
    """

    def __init__(self, psf, **kwargs):
        self.psf = psf
        self.kwargs = kwargs

    def __enter__(self):
        if not is_otf(self.psf):
            self.temp = tempfile.NamedTemporaryFile()
            if isinstance(self.psf, np.ndarray):
                with tempfile.NamedTemporaryFile() as tpsf:
                    tf.imsave(tpsf.name, self.psf)
                    makeotf(tpsf.name, self.temp.name, **self.kwargs)
            elif isinstance(self.psf, str) and os.path.isfile(self.psf):
                makeotf(self.psf, self.temp.name, **self.kwargs)
            else:
                raise NotImplementedError('Did not expect PSF file as {}'
                                          .format(type(self.psf)))
            self.path = self.temp.name
        elif is_otf(self.psf) and os.path.isfile(self.psf):
            self.path = self.psf
        elif is_otf(self.psf) and isinstance(self.psf, np.ndarray):
            raise NotImplementedError('cannot yet handle OTFs as numpy arrays')
        else:
            raise ValueError('Unrecognized input for otf')
        return self

    def __exit__(self, typ, val, traceback):
        try:
            self.temp.close()
        except Exception:
            pass


def decon(images, psf, fpattern='*.tif', **kwargs):
    """Deconvolve an image or images with a PSF or OTF file

    Inputs:
        images: file or directory path, a numpy array,  or a list/tuple of
                any of the above.
        psf: a filepath of a PSF or OTF file, or a 3D numpy PSF array
        kwargs:  optional keyword arguments, with defaults shown
            dzdata          [0.5]
            dxdata          [0.1]
            dxpsf           [0.1]
            dzpsf           [0.1]

            wavelength      [520]
            fixorigin       [10]
            otf_bgrd        [None]      None = autodetect
            na              [1.25]
            nimm            [1.3]
            krmax           [0]         pixels outside this limit will be zeroed
                                        (overwriting estimated value from NA and NIMM)
            cleanup_otf     [False]

            deskew          [0]
            rotate          [0]
            width           [0]
            background      [80]        'auto' = median val of last Z plane
            n_iters         [10]
            save_deskewed   [False]     save deskewed
            napodize        [15]
            nz_blend        [0]
            padVal          [0.0]       should be zero when background is auto
            dup_rev_z       [False]

    Returns:
        numpy array or list of arrays with deconvolved images
            if save_deskewed == True, returns a tuple (decon, deskewed)
            or a list of tuples (if input was iterable)
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
