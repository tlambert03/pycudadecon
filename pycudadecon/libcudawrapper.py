from .util import load_lib

import ctypes
import numpy as np
import logging
logger = logging.getLogger(__name__)


cudaLib = load_lib('libcudaDeconv')
rl_cleanup = None
cuda_reset = None


if not cudaLib:
    logger.error('Could not load libcudaDeconv!')
else:
    try:
        # setup
        camcor_interface_init = cudaLib.camcor_interface_init
        camcor_interface_init.restype = ctypes.c_int
        camcor_interface_init.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
        ]

        # execute camcor
        camcor_interface = cudaLib.camcor_interface
        camcor_interface.restype = ctypes.c_int
        camcor_interface.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS")
        ]

        # RL_interface_init must be used before using RL_interface
        RL_interface_init = cudaLib.RL_interface_init
        RL_interface_init.restype = ctypes.c_int
        RL_interface_init.argtypes = [
            ctypes.c_int,     # nx
            ctypes.c_int,               # ny
            ctypes.c_int,               # nz
            ctypes.c_float,             # dxdata
            ctypes.c_float,             # dzdata
            ctypes.c_float,             # dxpsf
            ctypes.c_float,             # dzpsf
            ctypes.c_float,             # angle
            ctypes.c_float,             # rotate
            ctypes.c_int,               # outputwidth
            ctypes.c_char_p,            # otfpath.encode()
        ]

        # used between init and RL_interface to retrieve the post-deskewed image dimensions
        get_output_nx = cudaLib.get_output_nx
        get_output_ny = cudaLib.get_output_ny
        get_output_nz = cudaLib.get_output_nz

        # The actual decon
        RL_interface = cudaLib.RL_interface
        RL_interface.restype = ctypes.c_int
        RL_interface.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),  # im
            ctypes.c_int,                                                   # nx
            ctypes.c_int,                                                   # ny
            ctypes.c_int,                                                   # nz
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),   # result
            ctypes.c_float,                                                 # background
            ctypes.c_bool,                                                  # doRescale
            ctypes.c_bool,                                                  # save_deskeweded
            ctypes.c_int,                                                   # n_iters
            ctypes.c_int,                                                   # shift
            ctypes.c_int,   # napodize
            ctypes.c_int,   # nz_blend
            ctypes.c_float, # pad_val
            ctypes.c_bool, # bDupRevStack
        ]
        # call after
        rl_cleanup = cudaLib.RL_cleanup

        cuda_reset = cudaLib.cuda_reset

    except AttributeError as e:
        logger.warning('Failed to properly import libcudaDeconv')
        print(e)


def quickCamcor(imstack, camparams):
    """Correct Flash residual pixel artifact on GPU"""
    camcor_init(imstack.shape, camparams)
    return camcor(imstack)


def camcor_init(rawdata_shape, camparams):
    """ initialize camera correction on GPU.
    shape is nz/ny/nx of the concatenated stacks from a single timepoint
    """
    nz, ny, nx = rawdata_shape
    if not np.issubdtype(camparams.dtype, np.float32):
        camparams = camparams.astype(np.float32)
    camcor_interface_init(nx, ny, nz, camparams)


def camcor(imstack):
    if not np.issubdtype(imstack.dtype, np.uint16):
        imstack = imstack.astype(np.uint16)
    nz, ny, nx = imstack.shape
    result = np.empty_like(imstack)
    camcor_interface(imstack, nx, ny, nz, result)
    return result


def quickDecon(im, otfpath, save_deskewed=False, **kwargs):
    """Perform deconvolution of im with otf at otfpath

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
        rl_cleanup()
        return decon_result, deskew_result
    else:
        decon_result = rl_decon(im, save_deskewed=False, **kwargs)
        rl_cleanup()
        return decon_result


def rl_init(rawdata_shape, otfpath, dzdata=0.5, dxdata=0.1, dxpsf=0.1,
            dzpsf=0.1, deskew=0, rotate=0, width=0, **kwargs):
    nz, ny, nx = rawdata_shape
    RL_interface_init(nx, ny, nz, dxdata, dzdata, dxpsf, dzpsf, deskew, rotate,
                      width, otfpath.encode())


class RLContext(object):
    """ Context manager to setup the GPU for RL decon

    Takes care of handing the OTF to the GPU and cleaning up after decon

    EXAMPLE:
    with RLContext(data.shape, otfpath, dz) as ctx:
        return rl_decon(data, ctx.out_shape)

    """
    def __init__(self, shape, otfpath, **kwargs):
        self.shape = shape
        self.otfpath = otfpath
        self.kwargs = kwargs
        self.out_shape = None
        self.deviceID = 0

    def __enter__(self):
        """ Setup the context and return the ZYX shape of the output image """
        rl_init(self.shape, self.otfpath, **self.kwargs)
        self.out_shape = (get_output_nz(), get_output_ny(), get_output_nx())
        return self

    def __exit__(self, typ, val, traceback):
        # exit receives a tuple with any exceptions raised during processing
        # if __exit__ returns True, exceptions will be supressed
        rl_cleanup()


def rl_decon(im, background=80, n_iters=10, shift=0, save_deskewed=False,
             rescale=False, output_shape=None, napodize=15, nz_blend=0,
             pad_val=0.0, dup_rev_z=False, **kwargs):
    nz, ny, nx = im.shape
    if output_shape is None:
        output_shape = (get_output_nz(), get_output_ny(), get_output_nx())
    else:
        assert len(output_shape) == 3, 'Decon output shape must have length==3'
    decon_result = np.empty(tuple(output_shape), dtype=np.float32)

    if save_deskewed:
        deskew_result = np.empty_like(decon_result)
    else:
        deskew_result = np.empty(1, dtype=np.float32)

    # must be 16 bit going in
    if not np.issubdtype(im.dtype, np.uint16):
        im = im.astype(np.uint16)

    if isinstance(background, str) and background == 'auto':
        background = np.median(im[-1])

    if not im.flags['C_CONTIGUOUS']:
        im = np.ascontiguousarray(im)
    RL_interface(im, nx, ny, nz, decon_result, deskew_result,
                 background, rescale, save_deskewed, n_iters, shift,
                 napodize, nz_blend, pad_val, dup_rev_z)

    if save_deskewed:
        return decon_result, deskew_result
    else:
        return decon_result
