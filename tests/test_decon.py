from pathlib import Path

import numpy.testing as npt
import pytest

from pycudadecon import RLContext, decon, rl_cleanup, rl_decon, rl_init
from pycudadecon.util import imread

ATOL = 0.05  # tolerance for np.allclose

DATA = Path(__file__).parent / "test_data"
PSF_PATH = str(DATA / "psf.tif")
OTF_PATH = str(DATA / "otf.tif")


@pytest.fixture
def config():
    return {
        "dzdata": 0.3,
        "deskew": 0,
        "n_iters": 10,
        "background": 98,
    }


def test_decon(deskewed_image, decon_image):
    """test that we can deconvolve an image"""
    rl_init(deskewed_image.shape, OTF_PATH, dzdata=0.3)
    decon_result = rl_decon(deskewed_image, n_iters=10, background=98)
    npt.assert_allclose(decon_result, decon_image, atol=ATOL)
    rl_cleanup()


def test_decon_context(deskewed_image, decon_image):
    """test that we can deconvolve an image using the context manager"""
    with RLContext(deskewed_image.shape, OTF_PATH, dzdata=0.3) as ctx:
        decon_result = rl_decon(
            deskewed_image, output_shape=ctx.out_shape, n_iters=10, background=98
        )
    npt.assert_allclose(decon_result, decon_image, atol=ATOL)


def test_decon_wrapper_with_otf(deskewed_image, decon_image, config):
    """test that we can deconvolve when provided OTF directly"""
    decon_result = decon(deskewed_image, OTF_PATH, **config)
    npt.assert_allclose(decon_result, decon_image, atol=ATOL)


def test_decon_wrapper_with_psf(deskewed_image, decon_image, config):
    """test that we can deconvolve an image with psf as string"""
    decon_result = decon(deskewed_image, PSF_PATH, **config)
    npt.assert_allclose(decon_result, decon_image, atol=ATOL)


def test_decon_wrapper_with_psf_array(deskewed_image, decon_image, config):
    """test that we can deconvolve an image with psf as array"""
    decon_result = decon(deskewed_image, imread(PSF_PATH), **config)
    npt.assert_allclose(decon_result, decon_image, atol=ATOL)


def test_decon_wrapper_with_many_inputs(deskewed_image, decon_image, config):
    """test passing a list of images to decon"""
    images = [deskewed_image, deskewed_image, deskewed_image]

    for d in decon(images, OTF_PATH, **config):
        npt.assert_allclose(d, decon_image, atol=ATOL)


def test_decon_wrapper_with_variable_shapes(deskewed_image, config):
    """test passing a list of variabel shape images to decon"""
    im = deskewed_image
    images = [im, im[:, 4:-4, 4:-4], im[2:-2, 16:-16, 16:-16]]
    decon(images, OTF_PATH, **config)


def test_decon_wrapper_save_deskewed(raw_image, config):
    """test that save_deskewed includes deskewed when passing raw image"""
    config["deskew"] = 31.5
    decon_result = decon(raw_image, OTF_PATH, save_deskewed=True, **config)
    assert len(decon_result) == 2
    # npt.assert_allclose(decon_image, decon_result[1], atol=ATOL)
