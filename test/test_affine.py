import numpy as np
import numpy.testing as npt
import pytest

from pycudadecon import affineGPU, deskewGPU, rotateGPU


def test_deskew(raw_image, deskewed_image):
    """test basic deskewing of a file aquired on a lattice light sheet"""
    deskewed = deskewGPU(raw_image, dzdata=0.3, angle=31.5, pad_val=98)
    npt.assert_allclose(deskewed, deskewed_image)


def test_affine_raises(raw_image):
    """Make sure that the affine transformation matrix must be the right size"""
    with pytest.raises(ValueError):
        affineGPU(raw_image, np.eye(5))
    with pytest.raises(ValueError):
        affineGPU(raw_image, np.eye(3))
    with pytest.raises(ValueError):
        affineGPU(raw_image, np.eye(2))


def test_affine_eye(raw_image):
    """test that an affine transform with identity matrix does not change the input"""
    result = affineGPU(raw_image, np.eye(4))
    npt.assert_allclose(result, raw_image)


def test_affine_translate(raw_image):
    """
    test affine translation leads to a bunch of empty values up to
    but not more than the requested amount of translation
    """
    xpix = -50
    ypix = -100
    zpix = -3
    T = np.array([[1, 0, 0, xpix], [0, 1, 0, ypix], [0, 0, 1, zpix], [0, 0, 0, 1]])
    result = affineGPU(raw_image, T)
    assert not np.allclose(raw_image, result, atol=10)
    assert np.all(result[:-zpix] == 0)
    assert not np.all(result[: -zpix + 1] == 0)
    assert np.all(result[:, :-xpix, :-zpix] == 0)


def test_affine_translate_RA(raw_image):
    """
    make sure the referenceing object works to accept transformation
    matrices in units of sample space, instead of world coordinates
    """
    xpix = -50
    ypix = -100
    zpix = -3
    T = np.array([[1, 0, 0, xpix], [0, 1, 0, ypix], [0, 0, 1, zpix], [0, 0, 0, 1]])
    voxsize = (0.5, 0.5, 0.5)
    result = affineGPU(raw_image, T, voxsize)
    assert not np.allclose(raw_image, result, atol=10)
    assert np.all(result[: -zpix * 2] == 0)
    assert not np.all(result[: (-zpix * 2) + 1] == 0)
    assert np.all(result[:, : -xpix * 2, : -zpix * 2] == 0)


def test_rotate(deskewed_image, rotated_image):
    """test that rotateGPU rotates the image on the Y axis by some angle"""
    result = rotateGPU(deskewed_image, 0.3, dxdata=0.1, angle=31.5)
    npt.assert_allclose(result, rotated_image)
