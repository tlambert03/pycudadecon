import unittest
import os
import numpy as np
from pycudadecon import deskewGPU, rotateGPU, affineGPU
from pycudadecon.util import imread


class TestAffine(unittest.TestCase):
    def setUp(self):
        self.raw = os.path.join(os.path.dirname(__file__), 'test_data', 'im_raw.tif')
        self.deskewed = os.path.join(os.path.dirname(__file__), 'test_data', 'im_deskewed.tif')

        self.raw = imread(self.raw)
        self.deskewed = imread(self.deskewed)

    def test_deskew(self):
        """
        test basic deskewing of a file aquired on a lattice light sheet
        """
        deskewed = deskewGPU(self.raw, dzdata=0.3, angle=31.5, pad_val=98)
        self.assertTrue(np.allclose(deskewed, self.deskewed))

    def test_affine_raises(self):
        """
        Make sure that the affine transformation matrix must be the right size
        """
        with self.assertRaises(ValueError):
            affineGPU(self.raw, np.eye(5))
        with self.assertRaises(ValueError):
            affineGPU(self.raw, np.eye(3))
        with self.assertRaises(ValueError):
            affineGPU(self.raw, np.eye(2))

    def test_affine_eye(self):
        """
        test that an affine transform with the identity matrix does not change the input
        """
        result = affineGPU(self.raw, np.eye(4))
        self.assertTrue(np.allclose(result, result))

    def test_affine_translate(self):
        """
        test affine translation leads to a bunch of empty values up to
        but not more than the requested amount of translation
        """
        xpix = -50
        ypix = -100
        zpix = -3
        T = np.array([[1, 0, 0, xpix],
                      [0, 1, 0, ypix],
                      [0, 0, 1, zpix],
                      [0, 0, 0, 1]])
        result = affineGPU(self.raw, T)
        self.assertFalse(np.allclose(self.raw, result))
        self.assertTrue(np.alltrue(result[:-zpix] == 0))
        self.assertFalse(np.alltrue(result[:-zpix + 1] == 0))
        self.assertTrue(np.alltrue(result[:, :-xpix, :-zpix] == 0))

    def test_affine_translate_RA(self):
        """
        make sure the referenceing object works to accept transformation
        matrices in units of sample space, instead of world coordinates
        """
        xpix = -50
        ypix = -100
        zpix = -3
        T = np.array([[1, 0, 0, xpix],
                      [0, 1, 0, ypix],
                      [0, 0, 1, zpix],
                      [0, 0, 0, 1]])
        voxsize = [0.5, 0.5, 0.5]
        result = affineGPU(self.raw, T, voxsize)
        self.assertFalse(np.allclose(self.raw, result))
        self.assertTrue(np.alltrue(result[:-zpix * 2] == 0))
        self.assertFalse(np.alltrue(result[:(-zpix * 2) + 1] == 0))
        self.assertTrue(np.alltrue(result[:, :-xpix * 2, :-zpix * 2] == 0))

    def test_rotate(self):
        """
        test that rotateGPU rotates the image on the Y axis by some angle
        """
        rotated = rotateGPU(self.deskewed, 0.3, dxdata=0.1, angle=31.5)
        stored_rot = os.path.join(os.path.dirname(__file__), 'test_data', 'im_rotated.tif')
        stored_rot = imread(stored_rot)
        self.assertTrue(np.allclose(rotated, stored_rot))


if __name__ == '__main__':
    unittest.main()

