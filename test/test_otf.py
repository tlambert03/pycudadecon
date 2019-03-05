import unittest
import os
import numpy as np
from pycudadecon import makeotf
from pycudadecon.util import imread, is_otf


class TestMakeOTF(unittest.TestCase):
    def setUp(self):
        self.psf = os.path.join(os.path.dirname(__file__), 'test_data', 'psf.tif')
        self.otf = os.path.join(os.path.dirname(__file__), 'test_data', 'otf.tif')
        self.dest = os.path.join(os.path.dirname(__file__), 'test_data', 'otf_test.tif')

    def test_make_otf(self):
        """
        Test that it can make an otf file from a psf
        """
        result = makeotf(self.psf, self.dest)
        self.assertTrue(os.path.isfile(result))
        known = imread(self.otf)

        # make sure it matches the known OTF
        self.assertTrue(np.allclose(imread(result), known))
        os.remove(self.dest)

    def test_make_otf_auto(self):
        """
        Test that it can make an otf file from a psf
        """
        result = makeotf(self.psf, self.dest, otf_bgrd='auto')
        self.assertTrue(os.path.isfile(result))
        os.remove(self.dest)

    def test_is_otf(self):
        """
        Test that it can make an otf file from a psf
        """
        self.assertTrue(is_otf(self.otf))

    def test_is_not_otf(self):
        """
        Test that it can make an otf file from a psf
        """
        self.assertFalse(is_otf(self.psf))

    def tearDown(self):
        """
        clean up the temporary OTF file we made
        """
        try:
            os.remove(self.dest)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
