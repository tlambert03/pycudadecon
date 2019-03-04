import unittest
import os
import numpy as np
from pycudadecon.otf import makeotf
from pycudadecon.util import imread


class TestMakeOTF(unittest.TestCase):
    def test_make_otf(self):
        """
        Test that it can make an otf file from a psf
        """
        psf = os.path.join(os.path.dirname(__file__), 'test_data', 'psf.tif')
        self.result = makeotf(psf)
        self.assertTrue(os.path.isfile(self.result))

    def test_make_otf_auto(self):
        """
        Test that it can make an otf file from a psf
        """
        psf = os.path.join(os.path.dirname(__file__), 'test_data', 'psf.tif')
        self.result = makeotf(psf, otf_bgrd='auto')
        self.assertTrue(os.path.isfile(self.result))

    def text_check_otf(self):
        """
        Test that the new otf looks about right
        """
        otf = os.path.join(os.path.dirname(__file__), 'test_data', 'otf.tif')
        newotf = imread(self.result)
        known = imread(otf)
        self.assertTrue(np.allclose(newotf, known))

    def tearDown(self):
        """
        clean up the temporary OTF file we made
        """
        os.remove(os.path.join(os.path.dirname(__file__), 'test_data', 'psf_otf.tif'))


if __name__ == '__main__':
    unittest.main()
