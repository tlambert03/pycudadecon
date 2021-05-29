import os
import unittest

import numpy as np

from pycudadecon import RLContext, decon, rl_cleanup, rl_decon, rl_init
from pycudadecon.util import imread


class TestDecon(unittest.TestCase):
    def setUp(self):
        self.raw = os.path.join(os.path.dirname(__file__), "test_data", "im_raw.tif")
        self.deskewed = os.path.join(
            os.path.dirname(__file__), "test_data", "im_deskewed.tif"
        )
        self.stored_decon = os.path.join(
            os.path.dirname(__file__), "test_data", "im_decon.tif"
        )
        self.otf = os.path.join(os.path.dirname(__file__), "test_data", "otf.tif")
        self.psf = os.path.join(os.path.dirname(__file__), "test_data", "psf.tif")

        self.raw = imread(self.raw)
        self.deskewed = imread(self.deskewed)
        self.stored_decon = imread(self.stored_decon)

        self.config = {
            "dzdata": 0.3,
            "deskew": 0,
            "n_iters": 10,
            "background": 98,
        }

    def test_decon(self):
        """
        test that we can deconvolve an image
        """
        rl_init(self.deskewed.shape, self.otf, **self.config)
        decon_result = rl_decon(self.deskewed, **self.config)
        self.assertTrue(np.allclose(decon_result, self.stored_decon))
        rl_cleanup()

    def test_decon_context(self):
        """
        test that we can deconvolve an image using the context manager
        """
        with RLContext(self.deskewed.shape, self.otf, **self.config) as ctx:
            decon_result = rl_decon(
                self.deskewed, output_shape=ctx.out_shape, **self.config
            )
        self.assertTrue(np.allclose(decon_result, self.stored_decon))

    def test_decon_wrapper_with_otf(self):
        """
        test that the
        """
        decon_result = decon(self.deskewed, self.otf, **self.config)
        self.assertTrue(np.allclose(decon_result, self.stored_decon))

    def test_decon_wrapper_with_psf(self):
        """
        test that we can deconvolve an image
        """
        decon_result = decon(self.deskewed, self.psf, **self.config)
        self.assertTrue(np.allclose(decon_result, self.stored_decon))

    def test_decon_wrapper_with_psf_array(self):
        """
        test that we can deconvolve an image
        """
        psf = imread(self.psf)
        decon_result = decon(self.deskewed, psf, **self.config)
        self.assertTrue(np.allclose(decon_result, self.stored_decon))

    def test_decon_wrapper_with_many_inputs(self):
        """
        test that the
        """
        images = [self.deskewed, self.deskewed, self.deskewed]
        decon_results = decon(images, self.otf, **self.config)
        self.assertTrue(all([np.allclose(d, self.stored_decon) for d in decon_results]))

    def test_decon_wrapper_with_variable_shapes(self):
        """
        test that the
        """
        images = [
            self.deskewed,
            self.deskewed[:, 4:-4, 4:-4],
            self.deskewed[2:-2, 16:-16, 16:-16],
        ]
        decon(images, self.otf, **self.config)

    def test_decon_wrapper_save_deskewed(self):
        """
        test that the
        """
        config = dict(self.config)
        config["deskew"] = 31.5
        decon_result = decon(self.raw, self.otf, save_deskewed=True, **config)
        self.assertTrue(len(decon_result) == 2)
        # self.assertTrue(np.allclose(self.stored_decon, decon_result[1]))

    def tearDown(self):
        rl_cleanup()


if __name__ == "__main__":
    unittest.main()
