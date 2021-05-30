from pathlib import Path

import numpy as np

from pycudadecon import make_otf
from pycudadecon.util import imread, is_otf

DATA = Path(__file__).parent / "test_data"
PSF_PATH = str(DATA / "psf.tif")
OTF_PATH = str(DATA / "otf.tif")


def test_make_otf(tmp_path):
    """Test that it can make an otf file from a psf."""
    dest = str(tmp_path / "otf_test.tif")
    result = make_otf(PSF_PATH, dest)
    assert Path(result).exists()

    # make sure it matches the known OTF
    assert np.allclose(imread(result), imread(OTF_PATH), atol=0.05)


def test_make_otf_auto(tmp_path):
    """Test that it can make an otf file from a psf with autobackground"""
    dest = str(tmp_path / "otf_test.tif")
    result = make_otf(PSF_PATH, dest, otf_bgrd=None)
    assert Path(result).exists()


def test_is_otf():
    """Test is_otf validates files"""
    assert is_otf(OTF_PATH)
    assert not is_otf(PSF_PATH)
