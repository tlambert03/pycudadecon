from pathlib import Path

import pytest

from pycudadecon.util import imread

DATA = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def raw_image():
    return imread(str(DATA / "im_raw.tif"))


@pytest.fixture(scope="session")
def deskewed_image():
    return imread(str(DATA / "im_deskewed.tif"))


@pytest.fixture(scope="session")
def decon_image():
    return imread(str(DATA / "im_decon.tif"))


@pytest.fixture(scope="session")
def rotated_image():
    return imread(str(DATA / "im_rotated.tif"))
