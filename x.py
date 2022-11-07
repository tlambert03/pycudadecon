from pycudadecon.util import imread
from pycudadecon import rl_init

deskewed_image = imread('tests/test_data/im_deskewed.tif')
rl_init(deskewed_image.shape, 'tests/test_data/otf.tif')
print("done")
