# pyCUDAdecon

This package provides a python wrapper and convenience functions for
[cudaDecon](https://github.com/scopetools/cudaDecon), which is a CUDA/C++
implementation of an accelerated Richardson Lucy Deconvolution
algorithm<sup>1</sup>.

* CUDA accelerated deconvolution with a handful of artifact-reducing features.
* radially averaged OTF generation with interpolation for voxel size
  independence between PSF and data volumes
* 3D deskew, rotation, general affine transformations
* CUDA-based camera-correction for [sCMOS artifact correction](https://llspy.readthedocs.io/en/latest/camera.html)


### Install

The conda package includes the required pre-compiled libraries for Windows and Linux. See GPU driver requirements [below](#gpu-requirements)

```sh
conda install -c conda-forge pycudadecon
```

*macOS is not supported*

### 📖   &nbsp; [Documentation](http://www.talleylambert.com/pycudadecon)


### GPU requirements

This software requires a CUDA-compatible NVIDIA GPU. The underlying cudadecon
libraries have been compiled against different versions of the CUDA toolkit.
The required CUDA libraries are bundled in the conda distributions so you don't
need to install the CUDA toolkit separately.  If desired, you can pick which
version of CUDA you'd like based on your needs, but please note that different
versions of the CUDA toolkit have different GPU driver requirements:

To specify a specific cudatoolkit version, install as follows (for instance, to
use `cudatoolkit=10.2`)

```sh
conda install -c conda-forge pycudadecon cudatoolkit=10.2
```

| CUDA | Linux driver | Win driver |
| ---- | ------------ | ---------- |
| 10.2 | ≥ 440.33     | ≥ 441.22   |
| 11.0 | ≥ 450.36.06  | ≥ 451.22   |
| 11.1 | ≥ 455.23     | ≥ 456.38   |
| 11.2 | ≥ 460.27.03  | ≥ 460.82   |


If you run into trouble, feel free to [open an
issue](https://github.com/tlambert03/pycudadecon/issues) and describe your
setup.


## Usage


The [`pycudadecon.decon()`](https://pycudadecon.readthedocs.io/en/latest/deconvolution.html#pycudadecon.decon) function is designed be able to handle most basic applications:

```python
from pycudadecon import decon

# pass filenames of an image and a PSF
result = decon('/path/to/3D_image.tif', '/path/to/3D_psf.tif')

# decon also accepts numpy arrays
result = decon(img_array, psf_array)

# the image source can also be a sequence of arrays or paths
result = decon([img_array, '/path/to/3D_image.tif'], psf_array)

# see docstrings for additional parameter options
```

For finer-tuned control, you may wish to make an OTF file from your PSF using [`pycudadecon.make_otf()`](https://pycudadecon.readthedocs.io/en/latest/otf.html?highlight=make_otf#pycudadecon.make_otf), and then use the [`pycudadecon.RLContext`](https://pycudadecon.readthedocs.io/en/latest/deconvolution.html?highlight=RLContext#pycudadecon.RLContext) context manager to setup the GPU for use with the [`pycudadecon.rl_decon()`](https://pycudadecon.readthedocs.io/en/latest/deconvolution.html?highlight=RLContext#pycudadecon.rl_decon) function.  (Note all images processed in the same context must have the same input shape).

```python
from pycudadecon import RLContext, rl_decon
from glob import glob
import tifffile

image_folder = '/path/to/some_images/'
imlist = glob(image_folder + '*488*.tif')
otf_path = '/path/to/pregenerated_otf.tif'

with tifffile.TiffFile(imlist[0]) as tf:
    imshape = tf.series[0].shape

with RLContext(imshape, otf_path, dz) as ctx:
    for impath in imlist:
        image = tifffile.imread(impath)
        result = rl_decon(image, ctx.out_shape)
        # do something with result...
```

If you have a 3D PSF volume, the [`pycudadecon.TemporaryOTF`](https://pycudadecon.readthedocs.io/en/latest/otf.html?highlight=temporaryotf#pycudadecon.TemporaryOTF) context manager facilitates temporary OTF generation...

```python
 # continuing with the variables from the previous example...
 psf_path = "/path/to/psf_3D.tif"
 with TemporaryOTF(psf) as otf:
     with RLContext(imshape, otf.path, dz) as ctx:
         for impath in imlist:
             image = tifffile.imread(impath)
             result = rl_decon(image, ctx.out_shape)
             # do something with result...
```

... and that bit of code is essentially what the [`pycudadecon.decon()`](https://pycudadecon.readthedocs.io/en/latest/deconvolution.html#pycudadecon.decon) function is doing, with a little bit of additional conveniences added in.

*Each of these functions has many options and accepts multiple keyword arguments. See the [documentation](https://pycudadecon.readthedocs.io/en/latest/index.html) for further information on the respective functions.*

For examples and information on affine transforms, volume rotations, and deskewing (typical of light sheet volumes acquired with stage-scanning), see the [documentation on Affine Transformations](https://pycudadecon.readthedocs.io/en/latest/affine.html)
___

<sup>1</sup> D.S.C. Biggs and M. Andrews, Acceleration of iterative image restoration algorithms, Applied Optics, Vol. 36, No. 8, 1997. https://doi.org/10.1364/AO.36.001766
