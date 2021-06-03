# pyCUDAdecon

This package provides a python wrapper and convenience functions for
[cudaDeconv](https://github.com/scopetools/cudaDecon), which is a CUDA/C++
implementation of an accelerated Richardson Lucy Deconvolution
algorithm {cite}`biggs_97`. ``cudaDeconv`` was originally
written by [Lin Shao](https://github.com/linshaova) and modified by [Dan Milkie](https://github.com/dmilkie), at Janelia Research campus.  This package
makes use of a shared library interface that I wrote for cudaDecon while
developing [LLSpy](https://github.com/tlambert03/LLSpy), that adds a couple
additional kernels for affine transformations and camera corrections.

- CUDA accelerated deconvolution with a handful of artifact-reducing features.
- radially averaged OTF generation with interpolation for voxel size
  independence between PSF and data volumes
- 3D deskew, rotation, general affine transformations
- CUDA-based camera-correction for
  [sCMOS artifact correction](https://llspy.readthedocs.io/en/latest/camera.html)

## Install

Install (Linux and Windows) from conda forge:

```bash
conda install -c conda-forge pycudadecon
```

see GPU requirements in [Installation](installation.md).

## Quickstart

If you have a PSF and an image volume and you just want to get started, check
out the {func}`pycudadecon.decon` function, which should be able to handle most
basic applications.

```python
from pycudadecon import decon

image_path = '/path/to/some_image.tif'
psf_path = '/path/to/psf_3D.tif'
result = decon(image_path, psf_path)
```

For finer-tuned control, you may wish to make an OTF file from your PSF using
{func}`~pycudadecon.make_otf`, and then use the {class}`~pycudadecon.RLContext`
context manager to setup the GPU for use with the {func}`~pycudadecon.rl_decon`
function.  (Note all images processed in the same context must have the same
input shape).

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
        result = rl_decon(image, output_shape=ctx.out_shape)
        # do something with result...
```

If you have a 3D PSF volume, the {class}`~pycudadecon.TemporaryOTF` context
manager facilitates temporary OTF generation...

```python
# continuing with the variables from the previous example...
psf_path = "/path/to/psf_3D.tif"

with TemporaryOTF(psf) as otf:
    with RLContext(imshape, otf.path, dz) as ctx:
        for impath in imlist:
            image = tifffile.imread(impath)
            result = rl_decon(image, output_shape=ctx.out_shape)
            # do something with result...
```

... and that bit of code is essentially what the {func}`~pycudadecon.decon`
function is doing, with a little bit of additional conveniences added in.

*Each of these functions has many options and accepts multiple keyword
arguments. For further information, see the documentation for the respective
functions.*

## References

```{bibliography}
:style: unsrt
```
