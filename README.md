# pyCUDAdecon
This package provides a python wrapper and convenience functions for [cudaDeconv](https://github.com/dmilkie/cudaDecon) (which is a CUDA/C++ implementation of an accelerated Richardson Lucy Deconvolution<sup>1</sup>).  cudaDeconv was originally written by [Lin Shao](https://github.com/linshaova) and modified by [Dan Milkie](https://github.com/dmilkie), at Janelia Research campus.  This package makes use of a shared library interface that I wrote for cudaDecon while developing [LLSpy](https://github.com/tlambert03/LLSpy), and the code here is mostly extracted from that package to allow it to be used independently of LLSpy.

The main features are:
* radially averaged OTF generation
* OTF interpolation for voxel size independence between PSF and data volumes
* CUDA accelerated deconvolution
* Deskew, Rotation, and general affine transformations
* CUDA-based camera-correction for [sCMOS artifact correction](https://llspy.readthedocs.io/en/latest/camera.html)
* a few context managers for setup/breakdown of GPU-I/O-heavy tasks and convenience functions cause why not

### Why do we need yet another python package for deconvolution?
Honestly, we probably don't.  But since cudaDecon was recently open-sourced, and I had mostly already written this wrapper, it seemed appropriate to release it.  I do think the C++ backbone is well done, and it's relatively mature and tested at this point.  That said, there are some other good python deconvolution packages out there such as [flowdec](https://github.com/hammerlab/flowdec), and probably many others.

### gputools
Similarly, if you've stumbled upon this looking for GPU-accelerated affine transformations, then feel free to try these out, but don't miss the fantastic [gputools](https://github.com/maweigert/gputools) package, which provides OpenCL-acceleration of a number of image processing algorithms including affine transforms.

## Installation
Conda installation coming soon, with precompiled libraries for Windows, Linux, and OSX.


___

<sup>1</sup> D.S.C. Biggs and M. Andrews, Acceleration of iterative image restoration algorithms, Applied Optics, Vol. 36, No. 8, 1997.
