"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation

from napari.types import ImageData
from pycudadecon.deconvolution import decon


@magic_factory(call_button="Deconvolve")
def deconvolve(
    image: ImageData,
    psf: ImageData,
    image_pixel_size=0.1,
    image_zstep=0.5,
    psf_pixel_size=0.1,
    psf_zstep=0.1,
    iterations=10,
) -> ImageData:
    return decon(
        image,
        psf,
        dxdata=image_pixel_size,
        dzdata=image_zstep,
        dxpsf=psf_pixel_size,
        dzpsf=psf_zstep,
        n_iters=iterations,
    )


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return deconvolve, {"name": "CUDA-Deconvolution"}
