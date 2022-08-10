from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari.types


def deconvolve(
    image: "napari.types.ImageData",
    psf: "napari.types.ImageData",
    image_pixel_size=0.1,
    image_zstep=0.5,
    psf_pixel_size=0.1,
    psf_zstep=0.1,
    iterations=10,
) -> "napari.types.ImageData":
    """Deconvolve `image` using `psf`.

    Parameters
    ----------
    image : ImageData
        image array
    psf : ImageData
        psf array
    image_pixel_size : float, optional
        image pixel size in microns, by default 0.1
    image_zstep : float, optional
        image z step in microns, by default 0.5
    psf_pixel_size : float, optional
        psf pixel size in microns, by default 0.1
    psf_zstep : float, optional
        psf z step size in microns, by default 0.1
    iterations : int, optional
        number of iterations, by default 10

    Returns
    -------
    ImageData
        deconvolved image.
    """
    from pycudadecon.deconvolution import decon

    return decon(
        image,
        psf,
        dxdata=image_pixel_size,
        dzdata=image_zstep,
        dxpsf=psf_pixel_size,
        dzpsf=psf_zstep,
        n_iters=iterations,
    )
