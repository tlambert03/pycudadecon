import os
from pathlib import Path

import numpy as np
from typing_extensions import Annotated

from ._ctyped import Library

try:
    lib = Library("libcudaDecon")
except FileNotFoundError:
    raise FileNotFoundError(
        "Unable to find library 'lidbcudaDecon'\n"
        "Please try `conda install -c conda-forge cudadecon`."
    ) from None


ndarray_uint16 = Annotated[np.ndarray, "uint16"]


@lib.function
def get_version() -> bytes:  # type: ignore [empty-body]
    """Return the version of the cudadecon library. Example b'0.7.0'."""


try:
    version = get_version().decode("utf-8")
    lib.version_string = version
except Exception:
    # the shared library doesn't have the function get_version
    # try to find it in the conda-meta directory
    _conda_prefix = os.getenv("CONDA_PREFIX")
    if _conda_prefix:
        conda_meta = Path(_conda_prefix) / "conda-meta"
        if conda_meta.exists():
            fname = next(conda_meta.glob("cudadecon*.json"), None)
            if fname is not None:
                name, *rest = fname.stem.split("-")
                _version = rest[0] if rest else None
                if _version:
                    lib.version_string = _version


@lib.function
def camcor_interface_init(  # type: ignore [empty-body]
    nx: int, ny: int, nz: int, camparam: np.ndarray
) -> int:
    """Setup for camera corrections."""


@lib.function
def camcor_interface(  # type: ignore [empty-body]
    raw_data: ndarray_uint16, nx: int, ny: int, nz: int, result: ndarray_uint16
) -> int:
    """Execute camera corrections."""


if lib.version and lib.version < (0, 6):

    @lib.function
    def RL_interface_init(  # type: ignore [empty-body]
        nx: int,
        ny: int,
        nz: int,
        dxdata: float,
        dzdata: float,
        dxpsf: float,
        dzpsf: float,
        deskewAngle: float,
        rotationAngle: float,
        outputWidth: int,
        otfpath: str,
    ) -> int:
        """Call RL_interface_init() before RL_interface when running decon.

        nx, ny, and nz: raw image dimensions
        dr: raw image pixel size
        dz: raw image Z step
        dr_psf: PSF pixel size
        dz_psf: PSF Z step
        deskewAngle: deskewing angle; usually -32.8 on Bi-chang scope and 32.8 on
        Wes scope
        rotationAngle: if 0 then no final rotation is done;
            otherwise set to the same as deskewAngle
        outputWidth: if set to 0, then calculate the output width because of
        deskewing; otherwise use this value as the output width
        OTF_file_name: file name of OTF
        """

    @lib.function
    def RL_interface(  # type: ignore [empty-body]
        raw_data: ndarray_uint16,
        nx: int,
        ny: int,
        nz: int,
        result: np.ndarray,
        raw_deskewed_result: np.ndarray,
        background: float,
        bDoRescale: bool,
        bSaveDeskewedRaw: bool,
        nIters: int,
        extraShift: int,
        napodize: int = 0,
        nZblend: int = 0,
        padVal: float = 0.0,
        bDupRevStack: bool = False,
    ) -> int:
        """Perform decon."""

else:  # include "bSkewDecon" argument

    @lib.function
    def RL_interface_init(  # type: ignore [empty-body]
        nx: int,
        ny: int,
        nz: int,
        dxdata: float,
        dzdata: float,
        dxpsf: float,
        dzpsf: float,
        deskewAngle: float,
        rotationAngle: float,
        outputWidth: int,
        bSkewedDecon: bool,
        otfpath: str,
    ) -> int:
        """Call RL_interface_init() before RL_interface when running decon.

        nx, ny, and nz: raw image dimensions
        dr: raw image pixel size
        dz: raw image Z step
        dr_psf: PSF pixel size
        dz_psf: PSF Z step
        deskewAngle: deskewing angle; usually -32.8 on Bi-chang scope and 32.8 on
        Wes scope
        rotationAngle: if 0 then no final rotation is done;
            otherwise set to the same as deskewAngle
        outputWidth: if set to 0, then calculate the output width because of
        deskewing; otherwise use this value as the output width
        bSkewedDecon: if true then do deconvolution in skewed space
        OTF_file_name: file name of OTF
        """

    @lib.function
    def RL_interface(  # type: ignore [empty-body]
        raw_data: ndarray_uint16,
        nx: int,
        ny: int,
        nz: int,
        result: np.ndarray,
        raw_deskewed_result: np.ndarray,
        background: float,
        bDoRescale: bool,
        bSaveDeskewedRaw: bool,
        nIters: int,
        extraShift: int,
        napodize: int = 0,
        nZblend: int = 0,
        padVal: float = 0.0,
        bDupRevStack: bool = False,
        bSkewedDecon: bool = False,
    ) -> int:
        """Perform decon."""


# The following are used between init and RL_interface to
# retrieve the post-deskewed image dimensions
# can be used to allocate result buffer before calling RL_interface()
@lib.function
def get_output_nx() -> int:  # type: ignore [empty-body]
    ...


@lib.function
def get_output_ny() -> int:  # type: ignore [empty-body]
    ...


@lib.function
def get_output_nz() -> int:  # type: ignore [empty-body]
    ...


@lib.function
def RL_cleanup() -> None:
    """Release GPU buffer and cleanup after deconvolution.

    Call this before program quits to release global GPUBuffer d_interpOTF.

    - Resets any bleach corrections
    - Removes OTF from GPU buffer
    - Destroys cuFFT plan
    - Releases GPU buffers
    """


@lib.function
def cuda_reset() -> None:
    """Calls `cudaDeviceReset`.

    Destroy all allocations and reset all state on the current device
    in the current process.
    """


@lib.function
def Deskew_interface(  # type: ignore [empty-body]
    raw_data: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dz: float,
    dr: float,
    deskewAngle: float,
    result: np.ndarray,
    outputWidth: int,
    extraShift: int,
    padVal: float = 0,
) -> int: ...


@lib.function
def Affine_interface(  # type: ignore [empty-body]
    raw_data: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    result: np.ndarray,
    affMat: np.ndarray,
) -> int: ...


@lib.function
def Affine_interface_RA(  # type: ignore [empty-body]
    raw_data: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    result: np.ndarray,
    affMat: np.ndarray,
) -> int: ...


try:
    otf_lib = Library("libradialft")
except FileNotFoundError:
    raise FileNotFoundError(
        "Unable to find library 'libradialft'\n"
        "Please try `conda install -c conda-forge cudadecon`."
    ) from None

if lib.version and lib.version < (0, 7):

    @otf_lib.function
    def makeOTF(
        ifiles: bytes,
        ofiles: bytes,
        lambdanm: int = 520,
        dz: float = 0.102,
        interpkr: int = 10,
        bUserBackground: bool = False,
        background: float = 90,
        NA: float = 1.25,
        NIMM: float = 1.3,
        dr: float = 0.102,
        krmax: int = 0,
        bDoCleanup: bool = False,
    ) -> None:
        """Make OTF file(s) from `ifiles`, write to `ofiles`."""

else:

    @otf_lib.function
    def makeOTF(
        ifiles: bytes,
        ofiles: bytes,
        lambdanm: int = 520,
        dz: float = 0.102,
        interpkr: int = 10,
        bUserBackground: bool = False,
        background: float = 90,
        NA: float = 1.25,
        NIMM: float = 1.3,
        dr: float = 0.102,
        krmax: int = 0,
        bDoCleanup: bool = False,
        b3Dout: bool = False,
    ) -> None:
        """Make OTF file(s) from `ifiles`, write to `ofiles`."""
