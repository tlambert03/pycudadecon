__version__ = "0.2.0"
from typing import Any

lib: Any
try:
    from . import _libwrap as lib
except FileNotFoundError as e:
    import warnings

    t = e
    warnings.warn(f"\n{e}\n\nMost functionality will fail!\n")

    class _stub:
        def __getattr__(self, name):
            raise t

    lib = _stub()


from .affine import affineGPU, deskewGPU, rotateGPU  # noqa
from .deconvolution import (  # noqa
    RLContext,
    decon,
    rl_cleanup,
    rl_context,
    rl_decon,
    rl_init,
)
from .otf import TemporaryOTF, make_otf  # noqa

__all__ = [
    "__version__",
    "affineGPU",
    "decon",
    "deskewGPU",
    "make_otf",
    "rl_cleanup",
    "rl_decon",
    "rl_init",
    "RLContext",
    "rl_context",
    "rotateGPU",
    "TemporaryOTF",
]
