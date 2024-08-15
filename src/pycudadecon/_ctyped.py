import ctypes
import functools
import os
import warnings
from ctypes.util import find_library
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Type

import numpy as np
from typing_extensions import Annotated, get_args, get_origin

if TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")


class Library:
    def __init__(self, name: str) -> None:
        self.name = name
        self.version_string = ""

        _file = name
        if not _file or not os.path.exists(_file):
            _file = find_library(name.replace("lib", "", 1))  # type: ignore
            if not _file:
                _file = find_library(name)  # type: ignore
        if not _file:
            raise FileNotFoundError(f"Unable to find library: {name}")

        self.lib = ctypes.CDLL(_file)
        if not self.lib._name:
            raise FileNotFoundError(f"Unable to find library: {self.name}")

    @property
    def version(self) -> Tuple[int, ...]:
        if not self.version_string:
            return ()
        return tuple(map(int, self.version_string.split(".")[:3]))

    def function(self, func: "Callable[P, R]") -> "Callable[P, R]":
        try:
            func_c = getattr(self.lib, func.__name__)
        except AttributeError:
            warnings.warn(
                f"Unable to find function: {func.__name__} in {self.name}", stacklevel=2
            )

            def _missing(*args: Any, **kw: Any) -> Any:
                raise NotImplementedError(
                    f"Sorry, we were unable to find the function "
                    f"{func.__name__!r} in library {self.name!r}."
                )

            return _missing

        sig = signature(func)
        func_c.restype = cast_type(sig.return_annotation)
        func_c.argtypes = [cast_type(p.annotation) for p in sig.parameters.values()]

        class CTypesFunction:
            def __init__(self, func: "Callable[P, R]") -> None:
                self._func = func
                functools.update_wrapper(self, func)

            @property
            def __signature__(self) -> "Signature":
                return sig

            def __call__(self, *args: "P.args", **kw: "P.kwargs") -> "R":
                return self._func(*args, **kw)

            def __repr__(_self) -> str:
                return (
                    f"<CTypesFunction: {os.path.basename(self.name)}.{func.__name__}>"
                )

        return CTypesFunction(func_c)


def cast_type(hint: Type) -> Optional[Type]:
    if isinstance(hint, str):
        raise ValueError("forward ref typehints not supported")

    if get_origin(hint) is Annotated:
        args = get_args(hint)
        if args and args[0] is np.ndarray:
            c_type = np.ctypeslib.as_ctypes_type(np.dtype(args[1]))
            return np.ctypeslib.ndpointer(c_type, flags="C_CONTIGUOUS")

    return {
        None: None,
        Parameter.empty: None,
        bool: ctypes.c_bool,
        float: ctypes.c_float,
        int: ctypes.c_int,
        str: ctypes.c_char_p,
        bytes: ctypes.c_char_p,
        np.ndarray: np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    }[hint]
