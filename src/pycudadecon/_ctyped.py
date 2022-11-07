import ctypes
import functools
import os
import sys
from ctypes.util import find_library
from inspect import Parameter, signature
from typing import Callable, Optional, Type, Tuple

import numpy as np

if sys.version_info >= (3, 7):
    from typing_extensions import Annotated, get_args, get_origin
else:
    # TODO: remove when py3.6 support is dropped
    from typing import Generic, GenericMeta

    from typing_extensions import Annotated, AnnotatedMeta

    def get_origin(tp):
        if isinstance(tp, AnnotatedMeta):
            return Annotated
        if isinstance(tp, GenericMeta):
            return tp.__origin__
        if tp is Generic:
            return Generic
        return None

    def get_args(tp):
        """Get type arguments with all substitutions performed."""
        if isinstance(tp, AnnotatedMeta):
            return (tp.__args__[0],) + tp.__metadata__
        if isinstance(tp, GenericMeta):
            import collections

            res = tp.__args__
            if tp.__origin__ is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return ()


class Library:
    def __init__(self, name: str, version: Tuple[int, ...]=(0, 0, 0)):
        self.name = name
        self.version = version

        _file = name
        if not _file or not os.path.exists(_file):
            _file = find_library(name.replace("lib", "", 1))  # type: ignore
            if not _file or not os.path.exists(_file):
                _file = find_library(name)  # type: ignore

        self.lib = ctypes.CDLL(_file)
        if not self.lib._name:
            raise FileNotFoundError(f"Unable to find library: {self.name}")

    def function(self, func: Callable) -> Callable:
        func_c = getattr(self.lib, func.__name__)

        sig = signature(func)
        func_c.restype = cast_type(sig.return_annotation)
        func_c.argtypes = [cast_type(p.annotation) for p in sig.parameters.values()]

        class CTypesFunction:
            def __init__(self, func):
                self._func = func
                functools.update_wrapper(self, func)

            @property
            def __signature__(self):
                return sig

            def __call__(self, *args, **kw):
                return self._func(*args, **kw)

            def __repr__(_self):
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
        np.ndarray: np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    }[hint]
