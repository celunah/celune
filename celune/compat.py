# SPDX-License-Identifier: Apache-2.0
"""Compatibility boundaries for optional runtime integrations."""

import sys
import importlib
import threading
from enum import Enum
from typing import cast
from contextlib import contextmanager
from collections.abc import Callable, Generator

import torch
import torch.utils._pytree

__all__ = ["torchao_compatibility"]


_TORCHAO_IMPORT_LOCK = threading.RLock()
_PYTREE_REGISTER_CONSTANT_NAME = "register_constant"
_TORCHAO_DECORATOR_NAME = "register_as_pytree_constant"


def _is_native_opaque_enum(value: type) -> bool:
    """Return whether PyTorch natively treats an Enum class as opaque."""
    if not issubclass(value, Enum):
        return False

    try:
        opaque_object = importlib.import_module("torch._library.opaque_object")
    except ImportError:
        return False
    is_opaque_type = getattr(opaque_object, "is_opaque_type", None)
    if not callable(is_opaque_type):
        return False
    return cast(Callable[[type], bool], is_opaque_type)(value)


def _register_torchao_constant(value: type) -> type:
    """Register non-Enum TorchAO constants using the active PyTorch API."""
    if not _is_native_opaque_enum(value):
        torch.utils._pytree.register_constant(value)
    return value


def _patch_torchao_decorator() -> None:
    """Replace TorchAO's deprecated Enum registration decorator when loaded."""
    torchao_utils = sys.modules.get("torchao.utils")
    if torchao_utils is not None:
        setattr(
            torchao_utils,
            _TORCHAO_DECORATOR_NAME,
            _register_torchao_constant,
        )


@contextmanager
def torchao_compatibility() -> Generator[None, None, None]:
    """Suppress redundant TorchAO Enum registration for the active operation.

    PyTorch versions that natively treat Enum classes as opaque compile values
    warn when TorchAO registers those same classes as pytree constants. The
    wrapper is active only while Celune imports or initializes an integration
    that may load TorchAO, and TorchAO's decorator is retained afterward for
    modules imported later in the process.
    """
    pytree = torch.utils._pytree
    register_constant = cast(Callable[[type], None], pytree.register_constant)

    def register_constant_compat(cls: type) -> None:
        if not _is_native_opaque_enum(cls):
            register_constant(cls)

    with _TORCHAO_IMPORT_LOCK:
        _patch_torchao_decorator()
        setattr(
            pytree,
            _PYTREE_REGISTER_CONSTANT_NAME,
            register_constant_compat,
        )
        try:
            yield
        finally:
            setattr(pytree, _PYTREE_REGISTER_CONSTANT_NAME, register_constant)
            _patch_torchao_decorator()
