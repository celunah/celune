#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Main entrypoint for the app launcher."""

import sys
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional

from celune.constants import APP_NAME, APP_SLUG

_ENTRYPOINT_MODULE: Optional[ModuleType] = None


def load_entrypoint_module() -> ModuleType:
    """Public interface for loading the app entrypoint module.

    Returns:
        ModuleType: The return value of _load_entrypoint_module().
    """
    return _load_entrypoint_module()


def _load_entrypoint_module() -> ModuleType:
    """Load entrypoint logic without eagerly importing unnecessary app modules."""
    global _ENTRYPOINT_MODULE
    if _ENTRYPOINT_MODULE is not None:
        return _ENTRYPOINT_MODULE

    entrypoint_path = Path(__file__).resolve().parent / "celune" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location(
        f"_{APP_SLUG}_entrypoint", entrypoint_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {APP_NAME} entrypoint from {entrypoint_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _ENTRYPOINT_MODULE = module
    return module


def main(argv: Optional[list[str]] = None) -> None:
    """Run the main entrypoint handler.

    Args:
        argv: Arguments to pass through to the entrypoint handler.
    """
    _load_entrypoint_module().main(sys.argv if argv is None else argv)


if __name__ == "__main__":
    main(sys.argv)
