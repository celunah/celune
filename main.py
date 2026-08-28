#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Main entrypoint for the app launcher."""

import sys
import json
import importlib.util
from pathlib import Path
from typing import Optional
from types import ModuleType

# fallback for unsupported interpreters (Python 3.11 and below)
APP_NAME = "Celune"
APP_SLUG = "".join(char if char.isalnum() else "_" for char in APP_NAME.lower())

_ENTRYPOINT_MODULE: Optional[ModuleType] = None


def _fallback_string(key: str, **kwargs: str) -> str:
    """Load one lightweight fallback translation without importing Celune."""
    language_path = Path(__file__).resolve().parent / "celune" / "lang" / "en.json"
    try:
        translations = json.loads(language_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return key
    text = translations.get(key, key)
    return text.format(**kwargs) if kwargs else text


def _too_old_python() -> bool:
    """Return whether the current interpreter is too old."""
    return sys.version_info < (3, 12)


def _print_too_old_python_notice(command: Optional[str] = None) -> None:
    """Print a user-facing unsupported Python version notice, bypassing app imports."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    print(
        _fallback_string("cli.python_unsupported", app_name=APP_NAME, version=version)
    )
    print(_fallback_string("cli.python_required"))
    print(_fallback_string("cli.python_setup", app_name=APP_NAME))
    if command == "doctor":
        print(
            _fallback_string("cli.python_doctor_unavailable", app_name=APP_NAME.lower())
        )


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
    resolved_argv = sys.argv if argv is None else argv
    command = resolved_argv[1].strip().lower() if len(resolved_argv) >= 2 else None

    if _too_old_python():
        _print_too_old_python_notice(command)
        sys.exit(6)

    _load_entrypoint_module().main(resolved_argv)


if __name__ == "__main__":
    main(sys.argv)
