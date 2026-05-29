# SPDX-License-Identifier: MIT
"""Celune's extension manager methods and classes."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import threading
import traceback
from typing import Type

from .base import CeluneContext, CeluneExtension
from ..exceptions import InvalidExtensionError, ExtensionAlreadyRegisteredError
from ..utils import format_error


class CeluneExtensionManager:
    """Celune's extension manager."""

    def __init__(self, context: CeluneContext) -> None:
        self.context = context
        self.extensions: dict[str, CeluneExtension] = {}
        self.auto_started = False

    def register(self, extension_cls: Type[CeluneExtension]) -> CeluneExtension:
        """Register a Celune extension class.

        Args:
            extension_cls: The extension class to instantiate and register.

        Returns:
            CeluneExtension: The registered extension instance.

        Raises:
            InvalidExtensionError: The object is not a CeluneExtension subclass.
            ExtensionAlreadyRegisteredError: An extension with the same name is already registered.
        """
        if not inspect.isclass(extension_cls) or not issubclass(
            extension_cls, CeluneExtension
        ):
            raise InvalidExtensionError(
                f"{extension_cls.__name__} must inherit from CeluneExtension"
            )

        instance = extension_cls(self.context)
        name = instance.name

        if name in self.extensions:
            self.context.log(f"[Core] {name} is already registered", "warning")
            raise ExtensionAlreadyRegisteredError(
                f"extension '{name}' is already registered"
            )

        self.extensions[name] = instance
        self.context.log_dev(f"[Core] Registered extension: {name}")
        return instance

    def autostart_all(self) -> None:
        """Autostart all available Celune extensions."""
        if self.auto_started:
            self.context.log(
                "[Core] Cannot autostart Celune extensions more than one time.",
                "warning",
            )
            return

        started = 0
        for name, ext in self.extensions.items():
            if ext.AUTOSTART:
                self.context.log_dev(f"[Core] Auto-starting: {name}")

                def runner(e=ext, n=name):
                    try:
                        e.autostart()
                    except Exception as ex:
                        self.context.log(
                            f"[Core] Could not autostart {n}: {traceback.format_exc() if self.context.dev else ex}",
                            "warning",
                        )

                started += 1
                threading.Thread(target=runner, daemon=True).start()

        if not started:
            self.context.log_dev("[Core] No extensions to autostart.")
        else:
            self.auto_started = True

    def invoke(self, name: str, *args, **kwargs) -> None:
        """Manually invoke a Celune extension.

        Args:
            name: The registered extension name to invoke.
            args: Positional arguments forwarded to the extension.
            kwargs: Keyword arguments forwarded to the extension.

        Raises:
            InvalidExtensionError: The requested extension is not registered.
        """
        ext = self.extensions.get(name)
        if ext is None:
            raise InvalidExtensionError(f"extension '{name}' is not registered")

        def runner() -> None:
            try:
                ext.invoke(*args, **kwargs)
            except Exception as ex:
                self.context.log(
                    f"[Core] Failed to invoke '{name}': "
                    f"{format_error(ex, self.context.dev)}",
                    "warning",
                )

        threading.Thread(target=runner, daemon=True).start()

    def list_extensions(self) -> list[str]:
        """List all installed Celune extensions.

        Returns:
            list[str]: The registered extension names.
        """
        return list(self.extensions.keys())

    def autoload(self, folder: str = "extensions") -> None:
        """Load all Celune extensions from a directory.

        Args:
            folder: The directory containing extension Python modules.
        """
        extensions_dir = Path(folder)

        if not extensions_dir.exists():
            self.context.log(
                f"[Core] Extension folder not found: {extensions_dir}", "warning"
            )
            self.context.log("Extensions will not be available.", "warning")
            return

        if not extensions_dir.is_dir():
            self.context.log(
                f"[Core] Extension path is not a directory: {extensions_dir}"
            )
            self.context.log("Extensions will not be available.", "warning")
            return

        self.context.log_dev(f"[Core] Scanning extension folder: {extensions_dir}")

        for file_path in sorted(extensions_dir.glob("*.py")):
            if file_path.name.startswith("_"):
                continue

            module_name = f"user_extension_{file_path.stem}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    self.context.log(
                        f"[Core] Could not load spec for: {file_path.name}"
                    )
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as e:
                self.context.log(
                    f"[Core] Failed to import '{file_path.name}': "
                    f"{traceback.format_exc() if self.context.dev else e}",
                    "warning",
                )
                continue

            found_any = False

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, CeluneExtension):
                    continue

                if obj is CeluneExtension:
                    continue

                if obj.__module__ != module.__name__:
                    continue

                try:
                    self.register(obj)
                    found_any = True
                except Exception as e:
                    self.context.log(
                        f"[Core] Failed to register '{obj.__name__}' "
                        f"from '{file_path.name}': {traceback.format_exc() if self.context.dev else e}",
                        "warning",
                    )

            if not found_any:
                self.context.log(
                    f"[Core] {file_path.name} is not a Celune extension, skipping",
                    "warning",
                )
