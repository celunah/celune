# pylint: disable=W0718
"""Celune's extension manager."""

from __future__ import annotations

import sys
import inspect
import traceback
import threading
import importlib.util
from pathlib import Path
from typing import Any, Type

from .base import CeluneContext, CeluneExtension


class CeluneExtensionManager:
    """Celune's extension manager."""

    def __init__(self, context: CeluneContext) -> None:
        self.context = context
        self.extensions: dict[str, CeluneExtension] = {}

    def register(self, extension_cls: Type[CeluneExtension]) -> CeluneExtension:
        """Register Celune extensions."""
        if not issubclass(extension_cls, CeluneExtension):
            raise TypeError(
                f"{extension_cls.__name__} must inherit from CeluneExtension"
            )

        instance = extension_cls(self.context)
        name = instance.name

        if name in self.extensions:
            raise ValueError(f"Extension '{name}' is already registered")

        self.extensions[name] = instance
        self.context.log(f"[Core] Registered extension: {name}")
        return instance

    def autostart_all(self) -> None:
        """Autostart all available Celune extensions."""
        started = 0
        for name, ext in self.extensions.items():
            if ext.AUTOSTART:
                self.context.log(f"[Core] Autostarting: {name}")

                def runner(e=ext, n=name):
                    try:
                        e.autostart()
                    except Exception:
                        self.context.log(
                            f"[Core] Autostart failed for {n}: {traceback.format_exc()}"
                        )

                started += 1
                threading.Thread(target=runner, daemon=True).start()

        if not started:
            self.context.log("[Core] Nothing to autostart.")

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Manually invoke a Celune extension."""
        ext = self.extensions.get(name)
        if ext is None:
            raise KeyError(f"Extension '{name}' is not registered")

        threading.Thread(
            target=ext.invoke, daemon=True, args=args, kwargs=kwargs
        ).start()

    def list_extensions(self) -> list[str]:
        """List all installed Celune extensions."""
        return list(self.extensions.keys())

    def autoload(self, folder: str = "extensions") -> None:
        """Load all Celune extensions from a directory."""
        extensions_dir = Path(folder)

        if not extensions_dir.exists():
            self.context.log(f"[Core] Extension folder not found: {extensions_dir}")
            return

        if not extensions_dir.is_dir():
            self.context.log(
                f"[Core] Extension path is not a directory: {extensions_dir}"
            )
            return

        self.context.log(f"[Core] Scanning extension folder: {extensions_dir}")

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
            except Exception:
                self.context.log(
                    f"[Core] Failed to import '{file_path.name}': {traceback.format_exc()}"
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
                except Exception:
                    self.context.log(
                        f"[Core] Failed to register '{obj.__name__}' "
                        f"from '{file_path.name}': {traceback.format_exc()}"
                    )

            if not found_any:
                self.context.log(
                    f"[Core] No extension class found in: {file_path.name}"
                )
