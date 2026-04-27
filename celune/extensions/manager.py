"""Celune's extension manager."""

from __future__ import annotations

import sys
import inspect
import traceback
import threading
import importlib.util
from pathlib import Path
from typing import Any, Type

from celune.exceptions import InvalidExtensionError, ExtensionAlreadyRegisteredError
from .base import CeluneContext, CeluneExtension


class CeluneExtensionManager:
    """Celune's extension manager."""

    def __init__(self, context: CeluneContext) -> None:
        """Initialize the extension manager.

        Args:
            context: Shared context passed to registered extensions.

        Returns:
            None: This constructor prepares extension registry state.
        """
        self.context = context
        self.extensions: dict[str, CeluneExtension] = {}
        self.auto_started = False

    def register(self, extension_cls: Type[CeluneExtension]) -> CeluneExtension:
        """Register a Celune extension class.

        Args:
            extension_cls: The extension class to instantiate and register.

        Returns:
            CeluneExtension: The registered extension instance.
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
                f"Extension '{name}' is already registered"
            )

        self.extensions[name] = instance
        if self.context.dev:
            self.context.log(f"[Core] Registered extension: {name}")
        return instance

    def autostart_all(self) -> None:
        """Autostart all available Celune extensions.

        Returns:
            None: Matching extensions are started on background threads.
        """
        if self.auto_started:
            self.context.log(
                "[Core] Cannot autostart Celune extensions more than one time.",
                "warning",
            )
            return

        started = 0
        for name, ext in self.extensions.items():
            if ext.AUTOSTART:
                if self.context.dev:
                    self.context.log(f"[Core] Auto-starting: {name}")

                def runner(e=ext, n=name):
                    """Run one extension autostart hook.

                    Args:
                        e: Extension instance to start.
                        n: Extension display name for logging.

                    Returns:
                        None: This worker logs autostart failures.
                    """
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
            if self.context.dev:
                self.context.log("[Core] No extensions to autostart.")
        else:
            self.auto_started = True

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Manually invoke a Celune extension.

        Args:
            name: The registered extension name to invoke.
            *args: Positional arguments forwarded to the extension.
            **kwargs: Keyword arguments forwarded to the extension.

        Returns:
            Any: The invocation thread is started asynchronously, so this returns
                ``None``.
        """
        ext = self.extensions.get(name)
        if ext is None:
            raise InvalidExtensionError(f"Extension '{name}' is not registered")

        threading.Thread(
            target=ext.invoke, daemon=True, args=args, kwargs=kwargs
        ).start()

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

        Returns:
            None: This method imports and registers any valid extensions it finds.
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

        if self.context.dev:
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
