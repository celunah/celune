# SPDX-License-Identifier: MIT
"""Celune's extension manager methods and classes."""

import sys
import inspect
import warnings
import threading
import traceback
import importlib.util
from pathlib import Path
from types import ModuleType
from dataclasses import dataclass
from typing import Callable, Optional, cast

from ..typing.events import EventName
from ..typing.events import EventPayload
from ..utils import format_error, discard
from ..dataclasses.events import ReadyEvent
from .base import CeluneContext, CeluneExtension
from ..exceptions import InvalidExtensionError, ExtensionAlreadyRegisteredError
from .events import (
    EventDispatcher,
    RegisteredEventHandler,
    iter_subscriptions,
)


@dataclass(frozen=True)
class _ModuleRegistration:
    """Track callbacks registered from one imported extension module."""

    module_name: str
    owner_key: str


class CeluneExtensionManager:
    """Celune's extension manager."""

    def __init__(
        self,
        context: CeluneContext,
        dispatcher: Optional[EventDispatcher] = None,
    ) -> None:
        self.context = context
        self.dispatcher = dispatcher or EventDispatcher(
            log_warning=context.log,
            dev=context.dev,
        )
        self.extensions: dict[str, CeluneExtension] = {}
        self._event_registrations: dict[str, list[RegisteredEventHandler]] = {}
        self._module_registrations: dict[str, _ModuleRegistration] = {}
        self._extension_modules: dict[str, str] = {}
        self.auto_started = False

    def register(self, extension_cls: type[CeluneExtension]) -> CeluneExtension:
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
        self._extension_modules[name] = extension_cls.__module__
        self._register_extension_handlers(instance)
        self._register_legacy_autostart_handler(instance)
        self.context.log_dev(f"[Core] Registered extension: {name}")
        return instance

    def unregister(self, name: str) -> None:
        """Unregister one extension and any event handlers it owns.

        Args:
            name: Registered extension name to remove.

        Raises:
            InvalidExtensionError: The requested extension is not currently registered.
        """
        extension = self.extensions.pop(name, None)
        if extension is None:
            raise InvalidExtensionError(f"extension '{name}' is not registered")

        self._unregister_owner(name)
        module_name = self._extension_modules.pop(name, "")
        if module_name and module_name not in self._extension_modules.values():
            module_registration = self._module_registrations.pop(module_name, None)
            if module_registration is not None:
                self._unregister_owner(module_registration.owner_key)

        self.context.log_dev(f"[Core] Unregistered extension: {name}")

    def unregister_all(self) -> None:
        """Unregister all loaded extensions and auto-registered handlers."""
        for name in list(self.extensions.keys()):
            self.unregister(name)

    def emit(self, event_name: EventName, event: EventPayload) -> None:
        """Forward one event to the shared dispatcher.

        Args:
            event_name: Event name to emit.
            event: Typed payload instance to deliver.
        """
        self.dispatcher.emit(event_name, event)

    def autostart_all(self) -> None:
        """Run deprecated legacy autostart handlers."""
        if self.auto_started:
            self.context.log(
                "[Core] Cannot autostart Celune extensions more than one time.",
                "warning",
            )
            return

        warnings.warn(
            "CeluneExtensionManager.autostart_all() is deprecated, "
            "please use @celune.subscribe('ready') in your extensions instead",
            DeprecationWarning,
            stacklevel=2,
        )
        started = 0
        for name, ext in self.extensions.items():
            if self._uses_legacy_autostart(ext):
                self.context.log_dev(f"[Core] Running autostart for: {name}")

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
            module_handlers_registered = self._register_module_handlers(
                module,
                owner_name=file_path.stem,
            )
            found_any = module_handlers_registered

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

    def _register_extension_handlers(self, extension: CeluneExtension) -> None:
        """Auto-register decorated event handlers declared on one extension class."""
        handlers: list[RegisteredEventHandler] = []
        for _, function in inspect.getmembers(type(extension), inspect.isfunction):
            subscriptions = iter_subscriptions(function)
            if not subscriptions:
                continue

            bound = getattr(extension, function.__name__)
            if not callable(bound):
                continue

            callback = bound
            for subscription in subscriptions:
                if not subscription.enabled:
                    self.context.log_dev(
                        f"[Core] Disabled subscription skipped for extension: {extension.name}"
                    )
                    continue
                registered = self._register_handler(
                    owner_name=extension.name,
                    event_name=subscription.event_name,
                    callback=cast(Callable[..., None], callback),
                )
                handlers.append(registered)

        if handlers:
            self._event_registrations[extension.name] = handlers

    def _register_legacy_autostart_handler(self, extension: CeluneExtension) -> None:
        """Bridge deprecated ``autostart()`` handlers onto the ``ready`` event."""
        if not self._uses_legacy_autostart(extension):
            return

        warnings.warn(
            "CeluneExtension.autostart() is deprecated, "
            "please use @celune.subscribe('ready') instead",
            DeprecationWarning,
            stacklevel=2,
        )

        def legacy_ready_callback(
            event: ReadyEvent, ext: CeluneExtension = extension
        ) -> None:
            discard(event)

            def runner() -> None:
                try:
                    ext.autostart()
                except Exception as ex:
                    self.context.log(
                        f"[Core] Could not autostart {ext.name}: "
                        f"{format_error(ex, self.context.dev)}",
                        "warning",
                    )

            threading.Thread(target=runner, daemon=True).start()

        handler = self._register_handler(
            owner_name=extension.name,
            event_name="ready",
            callback=legacy_ready_callback,
        )
        self._event_registrations.setdefault(extension.name, []).append(handler)

    @staticmethod
    def _uses_legacy_autostart(extension: CeluneExtension) -> bool:
        """Return whether one extension still relies on deprecated autostart hooks."""
        extension_type = type(extension)
        return extension_type.AUTOSTART or (
            extension_type.autostart is not CeluneExtension.autostart
        )

    def _register_module_handlers(
        self,
        module: ModuleType,
        owner_name: str,
    ) -> bool:
        """Auto-register decorated module-level handlers from one extension module."""
        owner_key = f"module:{module.__name__}"
        handlers: list[RegisteredEventHandler] = []

        for _, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__:
                continue

            subscriptions = iter_subscriptions(function)
            if not subscriptions:
                continue

            for subscription in subscriptions:
                if not subscription.enabled:
                    self.context.log_dev(
                        f"[Core] Disabled module subscription skipped for: {owner_name}"
                    )
                    continue
                handlers.append(
                    self._register_handler(
                        owner_name=owner_name,
                        event_name=subscription.event_name,
                        callback=function,
                    )
                )

        if not handlers:
            return False

        self._event_registrations[owner_key] = handlers
        self._module_registrations[module.__name__] = _ModuleRegistration(
            module_name=module.__name__,
            owner_key=owner_key,
        )
        self.context.log_dev(
            f"[Core] Registered {len(handlers)} event handler(s) from module: {owner_name}"
        )
        return True

    def _register_handler(
        self,
        *,
        owner_name: str,
        event_name: EventName,
        callback: Callable[..., None],
    ) -> RegisteredEventHandler:
        """Register one discovered callback against the dispatcher."""
        if not callable(callback):
            raise TypeError("event callback is not callable")

        typed_callback = cast(Callable[[EventPayload], None], callback)
        self.dispatcher.subscribe(
            event_name,
            typed_callback,
            owner_name=owner_name,
        )
        return RegisteredEventHandler(
            event_name=event_name,
            callback=typed_callback,
            owner_name=owner_name,
        )

    def _unregister_owner(self, owner_key: str) -> None:
        """Remove all dispatcher registrations owned by one extension or module."""
        registrations = self._event_registrations.pop(owner_key, ())
        for registration in registrations:
            self.dispatcher.unsubscribe(
                registration.event_name,
                registration.callback,
            )
