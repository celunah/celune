# SPDX-License-Identifier: MIT
"""Tests for backend resolution and extension infrastructure."""

import tempfile
import textwrap
import threading
import unittest
from pathlib import Path

from celune.backends import resolve_backend
from celune.extensions.base import CeluneContext, CeluneExtension
from celune.extensions.manager import CeluneExtensionManager
from celune.exceptions import ExtensionAlreadyRegisteredError, InvalidExtensionError

from tests.support import FakeBackend


class BackendTests(unittest.TestCase):
    """Tests for backend base behavior and backend resolution."""

    def test_base_backend_reports_models_and_progress(self) -> None:
        """Verify model metadata and progress helpers on a fake backend.

        Returns:
            None: Assertions verify backend helper output.

        Raises:
            AssertionError: A backend helper returns an unexpected value.
        """
        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        self.assertEqual(backend.default_model_id, "fake/balanced")
        self.assertEqual(backend.all_model_ids, ["fake/balanced", "fake/bold"])
        self.assertEqual(backend.voices, ["balanced", "bold"])
        self.assertEqual(backend.model_id_for_voice("bold"), "fake/bold")
        self.assertIsNone(backend.generation_progress_total("text"))
        self.assertEqual(backend.generation_progress_steps(None), 1)
        self.assertEqual(backend.generation_progress_steps({"chunk_steps": 3}), 3)
        self.assertEqual(backend.generation_progress_steps({"chunk_steps": 0}), 1)

    def test_resolve_backend_accepts_instance_type_and_rejects_unknown(self) -> None:
        """Verify supported backend specifications and invalid input failures.

        Returns:
            None: Assertions verify backend resolution behavior.

        Raises:
            AssertionError: Backend resolution behavior changes unexpectedly.
        """
        instance = FakeBackend(log=lambda _msg, _severity="info": None)
        self.assertIs(resolve_backend(instance), instance)
        self.assertIsInstance(resolve_backend(FakeBackend), FakeBackend)
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            resolve_backend("missing")
        with self.assertRaisesRegex(TypeError, "backend_name"):
            resolve_backend(123)  # type: ignore[arg-type]


class ExtensionTests(unittest.TestCase):
    """Tests for extension context and manager behavior."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.dev_logs: list[tuple[str, str]] = []
        self.invocations: list[tuple[str, tuple[str, ...]]] = []
        self.context = CeluneContext(
            log=lambda msg, severity="info": self.logs.append((msg, severity)),
            log_dev=lambda msg, severity="info": self.dev_logs.append((msg, severity)),
            say=lambda text, save=True, display_text=None: True,
            play=lambda sound_path, keep=False: True,
            status=lambda msg, severity="info": None,
            set_voice=lambda name: True,
            get_state=lambda: "idle",
            wait_until_ready=lambda timeout=30.0: True,
        )

    def test_context_and_extension_helpers_delegate_calls(self) -> None:
        """Verify extension helper methods delegate through their context.

        Returns:
            None: Assertions verify delegated extension behavior.

        Raises:
            AssertionError: Extension delegation behavior changes unexpectedly.
        """
        extension = DemoExtension(self.context)
        self.context.expose("token", "value")
        self.assertEqual(self.context.get("token"), "value")
        self.assertEqual(extension.state, "idle")
        extension.log("hello")
        self.assertEqual(self.logs[-1], ("[Demo] hello", "info"))
        self.assertEqual(extension.say("hello"), True)
        self.assertEqual(extension.play("tone.wav"), True)
        self.assertEqual(extension.set_voice("bold"), True)

    def test_manager_registers_invokes_and_autoloads_extensions(self) -> None:
        """Verify registration, duplicate handling, and directory autoloading.

        Returns:
            None: Assertions verify extension registration behavior.

        Raises:
            AssertionError: Extension manager behavior changes unexpectedly.
        """
        manager = CeluneExtensionManager(self.context)
        manager.register(DemoExtension)
        self.assertEqual(manager.list_extensions(), ["Demo"])
        with self.assertRaises(ExtensionAlreadyRegisteredError):
            manager.register(DemoExtension)
        with self.assertRaises(InvalidExtensionError):
            manager.register(object)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_file = Path(temp_dir) / "fixture.py"
            extension_file.write_text(
                textwrap.dedent(
                    """
                    from celune.extensions.base import CeluneExtension

                    class LoadedExtension(CeluneExtension):
                        EXTENSION_NAME = "Loaded"

                        def invoke(self, *args, **kwargs):
                            return None
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)
        self.assertIn("Loaded", manager.list_extensions())

    def test_manager_invoke_and_autostart_run_in_threads(self) -> None:
        """Verify threaded extension invocation and autostart behavior.

        Returns:
            None: Assertions verify asynchronous extension execution.

        Raises:
            AssertionError: Threaded extension behavior changes unexpectedly.
        """
        event = threading.Event()

        class AutoExtension(DemoExtension):
            """Autostart extension used by one manager test."""

            EXTENSION_NAME = "Auto"
            AUTOSTART = True

            def autostart(self) -> None:
                event.set()

        manager = CeluneExtensionManager(self.context)
        manager.register(AutoExtension)
        manager.autostart_all()
        self.assertEqual(event.wait(timeout=1), True)

        invoke_event = threading.Event()

        class InvokeExtension(DemoExtension):
            """Invokable extension used by one manager test."""

            EXTENSION_NAME = "Invoke"

            def invoke(self, *args, **kwargs) -> None:
                invoke_event.set()

        manager.register(InvokeExtension)
        manager.invoke("Invoke", "x")
        self.assertEqual(invoke_event.wait(timeout=1), True)
        with self.assertRaises(InvalidExtensionError):
            manager.invoke("Missing")


class DemoExtension(CeluneExtension):
    """Simple extension implementation used by manager tests."""

    EXTENSION_NAME = "Demo"

    def invoke(self, *args, **kwargs) -> None:
        return None
