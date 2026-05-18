# SPDX-License-Identifier: MIT
"""Tests for backend resolution and extension infrastructure."""

import sys
import tempfile
import textwrap
import threading
import importlib
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
from unittest import mock, TestCase
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from celune.utils import discard
from celune.backends import resolve_backend
from celune.extensions.manager import CeluneExtensionManager
from celune.extensions.base import CeluneContext, CeluneExtension
from celune.exceptions import ExtensionAlreadyRegisteredError, InvalidExtensionError

from tests.support import FakeBackend


class BackendTests(TestCase):
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

    def test_voxcpm2_uses_pack_cfg_scale_when_present(self) -> None:
        """Verify CEVOICE can override VoxCPM2's per-voice CFG scale."""
        with mock.patch.dict(sys.modules, {"voxcpm": SimpleNamespace(VoxCPM=object)}):
            voxcpm2 = importlib.import_module("celune.backends.voxcpm2")
            voxcpm2_cls = voxcpm2.VoxCPM2

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.cfg_value = None

                def generate_streaming(
                    self, *args, **kwargs
                ) -> Iterator[npt.NDArray[np.float32]]:
                    """Generate a fake stream of VoxCPM2 audio chunks.

                    Args:
                        args: Not used.
                        kwargs: Only used for a fake CFG scale value.

                    Returns:
                        npt.NDArray[np.float32]: Fake VoxCPM2 audio chunks.
                    """
                    discard(args)
                    self.cfg_value = kwargs["cfg_value"]
                    yield np.zeros((1,), dtype=np.float32)

            loader = SimpleNamespace(
                bundle=SimpleNamespace(voices={"calm": {"cfg_scale": 4.2}}),
                materialize=lambda voice, kind: Path(f"{voice}.{kind}"),
            )
            with (
                mock.patch.object(voxcpm2_cls, "_validate_refs"),
                mock.patch(
                    "celune.backends.voxcpm2.default_loader", return_value=loader
                ),
            ):
                backend = voxcpm2_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(
                    backend.generate_stream(
                        model,
                        text="hello",
                        voice="calm",
                        chunk_size=1,
                    )
                )

            self.assertEqual(model.cfg_value, 4.2)

    def test_qwen3_uses_pack_reference_text_when_present(self) -> None:
        """Verify CEVOICE can override Qwen3's per-voice reference text."""
        with mock.patch.dict(
            sys.modules,
            {
                "faster_qwen3_tts": SimpleNamespace(
                    FasterQwen3TTS=object,
                    __version__="0.2.5",
                )
            },
        ):
            qwen3 = importlib.import_module("celune.backends.qwen3")
            qwen3_cls = qwen3.Qwen3

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.ref_text = None

                def generate_voice_clone_streaming(
                    self, *args, **kwargs
                ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
                    """Generate a fake stream of Qwen3 audio chunks.

                    Args:
                        args: Not used.
                        kwargs: Not used.

                    Returns:
                        npt.NDArray[np.float32]: Fake Qwen3 audio chunks.
                    """
                    discard(args)
                    self.ref_text = kwargs["ref_text"]
                    yield np.zeros((1,), dtype=np.float32), 24000, None

            loader = SimpleNamespace(
                bundle=SimpleNamespace(
                    voices={"calm": {"reference_text": "Pack reference."}}
                ),
                materialize=lambda voice, kind: Path(f"{voice}.{kind}"),
            )
            with (
                mock.patch.object(qwen3_cls, "_validate_refs"),
                mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.ref_text, "Pack reference.")


class ExtensionTests(TestCase):
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
        self.assertTrue(event.wait(timeout=1))

        invoke_event = threading.Event()

        class InvokeExtension(DemoExtension):
            """Invokable extension used by one manager test."""

            EXTENSION_NAME = "Invoke"

            def invoke(self, *args, **kwargs) -> None:
                invoke_event.set()

        manager.register(InvokeExtension)
        manager.invoke("Invoke", "x")
        self.assertTrue(invoke_event.wait(timeout=5))
        with self.assertRaises(InvalidExtensionError):
            manager.invoke("Missing")


class DemoExtension(CeluneExtension):
    """Simple extension implementation used by manager tests."""

    EXTENSION_NAME = "Demo"

    def invoke(self, *args, **kwargs) -> None:
        return None
