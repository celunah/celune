# SPDX-License-Identifier: MIT
"""Tests for runtime validation and lightweight UI commands."""

import logging
import warnings
from pathlib import Path
from typing import cast
from types import SimpleNamespace
from unittest import mock, TestCase

from textual.widgets import Button, Label, TextArea

from celune import runtime
from celune.celune import Celune
from celune.config import Config
from celune.constants import JSONSerializable
from celune.backends.qwen3 import Qwen3
from celune.ui.commands import _attachment_source, process_command
from celune.ui.app import CeluneUI, _split_command_input
from celune.ui.headless import CeluneHeadlessUI


class RuntimeTests(TestCase):
    """Tests for runtime environment checks."""

    def test_check_supported_backends_reports_cpu_cuda_and_rocm(self) -> None:
        """Verify backend labels across supported runtime branches.

        Returns:
            None: Assertions verify runtime detection behavior.

        Raises:
            AssertionError: Runtime detection changes unexpectedly.
        """
        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=False),
            mock.patch(
                "celune.runtime.torch.backends.mps.is_available", return_value=False
            ),
        ):
            self.assertEqual(runtime.check_supported_backends(), ("CPU", False))

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
        ):
            self.assertEqual(runtime.check_supported_backends(), ("CUDA", True))

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", "6.0"),
        ):
            self.assertEqual(runtime.check_supported_backends(), ("ROCm", False))

    def test_validate_runtime_rejects_unsupported_backends_without_cuda_work(
        self,
    ) -> None:
        """Verify unsupported backends fail before CUDA work begins.

        Returns:
            None: Assertions verify runtime rejection behavior.

        Raises:
            AssertionError: Runtime rejection behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(msg: str, severity: str) -> None:
            logs.append((msg, severity))

        with (
            mock.patch("celune.runtime.sys.version_info", (3, 12, 0)),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
        ):
            self.assertEqual(
                runtime.validate_runtime(
                    log,
                    errors.append,
                    states.append,
                    False,
                    lambda exc, dev: str(exc),
                    False,
                ),
                False,
            )
        self.assertEqual(errors, ["No supported backend found"])
        self.assertEqual(states, ["error"])


class UICommandTests(TestCase):
    """Tests for lightweight slash command behavior."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.ui = SimpleNamespace()
        self.ui.input_box = SimpleNamespace(load_text=lambda text: None)
        self.ui.safe_log = lambda msg, severity="info": self.logs.append(
            (msg, severity)
        )
        self.ui.celune = SimpleNamespace(
            backend=SimpleNamespace(),
            voice_prompt=None,
            persona_attachments=[],
            can_use_rubberband=True,
            speed=1.0,
            reverb=SimpleNamespace(strength=0.0),
        )

    def _process_command(self, command: str, args: list[str]) -> None:
        """Process one command against the typed UI test double.

        Args:
            command: Command name without a leading slash.
            args: Parsed command arguments.

        Returns:
            None: This helper forwards to the production command handler.
        """
        process_command(cast(CeluneUI, self.ui), command, args)

    def test_xvectoronly_command_requires_qwen3_and_valid_value(self) -> None:
        """Verify the Qwen3-only toggle command and argument checks.

        Returns:
            None: Assertions verify command behavior.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("xvectoronly", [])
        self.assertEqual(self.logs[-1][1], "warning")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            backend = Qwen3(log=lambda msg, severity="info": None, mode="native")
        backend.x_vector_only = False
        self.ui.celune.backend = backend
        self._process_command("xvectoronly", [])
        self.assertEqual(self.logs[-1], ("Usage: /xvectoronly <true/false>", "warning"))
        self._process_command("xvectoronly", ["maybe"])
        self.assertEqual(self.logs[-1][1], "warning")
        self._process_command("xvectoronly", ["true"])
        self.assertEqual(backend.x_vector_only, True)
        self.assertEqual(
            self.logs[-1], ("Qwen3 identity-only cloning enabled.", "info")
        )

    def test_common_commands_update_state_and_validate_inputs(self) -> None:
        """Verify prompt, speed, and reverb command paths.

        Returns:
            None: Assertions verify command behavior.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("voiceprompt", ["gentle", "tone"])
        self.assertEqual(self.ui.celune.voice_prompt, "gentle tone")
        self._process_command("voiceprompt", ["clear"])
        self.assertIsNone(self.ui.celune.voice_prompt)

        self._process_command("speed", ["120%"])
        self.assertEqual(self.ui.celune.speed, 1.2)
        self._process_command("speed", ["200"])
        self.assertEqual(self.logs[-1][1], "warning")

        self._process_command("reverb", ["50"])
        self.assertEqual(self.ui.celune.reverb.strength, 0.5)
        self._process_command("reverb", ["150"])
        self.assertEqual(self.logs[-1][1], "warning")

    def test_voiceprompt_command_is_blocked_when_model_lacks_instruction_control(
        self,
    ) -> None:
        """Verify voice prompts are refused on the Qwen3 0.6B preset."""
        self.ui.celune.voice_prompt_supported = lambda: False
        self.ui.celune.voice_prompt = "old"

        self._process_command("voiceprompt", ["gentle", "tone"])

        self.assertIsNone(self.ui.celune.voice_prompt)
        self.assertEqual(
            self.logs[-1],
            (
                "Voice prompts are unavailable on the active Qwen3 0.6B model.",
                "warning",
            ),
        )

    def test_attach_command_stages_visual_media_for_persona_reply(self) -> None:
        """Verify /attach validates media and stores Qwen-compatible file URIs."""
        image = Path("demos/ready.png")

        self._process_command("attach", [str(image)])

        self.assertEqual(len(self.ui.celune.persona_attachments), 1)
        attachment = self.ui.celune.persona_attachments[0]
        self.assertEqual(attachment["type"], "image")
        self.assertEqual(attachment["path"], _attachment_source(image.resolve()))
        self.assertEqual(self.logs[-1][1], "info")

        self._process_command("attach", ["clear"])
        self.assertEqual(self.ui.celune.persona_attachments, [])

    def test_attach_command_accepts_remote_image_urls(self) -> None:
        """Verify /attach accepts HTTP image URLs without local file checks."""
        url = "https://example.com/images/reference.png"

        self._process_command("attach", [url])

        self.assertEqual(len(self.ui.celune.persona_attachments), 1)
        attachment = self.ui.celune.persona_attachments[0]
        self.assertEqual(attachment["type"], "image")
        self.assertEqual(attachment["path"], url)
        self.assertEqual(attachment["name"], "reference.png")
        self.assertEqual(self.logs[-1][1], "info")

    def test_windows_command_split_keeps_literal_backslashes(self) -> None:
        """Verify Windows slash commands keep single-backslash file paths intact."""
        with mock.patch("celune.ui.app.os.name", "nt"):
            parts = _split_command_input(
                r'attach "C:\Users\user\Downloads\bad suggestion.png"'
            )

        self.assertEqual(
            parts,
            ["attach", r"C:\Users\user\Downloads\bad suggestion.png"],
        )


class UIStartupTests(TestCase):
    """Tests for UI startup guard rails."""

    def tearDown(self) -> None:
        """Reset singleton UI guards after each test."""
        CeluneUI._instance = None
        CeluneHeadlessUI._instance = None

    def test_textual_ui_requires_attached_celune_on_mount(self) -> None:
        """Verify the Textual UI fails clearly without an attached Celune."""
        ui = CeluneUI()
        with self.assertRaisesRegex(
            RuntimeError,
            "CeluneUI requires an instance of Celune to be set",
        ):
            ui.on_mount()

    def test_headless_ui_warns_without_attached_celune(self) -> None:
        """Verify headless mode warns before doing nothing without Celune."""
        ui = CeluneHeadlessUI({"headless_nocolor": True})
        with (
            warnings.catch_warnings(record=True) as caught,
            mock.patch("celune.ui.headless.signal.signal"),
            mock.patch("celune.ui.headless.time.sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            warnings.simplefilter("always")
            ui.run()

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, RuntimeWarning))
        self.assertIn(
            "CeluneHeadlessUI has no attached Celune instance",
            str(caught[0].message),
        )

    def test_textual_input_lock_does_not_probe_persona_on_ui_thread(self) -> None:
        """Verify input state updates do not synchronously ping Persona."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = Button("Voice")
        ui.resources = cast(Label, None)
        persona_config: Config = {"talkback": True}
        ui.celune = cast(
            Celune,
            SimpleNamespace(config={"persona": cast(JSONSerializable, persona_config)}),
        )

        with mock.patch.object(ui, "_persona_loaded") as available:
            ui.change_input_state(locked=True)

        self.assertEqual(ui.input_box.placeholder, "Please wait")
        self.assertEqual(ui.style_button.disabled, True)
        available.assert_not_called()

        with (
            mock.patch.object(ui, "_persona_loaded") as available,
            mock.patch("celune.ui.app.threading.Thread") as thread_cls,
        ):
            ui.change_input_state(locked=False)

        self.assertEqual(ui.input_box.placeholder, "Enter text to speak here")
        self.assertEqual(ui.style_button.disabled, False)
        available.assert_not_called()
        thread_cls.return_value.start.assert_called_once()

    def test_placeholder_uses_loaded_persona_not_runtime_capability(self) -> None:
        """Verify the input placeholder reflects whether Persona actually loaded."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = Button("Voice")
        ui.resources = cast(Label, None)
        persona_config: Config = {"enabled": True, "talkback": True}
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={
                    "vram": "high",
                    "persona": cast(JSONSerializable, persona_config),
                },
                vision=None,
            ),
        )

        ui._persona_available = ui._persona_loaded()
        self.assertEqual(ui._normal_input_placeholder(), "Enter text to speak here")

        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={
                    "vram": "high",
                    "persona": cast(JSONSerializable, persona_config),
                },
                vision=object(),
            ),
        )
        ui._persona_available = ui._persona_loaded()
        self.assertEqual(ui._normal_input_placeholder(), "Say something...")

    def test_runtime_logger_warning_is_routed_into_ui_logs(self) -> None:
        """Verify known Python logger warnings do not bleed into the terminal."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("torch.utils.flop_counter")
        original_handlers = list(logger.handlers)
        original_propagate = logger.propagate
        self.addCleanup(setattr, logger, "handlers", original_handlers)
        self.addCleanup(setattr, logger, "propagate", original_propagate)

        ui._install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning(
            "triton not found; flop counting will not work for triton kernels"
        )

        self.assertEqual(
            captured,
            [
                (
                    "Internal runtime warning: triton not found; flop counting "
                    "will not work for triton kernels",
                    "warning",
                )
            ],
        )
