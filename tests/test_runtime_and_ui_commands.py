# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime validation and lightweight UI commands."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import sys
import asyncio
import tempfile
import subprocess
from types import SimpleNamespace
from typing import Optional, cast
from pathlib import Path
from unittest import mock
from collections.abc import Callable, Awaitable, Coroutine

import numpy as np
import pytest
from textual import events
from textual.widget import Widget
from textual.widgets import TextArea

from celune import runtime
from celune.ui import app as ui_app
from celune.i18n import string
from celune.agent import AgentTaskState
from celune.celune import Celune
from celune.ui.app import (
    ButtonActions,
    CeluneUI,
    VoiceButton,
)
from tests.support import FakeBackend, FakeVCBackend, CeluneTestCase
from tests.platform import WINDOWS_ONLY
from celune.terminal import set_terminal_title
from celune.constants import APP_NAME
from celune.ui.commands import process_command, attachment_source
from celune.ui.headless import CeluneHeadlessUI
from celune.typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
)
from celune.typing.common import JSONSerializable
from celune.typing.aliases import LogLevel
from .ui_startup_interactions import TestUIStartup as _TestUIStartup


class TestRuntime(CeluneTestCase):
    """Tests for runtime environment checks."""

    def test_core_runtime_classes_are_final_at_runtime(self) -> None:
        """Verify final core classes reject runtime subclass creation."""
        final_classes = (Celune, CeluneUI, CeluneHeadlessUI)

        for final_class in final_classes:
            assert getattr(final_class, "__final__", False)
            with pytest.raises(TypeError, match="is final and cannot be subclassed"):
                type(f"{final_class.__name__}Subclass", (final_class,), {})

    def test_ui_error_formatter_remains_available_after_lazy_import(self) -> None:
        """Verify deferred UI imports retain the formatter used by error paths."""
        assert ui_app.format_error(RuntimeError("startup"), "info") == ""
        assert ui_app.format_error(RuntimeError("startup"), "verbose") == "startup"

    def test_ui_terminal_title_remains_available_before_runtime_import(self) -> None:
        """Verify early shutdown can update the terminal title before runtime loading."""
        terminal = mock.Mock()

        ui_app.set_terminal_title((APP_NAME, "Ready", "Idle"), terminal)

        terminal.write.assert_called_once_with(
            f"\x1b]0;{APP_NAME} ・ Ready ・ Idle\x07"
        )
        terminal.flush.assert_called_once_with()

    def test_voice_button_actions_keep_menu_when_cycle_is_unavailable(self) -> None:
        """Verify voice cycling and voice-menu availability are independent."""
        button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=True),
        )
        ui = cast(
            CeluneUI,
            SimpleNamespace(
                style_button=button,
                update_resources=mock.Mock(),
                _run_on_ui_thread=lambda callback: callback(),
            ),
        )

        CeluneUI.change_voice_lock_state(
            ui,
            locked=True,
            can_open_menu=True,
        )

        assert button.actions == ButtonActions(press=False, hold=True)
        assert not button.disabled

        CeluneUI.change_voice_lock_state(
            ui,
            locked=False,
            can_open_menu=False,
        )

        assert button.actions == ButtonActions(press=True, hold=False)
        assert not button.disabled

    def test_voice_button_long_press_message_waits_for_release(self) -> None:
        """Verify a held voice button posts its modal message after release."""
        button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=True),
        )
        button._long_pressed = True
        button._stop_hold_timer = mock.Mock()
        button.suppress_click = mock.Mock()
        button.post_message = mock.Mock()

        async def release_button() -> None:
            await button._on_mouse_up(cast(events.MouseUp, SimpleNamespace()))

        with mock.patch.object(Widget, "_on_mouse_up", new=mock.AsyncMock()):
            asyncio.run(release_button())

        button.suppress_click.assert_called_once_with()
        button.post_message.assert_called_once()
        assert isinstance(button.post_message.call_args.args[0], VoiceButton.Held)

    def test_voice_button_press_is_gated_without_disabling_the_button(self) -> None:
        """Verify an unavailable press action leaves the native button enabled."""
        button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=True),
        )
        button.post_message = mock.Mock()

        button.press()

        assert not button.disabled
        button.post_message.assert_not_called()

    def test_voice_selection_wakes_sleeping_celune_before_loading_voice(self) -> None:
        """Verify choosing a voice from sleep wakes Celune before switching voices."""
        bundle_path = Path("voice.cevoice")
        call_order: list[str] = []

        async def wake() -> bool:
            call_order.append("wake")
            fake_celune.sleeping = False
            return True

        def set_voice(_entry: str) -> bool:
            call_order.append("voice")
            return True

        fake_celune = SimpleNamespace(
            sleeping=True,
            wake_from_sleep_async=wake,
            set_voice_and_wait=set_voice,
            voices=("balanced", "bold"),
        )
        ui = SimpleNamespace(
            celune=fake_celune,
            _voice_menu_paths={"Celune": bundle_path},
            celune_styles=(),
            style_index=0,
            tts_voice_changed=mock.Mock(),
            change_voice_lock_state=mock.Mock(),
            safe_log=mock.Mock(),
        )
        apply_voice_selection = cast(
            Callable[[CeluneUI, dict[str, JSONSerializable]], Awaitable[None]],
            getattr(
                CeluneUI._apply_voice_selection,
                "__wrapped__",
                CeluneUI._apply_voice_selection,
            ),
        )

        with mock.patch("celune.cevoice.active_bundle_path", return_value=bundle_path):
            asyncio.run(
                apply_voice_selection(
                    cast(CeluneUI, ui),
                    {"pack": "Celune", "entry": "bold"},
                )
            )

        assert call_order == ["wake", "voice"]
        assert ui.celune_styles == ("balanced", "bold")
        assert ui.style_index == 1
        ui.tts_voice_changed.assert_called_once_with("bold")

    def test_wake_resets_playback_widgets_before_persona_reinitializes(self) -> None:
        """Verify waking clears transient playback UI before wake work completes."""
        wake_started = asyncio.Event()
        release_wake = asyncio.Event()

        async def wake() -> bool:
            fake_celune.sleeping = False
            wake_started.set()
            await release_wake.wait()
            return True

        fake_celune = SimpleNamespace(sleeping=True, wake_from_sleep_async=wake)
        ui = SimpleNamespace(
            celune=fake_celune,
            cur_state="active",
            _reset_playback_widgets=mock.Mock(),
            _run_on_ui_thread=lambda callback: callback(),
            _schedule_sleep_timer=mock.Mock(),
            safe_status=mock.Mock(),
        )
        wake_from_sleep = cast(
            Callable[[CeluneUI], Coroutine[None, None, None]],
            getattr(CeluneUI.wake_from_sleep, "__wrapped__", CeluneUI.wake_from_sleep),
        )

        async def exercise_wake() -> None:
            wake_task = asyncio.create_task(wake_from_sleep(cast(CeluneUI, ui)))
            await asyncio.wait_for(wake_started.wait(), timeout=1.0)
            cast(mock.Mock, ui._reset_playback_widgets).assert_called_once_with()
            release_wake.set()
            await asyncio.wait_for(wake_task, timeout=1.0)

        asyncio.run(exercise_wake())

    def test_entrypoint_runtime_loader_keeps_heavy_imports_deferred(self) -> None:
        """Verify the pre-UI entrypoint import path stays torch-free."""
        project_root = Path(__file__).resolve().parents[1]
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from celune import entrypoint; "
                    "entrypoint._load_runtime(); "
                    "assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules; "
                    "assert 'celune.celune' not in sys.modules; "
                    "assert 'celune.ui.app' not in sys.modules"
                ),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(check.returncode, 0, check.stderr)

    def test_loading_ui_import_keeps_engine_and_model_libraries_deferred(self) -> None:
        """Verify loading-screen imports do not initialize engine dependencies."""
        project_root = Path(__file__).resolve().parents[1]
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from celune.ui import CeluneUI; "
                    "from celune.ui import app as ui_app; "
                    "assert CeluneUI.__name__ == 'CeluneUI'; "
                    "assert ui_app.indent('status', spaces=2) == '  status'; "
                    "assert ui_app.supports_ansi() is False; "
                    "blocked = {'celune.celune', 'celune.persona', 'lingua', "
                    "'numpy', 'psutil', 'sounddevice', 'sympy', 'torch', 'transformers'}; "
                    "assert blocked.isdisjoint(sys.modules), "
                    "sorted(blocked.intersection(sys.modules))"
                ),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(check.returncode, 0, check.stderr)

    def test_terminal_title_formats_and_sanitizes_status(self) -> None:
        """Verify structured terminal titles remove control characters."""
        terminal = mock.Mock()

        set_terminal_title((APP_NAME, "Ready", "Idle\n\x1b[31m"), terminal)

        terminal.write.assert_called_once_with(
            f"\x1b]0;{APP_NAME} ・ Ready ・ Idle\x07"
        )
        terminal.flush.assert_called_once_with()

    def test_terminal_title_caps_long_dynamic_actions(self) -> None:
        """Verify long terminal actions cannot crowd out the state glossary."""
        terminal = mock.Mock()

        set_terminal_title((APP_NAME, "Error", "x" * 200), terminal)

        title_escape = terminal.write.call_args.args[0]
        title = title_escape.removeprefix("\x1b]0;").removesuffix("\x07")
        assert len(title.removeprefix(APP_NAME)) <= 20

    def test_terminal_title_keeps_state_and_action_within_tab_budget(self) -> None:
        """Verify the separators and compact glossary fit after the app name."""
        terminal = mock.Mock()

        set_terminal_title(
            (
                APP_NAME,
                string("osc.state_recording"),
                string("osc.action_transcribing_speech"),
            ),
            terminal,
        )

        title = terminal.write.call_args.args[0]
        title = title.removeprefix("\x1b]0;").removesuffix("\x07")
        assert title == f"{APP_NAME} ・ Record ・ Transcr."
        assert len(title.removeprefix(APP_NAME)) <= 20

    def test_terminal_title_omits_duplicate_state_and_action(self) -> None:
        """Verify duplicate state and action labels are rendered only once."""
        terminal = mock.Mock()

        set_terminal_title(
            (
                APP_NAME,
                string("osc.state_exiting"),
                string("osc.action_exiting"),
            ),
            terminal,
        )

        title = terminal.write.call_args.args[0]
        title = title.removeprefix("\x1b]0;").removesuffix("\x07")
        assert title == f"{APP_NAME} ・ Exit"

    def test_check_supported_backends_reports_cpu_cuda_and_rocm(self) -> None:
        """Verify backend labels across supported runtime branches.

        Raises:
            AssertionError: Runtime detection changes unexpectedly.
        """
        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=False),
            mock.patch(
                "celune.runtime.torch.backends.mps.is_available", return_value=False
            ),
        ):
            assert runtime.check_supported_backends() == ("CPU", False)

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
            mock.patch(
                "celune.runtime.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 4090",
            ),
        ):
            assert runtime.check_supported_backends() == ("CUDA", True)

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", "6.0"),
        ):
            assert runtime.check_supported_backends() == ("ROCm", False)

    def test_check_supported_backends_treats_missing_driver_as_unusable_cuda(
        self,
    ) -> None:
        """Verify missing-driver CUDA init failures do not crash backend detection."""
        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
            mock.patch(
                "celune.runtime.torch.cuda.get_device_name",
                side_effect=RuntimeError(
                    "Found no NVIDIA driver on your system. Please check that you "
                    "have an NVIDIA GPU and installed a driver from "
                    "http://www.nvidia.com/Download/index.aspx"
                ),
            ),
        ):
            assert runtime.check_supported_backends() == ("CUDA", False)

    def test_validate_runtime_rejects_unsupported_backends_without_cuda_work(
        self,
    ) -> None:
        """Verify unsupported backends fail before CUDA work begins.

        Raises:
            AssertionError: Runtime rejection behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: LogLevel = "info",
        ) -> None:
            _ = loglevel
            logs.append((msg, severity))

        with (
            mock.patch("celune.runtime.sys.version_info", (3, 12, 0)),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
        ):
            assert not runtime.validate_runtime(
                log,
                errors.append,
                states.append,
                False,
                lambda exc, dev: str(exc),
                "info",
                "qwen3",
            )
        assert errors == ["No supported backend found"]
        assert states == ["error"]

    def test_validate_runtime_allows_cpu_for_mini_backend(self) -> None:
        """Verify CPU-only environments remain usable with the mini backend."""
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: LogLevel = "info",
        ) -> None:
            _ = loglevel
            logs.append((msg, severity))

        with (
            mock.patch("celune.runtime.sys.version_info", (3, 12, 0)),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
        ):
            assert runtime.validate_runtime(
                log,
                errors.append,
                states.append,
                False,
                lambda exc, dev: str(exc),
                "info",
                "mini",
            )

        assert not errors
        assert not states


class TestUICommand(CeluneTestCase):
    """Tests for lightweight slash command behavior."""

    @staticmethod
    def _thread_runs_immediately(*args, **kwargs):
        """Return a thread-like object whose start runs the target immediately."""
        target = kwargs.get("target")
        if target is None and args:
            target = args[0]

        class _ImmediateThread:
            """An immediate thread harness object."""

            @staticmethod
            def start() -> None:
                """Start the thread."""
                if target is not None:
                    target()

        return _ImmediateThread()

    def setUp(self) -> None:
        """Create the lightweight command-test UI fixture."""
        self.logs: list[tuple[str, str]] = []
        self.ui = SimpleNamespace()
        self.ui.input_box = SimpleNamespace(load_text=lambda text: None)
        self.ui.safe_log = lambda msg, severity="info", **kwargs: self.logs.append(
            (msg, severity)
        )
        self.ui.safe_log_dev = self.ui.safe_log
        self.ui.refresh_vc_controls = mock.Mock()
        self.ui.celune = SimpleNamespace(
            config={"ipa": False},
            backend=FakeBackend,
            voice_prompt=None,
            persona_attachments=[],
            can_use_rubberband=True,
            speed=1.0,
            reverb=SimpleNamespace(strength=0.0),
            say=mock.Mock(return_value=True),
            play=mock.Mock(return_value=True),
            try_play_signal=mock.Mock(return_value=True),
            vision=SimpleNamespace(enabled=True, talkback=True),
            dev=False,
            input_mode="text_to_speech",
            vc_f0_condition=False,
            vc_pitch_shift=0,
            vc_backend=SimpleNamespace(),
        )

    def _process_command(self, command: str, args: list[str]) -> None:
        """Process one command against the typed UI test double."""
        process_command(cast(CeluneUI, self.ui), command, args)

    def test_restartaudio_command_restarts_host_audio_server(self) -> None:
        """Verify the audio recovery command reports a successful restart."""
        with (
            mock.patch("celune.ui.commands.restart_audio_server") as restart,
            mock.patch(
                "celune.ui.commands.threading.Thread",
                side_effect=self._thread_runs_immediately,
            ),
        ):
            self.ui.celune._close_stream = mock.Mock()
            self._process_command("restartaudio", [])

        restart.assert_called_once_with()
        self.ui.celune._close_stream.assert_called_once_with(abort=True)
        self.ui.celune.try_play_signal.assert_called_once_with("readiness")
        assert self.logs[-1] == (
            "Host audio server restarted.",
            "info",
        )

    def test_common_commands_update_state_and_validate_inputs(self) -> None:
        """Verify prompt, speed, and reverb command paths.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("voiceprompt", ["gentle", "tone"])
        assert self.ui.celune.voice_prompt == "gentle tone"
        self._process_command("voiceprompt", ["clear"])
        assert cast(Optional[str], self.ui.celune.voice_prompt) is None

        self._process_command("speed", ["120%"])
        assert self.ui.celune.speed == 1.2
        self._process_command("speed", ["200"])
        assert self.logs[-1][1] == "warning"

        self._process_command("reverb", ["50"])
        assert self.ui.celune.reverb.strength == 0.5
        self._process_command("reverb", ["150"])
        assert self.logs[-1][1] == "warning"

    def test_backend_and_cevoice_commands_request_hot_reloads(self) -> None:
        """Verify slash commands delegate backend and CEVOICE hot reloads into Celune."""
        self.ui.celune.set_backend_async = mock.AsyncMock(return_value=True)
        self.ui.celune.set_cevoice_async = mock.AsyncMock(return_value=True)

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("backend", ["mini"])
            self._process_command("cevoice", ["nova"])

        self.ui.celune.set_backend_async.assert_awaited_once_with("mini")
        self.ui.celune.set_cevoice_async.assert_awaited_once_with("nova")
        assert self.logs[-2] == ("Switched to backend: mini", "info")
        assert self.logs[-1] == ("Character changed: nova", "info")

    def test_backend_command_uses_active_vc_backend_name_for_duplicate_guard(
        self,
    ) -> None:
        """Verify the backend slash command checks the active VC runtime name too."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.vc_backend = FakeVCBackend(
            log=lambda _msg, _severity="info": None
        )
        self.ui.celune._active_runtime_backend_name = mock.Mock(return_value="fake-vc")
        self.ui.celune.set_backend_async = mock.AsyncMock(return_value=True)

        self._process_command("backend", ["fake-vc"])

        self.ui.celune._active_runtime_backend_name.assert_called_once_with()
        self.ui.celune.set_backend_async.assert_not_called()
        assert self.logs[-1][1] == "warning"

    def test_voice_conversion_commands_update_seedvc_state(self) -> None:
        """Verify VC slash commands update engine and backend state."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.vc_backend = SimpleNamespace(f0_condition=False, pitch_shift=0)

        self._process_command("vcmode", ["sing"])
        self._process_command("vcpitch", ["-2"])
        self._process_command("vcpitch", ["clear"])

        assert self.ui.celune.vc_f0_condition
        assert self.ui.celune.vc_backend.f0_condition
        assert self.ui.celune.vc_pitch_shift == 0
        assert self.ui.celune.vc_backend.pitch_shift == 0
        assert self.logs[-3] == ("Voice conversion mode set to Sing.", "info")
        assert self.logs[-2] == (
            "Voice conversion pitch shift set to -2 semitones.",
            "info",
        )
        assert self.logs[-1] == (
            "Voice conversion pitch shift set to 0 semitones.",
            "info",
        )
        assert self.ui.refresh_vc_controls.call_count == 3

    def test_vc_command_submits_audio_file_through_engine_pipeline(self) -> None:
        """Verify /vc decodes a file and submits it through Celune's VC pipeline."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.submit_audio = mock.Mock(return_value=True)
        audio = np.ones((8, 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            source_path.write_bytes(b"wav")

            with (
                mock.patch(
                    "celune.ui.commands.threading.Thread",
                    side_effect=self._thread_runs_immediately,
                ),
                mock.patch(
                    "celune.ui.commands.sf.read",
                    return_value=(audio, 48000),
                ),
            ):
                self._process_command("vc", [str(source_path)])

        self.ui.celune.submit_audio.assert_called_once_with(
            audio,
            48000,
            label="source.wav",
        )
        assert self.logs[-1] == (f"Running voice conversion on {source_path}", "info")

    def test_voice_conversion_commands_validate_mode_and_pitch(self) -> None:
        """Verify VC slash commands stay gated to VC mode and validate arguments."""
        self._process_command("vc", ["sample.wav"])
        assert self.logs[-1] == (
            "This command is only available in voice conversion mode.",
            "warning",
        )

        self._process_command("vcmode", ["sing"])
        assert self.logs[-1] == (
            "This command is only available in voice conversion mode.",
            "warning",
        )

        self.ui.celune.input_mode = "voice_conversion"
        self._process_command("vc", [])
        assert self.logs[-1] == ("Usage: /vc <file>", "warning")

        self._process_command("vc", ["missing.wav"])
        assert self.logs[-1] == (
            "Voice conversion input not found: missing.wav",
            "warning",
        )

        self._process_command("vcmode", ["maybe"])
        assert self.logs[-1] == ("Usage: /vcmode <talk|sing>", "warning")

        self._process_command("vcpitch", ["13"])
        assert self.logs[-1] == (
            "Pitch shift must be between -3 and 3 semitones.",
            "warning",
        )

        self._process_command("vcpitch", ["high"])
        assert self.logs[-1] == ("Usage: /vcpitch <semitones|clear>", "warning")

    def test_vc_command_reports_decode_failure(self) -> None:
        """Verify /vc surfaces decode errors cleanly."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.log_level = "verbose"

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "broken.wav"
            source_path.write_bytes(b"broken")

            with (
                mock.patch(
                    "celune.ui.commands.threading.Thread",
                    side_effect=self._thread_runs_immediately,
                ),
                mock.patch(
                    "celune.ui.commands.sf.read",
                    side_effect=RuntimeError("decode failed"),
                ),
            ):
                self._process_command("vc", [str(source_path)])

        assert self.logs[-1] == (
            "Cannot read this voice conversion input: decode failed",
            "error",
        )

    def test_cevoice_command_reports_failed_character_switch(self) -> None:
        """Verify /cevoice warns when the requested pack cannot be loaded."""
        self.ui.celune.set_cevoice_async = mock.AsyncMock(return_value=False)

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("cevoice", ["invalid_character"])

        self.ui.celune.set_cevoice_async.assert_awaited_once_with("invalid_character")
        assert self.logs[-1] == (
            "Could not switch character to invalid_character.",
            "warning",
        )

    def test_cevoice_command_rejects_already_loaded_character(self) -> None:
        """Verify /cevoice warns instead of reloading the active pack."""
        self.ui.celune.set_cevoice_async = mock.AsyncMock(return_value=True)

        with (
            mock.patch(
                "celune.ui.commands.resolve_bundle_path",
                return_value=Path("voices/nova.cevoice"),
            ),
            mock.patch(
                "celune.ui.commands.active_bundle_path",
                return_value=Path("voices/nova.cevoice"),
            ),
            mock.patch(
                "celune.ui.commands.threading.Thread",
                side_effect=self._thread_runs_immediately,
            ),
        ):
            self._process_command("cevoice", ["nova"])

        self.ui.celune.set_cevoice_async.assert_not_called()
        assert self.logs[-1] == ("This character is already loaded.", "warning")

    def test_voiceprompt_command_is_blocked_when_model_lacks_instruction_control(
        self,
    ) -> None:
        """Verify voice prompts are refused on the Qwen3 0.6B preset."""
        self.ui.celune.voice_prompt_supported = lambda: False
        self.ui.celune.voice_prompt = "old"

        self._process_command("voiceprompt", ["gentle", "tone"])

        assert cast(Optional[str], self.ui.celune.voice_prompt) is None
        assert self.logs[-1] == (
            "Voice prompts are unavailable with the currently loaded model.",
            "warning",
        )

    def test_attach_command_stages_visual_media_for_persona_reply(self) -> None:
        """Verify /attach validates media and stores Qwen-compatible file URIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "ready.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            self._process_command("attach", [str(image)])

            assert len(self.ui.celune.persona_attachments) == 1
            attachment = self.ui.celune.persona_attachments[0]
            assert attachment["type"] == "image"
            assert attachment["path"] == attachment_source(image.resolve())
            assert self.logs[-1][1] == "info"

        self._process_command("attach", ["clear"])
        assert self.ui.celune.persona_attachments == []

    def test_attach_command_accepts_remote_image_urls(self) -> None:
        """Verify /attach accepts HTTP image URLs without local file checks."""
        url = "https://example.com/images/reference.png"

        self._process_command("attach", [url])

        assert len(self.ui.celune.persona_attachments) == 1
        attachment = self.ui.celune.persona_attachments[0]
        assert attachment["type"] == "image"
        assert attachment["path"] == url
        assert attachment["name"] == "reference.png"
        assert self.logs[-1][1] == "info"

    def test_say_command_bypasses_persona_and_queues_direct_speech(self) -> None:
        """Verify /say sends literal speech through Celune.say()."""
        self._process_command("say", ["Hello", "there"])

        self.ui.celune.say.assert_called_once_with(
            "Hello there",
            display_text="Hello there",
        )

    def test_say_command_warns_when_text_is_missing(self) -> None:
        """Verify /say validates that direct speech text is present."""
        self._process_command("say", [])

        self.ui.celune.say.assert_not_called()
        assert self.logs[-1] == ("Usage: /say <text>", "warning")

    def test_play_command_passes_optional_volume(self) -> None:
        """Verify /play forwards the optional volume argument to Celune."""
        self.ui.celune.play.return_value = True

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("play", ["tone.wav", "0.4"])

        self.ui.celune.play.assert_called_once_with("tone.wav", volume=0.4)
        assert self.logs[-1] == ("Playing tone.wav at 40% volume", "info")

    def test_play_command_reports_youtube_before_play_returns(self) -> None:
        """Verify the YouTube status can be emitted during the playback setup."""

        def play(_path: str, *, volume: float, on_started) -> bool:
            assert volume == 0.4
            on_started()
            return True

        self.ui.celune.play.side_effect = play

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("play", ["https://youtu.be/demo", "0.4"])

        self.ui.celune.play.assert_called_once()
        assert self.logs[-1] == ("Playing YouTube audio at 40% volume", "info")

    def test_play_command_rejects_invalid_volume(self) -> None:
        """Verify /play validates a numeric optional volume argument."""
        self._process_command("play", ["tone.wav", "loud"])

        self.ui.celune.play.assert_not_called()
        assert self.logs[-1] == (
            "Invalid volume for 'play', must be numeric.",
            "warning",
        )

    def test_say_command_reports_unmatched_ipa_characters(self) -> None:
        """Verify /say keeps the usual unmatched-IPA warning path."""
        with mock.patch(
            "celune.ui.commands.replace_ipa",
            return_value=("hello", 2),
        ):
            self._process_command("say", ["həˈloʊ"])

        self.ui.celune.say.assert_called_once_with("hello", display_text="həˈloʊ")
        assert self.logs[-1] == (
            "Found 2 unmatched IPA characters, output may be inaccurate.",
            "warning",
        )

    @WINDOWS_ONLY
    def test_windows_command_split_keeps_literal_backslashes(self) -> None:
        """Verify Windows slash commands keep single-backslash file paths intact."""
        parts = CeluneUI.split_command_input(
            r'attach "C:\Users\user\Downloads\bad suggestion.png"'
        )

        assert parts == ["attach", r"C:\Users\user\Downloads\bad suggestion.png"]


class TestUIStartup(_TestUIStartup):
    """Collect interactive startup tests."""


class TestAgentStatusUI(CeluneTestCase):
    """Tests for the typed agent lifecycle projection in the UI."""

    def tearDown(self) -> None:
        """Reset singleton UI guards after each test."""
        CeluneUI._instance = None

    @staticmethod
    def _ui_for_task(
        state: AgentTaskState,
        *,
        needs_compaction: bool = False,
        busy: Optional[ComponentBusyResult] = None,
    ) -> tuple[CeluneUI, SimpleNamespace, mock.Mock]:
        """Build a headless UI around one typed task-shaped test fixture."""
        task = SimpleNamespace(
            task_id="task-1",
            state=state,
            iterations=2,
            needs_context_compaction=needs_compaction,
            config=SimpleNamespace(max_loops=5),
        )
        agent_runtime = SimpleNamespace(
            get_active_task=mock.Mock(return_value=task),
            get_task=mock.Mock(return_value=task),
        )
        ui = CeluneUI()
        safe_status = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                agent_runtime=agent_runtime,
                cur_state="thinking",
                last_component_busy=busy,
                test_finished=False,
            ),
        )
        ui.celune_ready = True
        ui.safe_status = safe_status
        return ui, task, safe_status

    def test_agent_status_renders_lifecycle_states_once(self) -> None:
        """Verify lifecycle events become localized status text without redraw spam."""
        ui, task, safe_status = self._ui_for_task(AgentTaskState.IDLE)
        for state, expected in (
            (AgentTaskState.IDLE, "agent.status.idle"),
            (AgentTaskState.CLASSIFYING, "agent.status.classifying"),
            (AgentTaskState.WORKING, "agent.status.working"),
            (AgentTaskState.AWAITING_APPROVAL, "agent.status.awaiting_approval"),
            (AgentTaskState.AWAITING_CHOICE, "agent.status.awaiting_choice"),
            (AgentTaskState.PAUSED, "agent.status.paused"),
            (AgentTaskState.INTERRUPTED, "agent.status.interrupted"),
            (AgentTaskState.CANCELLING, "agent.status.cancelling"),
            (AgentTaskState.COMPLETED, "agent.status.completed"),
            (AgentTaskState.FAILED, "agent.status.failed"),
            (AgentTaskState.CANCELLED, "agent.status.cancelled"),
            (AgentTaskState.ABORTED, "agent.status.aborted"),
        ):
            with self.subTest(state=state):
                task.state = state
                ui._agent_status_signature = None
                safe_status.reset_mock()

                ui._refresh_agent_status()
                ui._refresh_agent_status()

                if state == AgentTaskState.WORKING:
                    safe_status.assert_called_once_with(
                        string(
                            "agent.status.working",
                            iteration=2,
                            maximum=5,
                        ),
                        "info",
                    )
                else:
                    safe_status.assert_called_once_with(string(expected), "info")

    def test_agent_status_prioritizes_compaction_and_structured_busy_components(
        self,
    ) -> None:
        """Verify compaction and typed lock contention are visible without string parsing."""
        ui, task, safe_status = self._ui_for_task(
            AgentTaskState.PLANNING,
            needs_compaction=True,
        )

        ui._refresh_agent_status()

        safe_status.assert_called_once_with(string("agent.status.compacting"), "info")
        task.needs_context_compaction = False
        ui.celune.last_component_busy = ComponentBusyResult(
            components=(ComponentLockName.VLM, ComponentLockName.TTS),
            owners=(
                (
                    ComponentLockName.VLM,
                    ComponentLockOwner(operation_id="persona"),
                ),
                (
                    ComponentLockName.TTS,
                    ComponentLockOwner(operation_id="speech"),
                ),
            ),
        )

        ui._refresh_agent_status()

        safe_status.assert_called_with(
            string(
                "agent.status.busy_components",
                components="VLM, TTS",
            ),
            "warning",
        )

    def test_stopped_state_wins_over_agent_status_and_uses_stopped_placeholder(
        self,
    ) -> None:
        """Verify STOPPED remains visible and rejects every key except Ctrl+Q."""
        ui, _task, _safe_status = self._ui_for_task(AgentTaskState.WORKING)
        ui.celune.cur_state = "stopped"
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=False),
        )
        ui.refresh_vc_controls = mock.Mock()
        ui.update_resources = mock.Mock()
        ui.change_input_state(locked=True)

        self.assertEqual(ui.input_box.placeholder, string("ui.stopped_placeholder"))
        self.assertEqual(
            ui._terminal_status_for(string("status.speaking"), "info"),
            ("stopped", string("osc.action_stopped")),
        )
        ui.celune.cur_state = "thinking"
        ui._agent_task_state = AgentTaskState.AWAITING_APPROVAL
        self.assertEqual(
            ui._terminal_status_for(string("agent.status.awaiting_approval"), "info"),
            ("awaiting", string("agent.status.awaiting_approval")),
        )
        ui._agent_task_state = AgentTaskState.PAUSED
        self.assertEqual(
            ui._terminal_status_for(string("agent.status.paused"), "info"),
            ("paused", string("agent.status.paused")),
        )
        ui.celune.cur_state = "stopped"

        event = SimpleNamespace(
            key="ctrl+r",
            prevent_default=mock.Mock(),
            stop=mock.Mock(),
        )
        ui.on_key(cast(events.Key, event))

        event.prevent_default.assert_called_once_with()
        event.stop.assert_called_once_with()
