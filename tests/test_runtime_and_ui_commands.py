# SPDX-License-Identifier: MIT
"""Tests for runtime validation and lightweight UI commands."""

import sys
import time
import queue
import logging
import tempfile
import warnings
from typing import Optional, cast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock, TestCase

import numpy as np
import sounddevice as sd
from textual import events
from textual.widgets import Button, Label, RichLog, TextArea

from celune import colors
from celune import runtime
from celune.config import Config
from celune.celune import Celune
from celune.backends.tts.qwen3 import Qwen3
from celune.constants import APP_NAME, JSONSerializable
from celune.i18n import string
from celune.ui import app as ui_app
from celune.ui.app import CeluneUI
from celune.ui.headless import CeluneHeadlessUI
from celune.ui import resources as ui_resources
from celune.ui import terminal as ui_terminal
from celune.ui.commands import attachment_source, process_command
from celune.ui.theme import severity_color

from tests.support import FakeBackend, FakeVCBackend


class RuntimeTests(TestCase):
    """Tests for runtime environment checks."""

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
            self.assertEqual(runtime.check_supported_backends(), ("CPU", False))

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
            mock.patch(
                "celune.runtime.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 4090",
            ),
        ):
            self.assertEqual(runtime.check_supported_backends(), ("CUDA", True))

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", "6.0"),
        ):
            self.assertEqual(runtime.check_supported_backends(), ("ROCm", False))

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
            self.assertEqual(runtime.check_supported_backends(), ("CUDA", False))

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
                    "qwen3",
                ),
                False,
            )
        self.assertEqual(errors, ["No supported backend found"])
        self.assertEqual(states, ["error"])

    def test_validate_runtime_allows_cpu_for_mini_backend(self) -> None:
        """Verify CPU-only environments remain usable with the mini backend."""
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
                    "mini",
                ),
                True,
            )

        self.assertEqual(errors, [])
        self.assertEqual(states, [])


class UICommandTests(TestCase):
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
        self.logs: list[tuple[str, str]] = []
        self.ui = SimpleNamespace()
        self.ui.input_box = SimpleNamespace(load_text=lambda text: None)
        self.ui.safe_log = lambda msg, severity="info": self.logs.append(
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

    def test_xvectoronly_command_requires_qwen3_and_valid_value(self) -> None:
        """Verify the Qwen3-only toggle command and argument checks.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("xvectoronly", [])
        self.assertEqual(self.logs[-1][1], "warning")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            backend = Qwen3(log=lambda msg, severity="info": None)
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

    def test_backend_and_cevoice_commands_request_hot_reloads(self) -> None:
        """Verify slash commands delegate backend and CEVOICE hot reloads into Celune."""
        self.ui.celune.set_backend_and_wait = mock.Mock(return_value=True)
        self.ui.celune.set_cevoice_and_wait = mock.Mock(return_value=True)

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("backend", ["mini"])
            self._process_command("cevoice", ["nova"])

        self.ui.celune.set_backend_and_wait.assert_called_once_with("mini")
        self.ui.celune.set_cevoice_and_wait.assert_called_once_with("nova")
        self.assertEqual(self.logs[-2], ("Switched to backend: mini", "info"))
        self.assertEqual(self.logs[-1], ("Character changed: nova", "info"))

    def test_backend_command_uses_active_vc_backend_name_for_duplicate_guard(
        self,
    ) -> None:
        """Verify the backend slash command checks the active VC runtime name too."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.vc_backend = FakeVCBackend(
            log=lambda _msg, _severity="info": None
        )
        self.ui.celune._active_runtime_backend_name = mock.Mock(return_value="fake-vc")
        self.ui.celune.set_backend_and_wait = mock.Mock(return_value=True)

        self._process_command("backend", ["fake-vc"])

        self.ui.celune._active_runtime_backend_name.assert_called_once_with()
        self.ui.celune.set_backend_and_wait.assert_not_called()
        self.assertEqual(self.logs[-1][1], "warning")

    def test_voice_conversion_commands_update_seedvc_state(self) -> None:
        """Verify VC slash commands update engine and backend state."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.vc_backend = SimpleNamespace(f0_condition=False, pitch_shift=0)

        self._process_command("vcmode", ["sing"])
        self._process_command("vcpitch", ["-5"])
        self._process_command("vcpitch", ["clear"])

        self.assertEqual(self.ui.celune.vc_f0_condition, True)
        self.assertEqual(getattr(self.ui.celune.vc_backend, "f0_condition"), True)
        self.assertEqual(self.ui.celune.vc_pitch_shift, 0)
        self.assertEqual(getattr(self.ui.celune.vc_backend, "pitch_shift"), 0)
        self.assertEqual(self.logs[-3], ("Voice conversion mode set to Sing.", "info"))
        self.assertEqual(
            self.logs[-2],
            ("Voice conversion pitch shift set to -5 semitones.", "info"),
        )
        self.assertEqual(
            self.logs[-1],
            ("Voice conversion pitch shift set to 0 semitones.", "info"),
        )
        self.assertEqual(self.ui.refresh_vc_controls.call_count, 3)

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

        cast(mock.Mock, self.ui.celune.submit_audio).assert_called_once_with(
            audio,
            48000,
            label="source.wav",
        )
        self.assertEqual(
            self.logs[-1],
            (f"Running voice conversion on {source_path}", "info"),
        )

    def test_voice_conversion_commands_validate_mode_and_pitch(self) -> None:
        """Verify VC slash commands stay gated to VC mode and validate arguments."""
        self._process_command("vc", ["sample.wav"])
        self.assertEqual(
            self.logs[-1],
            ("This command is only available in voice conversion mode.", "warning"),
        )

        self._process_command("vcmode", ["sing"])
        self.assertEqual(
            self.logs[-1],
            ("This command is only available in voice conversion mode.", "warning"),
        )

        self.ui.celune.input_mode = "voice_conversion"
        self._process_command("vc", [])
        self.assertEqual(self.logs[-1], ("Usage: /vc <file>", "warning"))

        self._process_command("vc", ["missing.wav"])
        self.assertEqual(
            self.logs[-1],
            ("Voice conversion input not found: missing.wav", "warning"),
        )

        self._process_command("vcmode", ["maybe"])
        self.assertEqual(self.logs[-1], ("Usage: /vcmode <talk|sing>", "warning"))

        self._process_command("vcpitch", ["13"])
        self.assertEqual(
            self.logs[-1],
            ("Pitch shift must be between -12 and 12 semitones.", "warning"),
        )

        self._process_command("vcpitch", ["high"])
        self.assertEqual(
            self.logs[-1],
            ("Usage: /vcpitch <semitones|clear>", "warning"),
        )

    def test_vc_command_reports_decode_failure(self) -> None:
        """Verify /vc surfaces decode errors cleanly."""
        self.ui.celune.input_mode = "voice_conversion"

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

        self.assertEqual(
            self.logs[-1],
            ("Cannot read this voice conversion input: decode failed", "error"),
        )

    def test_cevoice_command_reports_failed_character_switch(self) -> None:
        """Verify /cevoice warns when the requested pack cannot be loaded."""
        self.ui.celune.set_cevoice_and_wait = mock.Mock(return_value=False)

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("cevoice", ["invalid_character"])

        self.ui.celune.set_cevoice_and_wait.assert_called_once_with("invalid_character")
        self.assertEqual(
            self.logs[-1],
            ("Could not switch character to invalid_character.", "warning"),
        )

    def test_cevoice_command_rejects_already_loaded_character(self) -> None:
        """Verify /cevoice warns instead of reloading the active pack."""
        self.ui.celune.set_cevoice_and_wait = mock.Mock(return_value=True)

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

        self.ui.celune.set_cevoice_and_wait.assert_not_called()
        self.assertEqual(
            self.logs[-1],
            ("This character is already loaded.", "warning"),
        )

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
                "Voice prompts are unavailable with the currently loaded model.",
                "warning",
            ),
        )

    def test_attach_command_stages_visual_media_for_persona_reply(self) -> None:
        """Verify /attach validates media and stores Qwen-compatible file URIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "ready.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            self._process_command("attach", [str(image)])

            self.assertEqual(len(self.ui.celune.persona_attachments), 1)
            attachment = self.ui.celune.persona_attachments[0]
            self.assertEqual(attachment["type"], "image")
            self.assertEqual(attachment["path"], attachment_source(image.resolve()))
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
        self.assertEqual(self.logs[-1], ("Usage: /say <text>", "warning"))

    def test_play_command_passes_optional_volume(self) -> None:
        """Verify /play forwards the optional volume argument to Celune."""
        self.ui.celune.play.return_value = True

        with mock.patch(
            "celune.ui.commands.threading.Thread",
            side_effect=self._thread_runs_immediately,
        ):
            self._process_command("play", ["tone.wav", "0.4"])

        self.ui.celune.play.assert_called_once_with("tone.wav", volume=0.4)
        self.assertEqual(self.logs[-1], ("Playing tone.wav at 40% volume", "info"))

    def test_play_command_rejects_invalid_volume(self) -> None:
        """Verify /play validates a numeric optional volume argument."""
        self._process_command("play", ["tone.wav", "loud"])

        self.ui.celune.play.assert_not_called()
        self.assertEqual(
            self.logs[-1],
            ("Invalid volume for 'play', must be numeric.", "warning"),
        )

    def test_say_command_reports_unmatched_ipa_characters(self) -> None:
        """Verify /say keeps the usual unmatched-IPA warning path."""
        with mock.patch(
            "celune.ui.commands.replace_ipa",
            return_value=("hello", 2),
        ):
            self._process_command("say", ["həˈloʊ"])

        self.ui.celune.say.assert_called_once_with("hello", display_text="həˈloʊ")
        self.assertEqual(
            self.logs[-1],
            (
                "Found 2 unmatched IPA characters, output may be inaccurate.",
                "warning",
            ),
        )

    def test_windows_command_split_keeps_literal_backslashes(self) -> None:
        """Verify Windows slash commands keep single-backslash file paths intact."""
        with mock.patch("celune.ui.app.os.name", "nt"):
            parts = CeluneUI.split_command_input(
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

    def test_crossfade_vc_overlap_keeps_mono_audio_one_dimensional(self) -> None:
        """Verify mono live VC overlap crossfades stay valid 1D audio."""
        ui = CeluneUI()
        previous = np.linspace(-1.0, -0.25, 4, dtype=np.float32)
        current = np.linspace(0.25, 1.0, 4, dtype=np.float32)

        blended = ui._crossfade_vc_overlap(previous, current)

        self.assertEqual(blended.shape, (4,))
        self.assertEqual(blended.dtype, np.float32)

    def test_crossfade_vc_overlap_upmixes_mixed_channel_shapes_to_stereo(self) -> None:
        """Verify mixed mono/stereo live VC overlap crossfades return stereo audio."""
        ui = CeluneUI()
        previous = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
        current = np.column_stack(
            (
                np.linspace(1.0, 0.25, 4, dtype=np.float32),
                np.linspace(-1.0, -0.25, 4, dtype=np.float32),
            )
        )

        blended = ui._crossfade_vc_overlap(previous, current)

        self.assertEqual(blended.shape, (4, 2))
        self.assertEqual(blended.dtype, np.float32)

    def test_vc_live_overlap_frames_scale_with_chunk_duration(self) -> None:
        """Verify live VC overlap grows with larger chunk sizes."""
        expected_frames = int(
            48000
            * min(
                ui_app._VC_LIVE_STREAM_MAX_OVERLAP_SECONDS,
                max(
                    ui_app._VC_LIVE_STREAM_OVERLAP_SECONDS,
                    ui_app._VC_LIVE_STREAM_CHUNK_SECONDS
                    * ui_app._VC_LIVE_STREAM_OVERLAP_RATIO,
                ),
            )
        )

        self.assertEqual(
            CeluneUI._vc_live_overlap_frames(48000),
            expected_frames,
        )

    def test_enqueue_vc_submission_chunk_drops_oldest_backlog_item(self) -> None:
        """Verify live VC queueing preserves newer pending chunks when full."""
        submission_queue: queue.Queue[Optional[tuple[np.ndarray, int, str, bool]]] = (
            queue.Queue(maxsize=2)
        )
        oldest_chunk = (np.zeros((8, 2), dtype=np.float32), 48000, "oldest", False)
        newer_chunk = (np.ones((8, 2), dtype=np.float32), 48000, "newer", False)
        fresh_chunk = (np.full((8, 2), 2.0, dtype=np.float32), 48000, "fresh", False)
        submission_queue.put(oldest_chunk)
        submission_queue.put(newer_chunk)

        CeluneUI._enqueue_vc_submission_chunk(submission_queue, fresh_chunk)

        queued_labels: list[str] = []
        while not submission_queue.empty():
            queued = submission_queue.get_nowait()
            self.assertIsNotNone(queued)
            if queued is None:
                self.fail("expected queued VC chunks")
            queued_labels.append(queued[2])

        self.assertEqual(queued_labels, ["newer", "fresh"])

    def test_enqueue_vc_submission_chunk_replaces_stale_backlog_when_single_slot(
        self,
    ) -> None:
        """Verify single-slot live VC queueing still prefers the freshest chunk."""
        submission_queue: queue.Queue[Optional[tuple[np.ndarray, int, str, bool]]] = (
            queue.Queue(maxsize=1)
        )
        stale_chunk = (np.zeros((8, 2), dtype=np.float32), 48000, "stale", False)
        fresh_chunk = (np.ones((8, 2), dtype=np.float32), 48000, "fresh", False)
        submission_queue.put(stale_chunk)

        CeluneUI._enqueue_vc_submission_chunk(submission_queue, fresh_chunk)

        queued = submission_queue.get_nowait()
        self.assertIsNotNone(queued)
        if queued is None:
            self.fail("expected a queued VC chunk")
        self.assertEqual(queued[2], "fresh")

    def test_finish_vc_submission_queue_replaces_stale_chunks_with_final_chunk(
        self,
    ) -> None:
        """Verify live VC stop flushes stale backlog and preserves the final chunk."""
        submission_queue: queue.Queue[Optional[tuple[np.ndarray, int, str, bool]]] = (
            queue.Queue(maxsize=2)
        )
        submission_queue.put(
            (np.zeros((8, 2), dtype=np.float32), 48000, "stale", False)
        )
        final_chunk = (np.ones((4, 2), dtype=np.float32), 48000, "final", True)

        CeluneUI._finish_vc_submission_queue(submission_queue, final_chunk)

        queued_final = submission_queue.get_nowait()
        queued_stop = submission_queue.get_nowait()
        self.assertIsNotNone(queued_final)
        if queued_final is None:
            self.fail("expected a final VC chunk before the stop marker")
        self.assertEqual(queued_final[2], "final")
        self.assertEqual(queued_final[3], True)
        self.assertIsNone(queued_stop)

    def test_vc_feedback_detection_ignores_early_spikes(self) -> None:
        """Verify VC feedback auto-stop does not trigger before enough audio is captured."""
        ui = CeluneUI()
        min_frames = ui._vc_feedback_min_capture_frames(48000)

        self.assertGreater(
            min_frames,
            32,
        )
        self.assertEqual(
            ui._vc_feedback_rise_detected(0.08, 0.24),
            True,
        )
        self.assertEqual(ui_app._VC_FEEDBACK_REQUIRED_CONSECUTIVE_SPIKES, 2)

    def test_textual_ui_requires_attached_celune_on_mount(self) -> None:
        """Verify the Textual UI fails clearly without an attached Celune."""
        ui = CeluneUI()
        with self.assertRaisesRegex(
            RuntimeError,
            f"CeluneUI requires an instance of {APP_NAME} to be set",
        ):
            ui.on_mount()

    def test_textual_ui_mount_enables_stdio_redirects_before_runtime_load(self) -> None:
        """Verify mount captures startup stdio before Celune begins loading."""
        ui = CeluneUI()
        fake_widgets = {
            "#logs": RichLog(),
            "#input": TextArea(),
            "#status": Label(),
            "#resources": Label(),
            "#style": Button(),
            "#vc-mode": Button(),
            "#vc-pitch": Button(),
            "#progress": SimpleNamespace(update=lambda **_: None),
            "#header": Label(),
        }
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                close=lambda: None,
                backend=FakeBackend,
            ),
        )

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            with (
                mock.patch("celune.ui.app.colors.configure_theme"),
                mock.patch("celune.ui.app.default_loader", return_value=None),
                mock.patch("celune.ui.app.ui_resources.prime_usage"),
                mock.patch.object(
                    ui,
                    "query_one",
                    side_effect=lambda selector, *_args: fake_widgets[selector],
                ),
                mock.patch.object(ui, "query", return_value=[]),
                mock.patch.object(ui, "set_interval"),
                mock.patch.object(ui, "set_focus") as set_focus,
                mock.patch.object(ui, "call_after_refresh"),
                mock.patch.object(ui, "safe_status"),
                mock.patch.object(ui, "update_resources"),
                mock.patch.object(ui, "_refresh_status"),
                mock.patch.object(ui, "_refresh_theme_text"),
                mock.patch.object(ui, "_refresh_logs"),
            ):
                ui.on_mount()

            self.assertIs(sys.stdout, ui._log_stdout)
            self.assertIs(sys.stderr, ui._log_stderr)
            self.assertTrue(ui._runtime_log_capture_enabled)
            set_focus.assert_called_once_with(None)
        finally:
            ui.disable_runtime_log_capture()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_textual_ui_mount_binds_callbacks_for_attached_runtime(self) -> None:
        """Verify an attached runtime adopts the Textual UI callbacks on mount."""
        ui = CeluneUI()
        fake_widgets = {
            "#logs": RichLog(),
            "#input": TextArea(),
            "#status": Label(),
            "#resources": Label(),
            "#style": Button(),
            "#vc-mode": Button(),
            "#vc-pitch": Button(),
            "#progress": SimpleNamespace(update=lambda **_: None),
            "#header": Label(),
        }
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                close=lambda: None,
                backend=FakeBackend,
                log_callback=None,
                status_callback=None,
                error_callback=None,
                idle_callback=None,
                queue_avail_callback=None,
                voice_changed_callback=None,
                change_input_state_callback=None,
                change_voice_lock_state_callback=None,
                progress_callback=None,
                glow=SimpleNamespace(fatal=lambda: None),
            ),
        )

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            with (
                mock.patch("celune.ui.app.colors.configure_theme"),
                mock.patch("celune.ui.app.default_loader", return_value=None),
                mock.patch("celune.ui.app.ui_resources.prime_usage"),
                mock.patch.object(
                    ui,
                    "query_one",
                    side_effect=lambda selector, *_args: fake_widgets[selector],
                ),
                mock.patch.object(ui, "query", return_value=[]),
                mock.patch.object(ui, "set_interval"),
                mock.patch.object(ui, "set_focus"),
                mock.patch.object(ui, "call_after_refresh"),
                mock.patch.object(ui, "update_resources"),
                mock.patch.object(ui, "_refresh_status"),
                mock.patch.object(ui, "_refresh_theme_text"),
                mock.patch.object(ui, "_refresh_logs"),
            ):
                ui.on_mount()

            self.assertIs(ui.celune.log_callback.__self__, ui)
            self.assertIs(ui.celune.log_callback.__func__, CeluneUI.tts_log)
            self.assertIs(ui.celune.status_callback.__self__, ui)
            self.assertIs(ui.celune.status_callback.__func__, CeluneUI.safe_status)
            self.assertIs(ui.celune.error_callback.__self__, ui)
            self.assertIs(ui.celune.error_callback.__func__, CeluneUI.error)
            self.assertIs(ui.celune.idle_callback.__self__, ui)
            self.assertIs(ui.celune.idle_callback.__func__, CeluneUI.tts_idle)
            self.assertIs(ui.celune.queue_avail_callback.__self__, ui)
            self.assertIs(
                ui.celune.queue_avail_callback.__func__,
                CeluneUI.tts_queue_avail,
            )
            self.assertIs(ui.celune.voice_changed_callback.__self__, ui)
            self.assertIs(
                ui.celune.voice_changed_callback.__func__,
                CeluneUI.tts_voice_changed,
            )
            self.assertIs(ui.celune.change_input_state_callback.__self__, ui)
            self.assertIs(
                ui.celune.change_input_state_callback.__func__,
                CeluneUI.change_input_state,
            )
            self.assertIs(ui.celune.change_voice_lock_state_callback.__self__, ui)
            self.assertIs(
                ui.celune.change_voice_lock_state_callback.__func__,
                CeluneUI.change_voice_lock_state,
            )
            self.assertIs(ui.celune.progress_callback.__self__, ui)
            self.assertIs(ui.celune.progress_callback.__func__, CeluneUI.safe_progress)
        finally:
            ui.disable_runtime_log_capture()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_runtime_log_capture_restores_stdio_after_shutdown(self) -> None:
        """Verify explicit runtime capture swaps and restores stdio cleanly."""
        ui = CeluneUI()
        ui.safe_log = lambda *_args, **_kwargs: None
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            with (
                mock.patch.object(ui, "_install_runtime_log_redirects"),
                mock.patch.object(ui, "_remove_runtime_log_redirects"),
            ):
                ui.enable_runtime_log_capture()
                self.assertTrue(ui._runtime_log_capture_enabled)
                self.assertIs(sys.stdout, ui._log_stdout)
                self.assertIs(sys.stderr, ui._log_stderr)

                ui.disable_runtime_log_capture()

            self.assertFalse(ui._runtime_log_capture_enabled)
            self.assertIs(sys.stdout, original_stdout)
            self.assertIs(sys.stderr, original_stderr)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_runtime_log_capture_preserves_original_terminal_passthrough(self) -> None:
        """Verify runtime capture keeps ANSI passthrough bound to the original terminal."""
        ui = CeluneUI()
        ui.safe_log = lambda *_args, **_kwargs: None
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        terminal = mock.Mock()
        terminal.isatty.return_value = True
        redirected_stdout = mock.Mock()
        redirected_stderr = mock.Mock()
        redirected_stdout.isatty.return_value = True
        redirected_stderr.isatty.return_value = True
        ui._old_stdout = terminal
        ui._old_stderr = terminal

        try:
            sys.stdout = redirected_stdout
            sys.stderr = redirected_stderr

            with mock.patch.object(ui, "_install_runtime_log_redirects"):
                ui.enable_runtime_log_capture()

            self.assertIs(ui._old_stdout, terminal)
            self.assertIs(ui._old_stderr, terminal)
            self.assertIsNotNone(ui._log_stdout)
            self.assertIsNotNone(ui._log_stderr)
            assert ui._log_stdout is not None
            assert ui._log_stderr is not None
            self.assertIs(ui._log_stdout.underlying_stdout, terminal)
            self.assertIs(ui._log_stdout.underlying_stderr, terminal)
            self.assertIs(ui._log_stderr.underlying_stdout, terminal)
            self.assertIs(ui._log_stderr.underlying_stderr, terminal)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_log_redirect_ansi_forwards_and_flushes_underlying_stdout(self) -> None:
        """Verify ANSI escape forwarding reaches the original terminal stream."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda *_args, **_kwargs: None,
        )

        redirect.ansi(f"\x1b]2;{APP_NAME}\x07")

        stream.write.assert_called_once_with(f"\x1b]2;{APP_NAME}\x07")
        stream.flush.assert_called_once_with()

    def test_log_redirect_reclassifies_warning_like_stdout_lines(self) -> None:
        """Verify raw stdout warning text is surfaced with warning severity."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
        )

        redirect.write(
            "C:/tmp/hub.py:110: FutureWarning: TRANSFORMERS_CACHE is deprecated\n"
        )

        self.assertEqual(
            captured,
            [
                (
                    "C:/tmp/hub.py:110: FutureWarning: TRANSFORMERS_CACHE is deprecated",
                    "warning",
                )
            ],
        )

    def test_log_redirect_suppresses_filtered_partial_stdout_lines(self) -> None:
        """Verify redirected stdout suppressions match message content, not exact chunks."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
            filter_messages={"Loading weights from"},
        )

        redirect.write("Loading weights from C:/models/checkpoint.safetensors\n")

        self.assertEqual(captured, [])

    def test_log_redirect_suppresses_tqdm_progress_lines(self) -> None:
        """Verify tqdm carriage-return progress lines are filtered out."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
            filter_messages={"it/s]"},
        )

        redirect.write(" 10%|#         | 1/10 [00:00<00:03,  2.52it/s]\r")

        self.assertEqual(captured, [])

    def test_load_tts_writes_terminal_title_to_original_stdout(self) -> None:
        """Verify the ready-state title reset targets the original terminal stream."""
        ui = CeluneUI()
        ui.safe_log = lambda *_args, **_kwargs: None
        ui.safe_status = mock.Mock()
        ui.tts_voice_changed = mock.Mock()
        ui.safe_progress = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui._schedule_sleep_timer = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                load=lambda: True,
                voices=("balanced", "bold"),
                current_voice="balanced",
                use_normalization=False,
                dev=False,
                glow=SimpleNamespace(fatal=lambda: None),
                try_play_signal=mock.Mock(return_value=True),
            ),
        )
        terminal = mock.Mock()
        terminal.isatty.return_value = True
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        ui._old_stdout = terminal
        ui._old_stderr = terminal

        try:
            with (
                mock.patch("celune.ui.app.supports_ansi", return_value=True),
                mock.patch.object(ui, "_install_runtime_log_redirects"),
                mock.patch.object(
                    ui,
                    "call_from_thread",
                    side_effect=lambda callback, *args: callback(*args),
                ),
            ):
                load_tts = getattr(CeluneUI.load_tts, "__wrapped__", CeluneUI.load_tts)
                load_tts(ui)

            terminal.write.assert_called_with(f"\x1b]2;{APP_NAME}\x07")
            terminal.flush.assert_called()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

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
            f"CeluneHeadlessUI has no attached {APP_NAME} instance",
            str(caught[0].message),
        )

    def test_load_tts_marks_ui_error_when_startup_returns_false(self) -> None:
        """Verify handled startup failures leave the UI in an error state."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = Button("No Voice Set")
        ui.resources = cast(Label, None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                load=lambda: False,
                dev=False,
                glow=SimpleNamespace(fatal=lambda: None),
            ),
        )
        ui.error = mock.Mock()

        load_tts = getattr(CeluneUI.load_tts, "__wrapped__", CeluneUI.load_tts)
        load_tts(ui)

        ui.error.assert_called_once_with(f"{APP_NAME} could not start")
        self.assertEqual(ui.cur_state, "error")
        self.assertEqual(ui.input_box.placeholder, "Please wait")
        self.assertEqual(ui.style_button.disabled, True)
        self.assertFalse(ui._fatal_error_active)

    def test_load_tts_enters_ui_test_mode_without_error_for_fake_backend(self) -> None:
        """Verify fake-backend UI test mode does not present as a startup failure."""
        ui = CeluneUI()
        ui.safe_status = mock.Mock()
        ui.safe_progress = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui.error = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                load=lambda: True,
                voices=(),
                current_voice=None,
                use_normalization=False,
                backend=SimpleNamespace(is_fake=True),
                dev=False,
                glow=SimpleNamespace(fatal=lambda: None),
                try_play_signal=mock.Mock(return_value=True),
            ),
        )

        load_tts = getattr(CeluneUI.load_tts, "__wrapped__", CeluneUI.load_tts)
        patched_modules = dict(sys.modules)
        patched_modules.pop("pytest", None)
        with mock.patch("celune.ui.app.sys.modules", patched_modules):
            load_tts(ui)

        ui.safe_status.assert_called_once_with(string("ui.test_mode_active"))
        ui.safe_progress.assert_called_once_with(1, 1)
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.change_voice_lock_state.assert_called_once_with(locked=True)
        ui.error.assert_not_called()
        self.assertNotEqual(ui.cur_state, "error")

    def test_submit_text_is_noop_in_ui_test_mode(self) -> None:
        """Verify interactive fake-backend UI test mode ignores submitted input."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.input_box.load_text("hello world")
        ui.safe_status = mock.Mock()
        ui.process_command = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                backend=SimpleNamespace(is_fake=True),
                cur_state="idle",
                sleeping=False,
                config={},
                think=mock.Mock(),
                say=mock.Mock(),
            ),
        )

        patched_modules = dict(sys.modules)
        patched_modules.pop("pytest", None)
        with mock.patch("celune.ui.app.sys.modules", patched_modules):
            handled = ui._submit_text(ui.input_box.text)

        self.assertTrue(handled)
        self.assertEqual(ui.input_box.text, "")
        ui.safe_status.assert_called_once_with(string("ui.test_mode_active"))
        ui.process_command.assert_not_called()
        ui.celune.think.assert_not_called()
        ui.celune.say.assert_not_called()

    def test_tts_idle_does_not_recover_error_state_before_runtime_ready(self) -> None:
        """Verify signal callbacks cannot revert a failed startup back to idle."""
        ui = CeluneUI()
        ui.celune_ready = False
        ui.cur_state = "error"
        ui.input_box = TextArea()
        ui.style_button = Button("No Voice Set")
        ui.resources = cast(Label, None)
        ui.status = Label()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                locked=True,
                sleeping=False,
                is_in_tutorial=False,
                voices=(),
                cur_state="error",
            ),
        )

        ui.tts_idle()

        self.assertEqual(ui.cur_state, "error")
        self.assertEqual(ui.input_box.placeholder, "Please wait")
        self.assertEqual(ui.style_button.disabled, True)

    def test_tts_idle_keeps_controls_locked_while_runtime_is_reloading(self) -> None:
        """Verify idle playback callbacks do not unlock the UI mid-reload."""
        ui = CeluneUI()
        ui.celune_ready = True
        ui.cur_state = "idle"
        ui.input_box = TextArea()
        ui.style_button = Button("Balanced")
        ui.resources = cast(Label, None)
        ui.status = Label()
        ui.change_input_state = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui.safe_status = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                locked=True,
                sleeping=False,
                is_in_tutorial=False,
                voices=("balanced", "bold"),
                cur_state="reloading",
            ),
        )

        ui.tts_idle()

        self.assertEqual(ui.celune.locked, True)
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.change_voice_lock_state.assert_called_once_with(locked=True)
        ui.safe_status.assert_not_called()

    def test_on_button_pressed_ignores_voice_switch_when_no_voices_loaded(self) -> None:
        """Verify voice cycling is blocked cleanly when startup left no voices loaded."""
        ui = CeluneUI()
        ui.celune_ready = False
        ui.style_button = Button("No Voice Set")
        ui.safe_log = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                is_in_tutorial=False,
                voices=(),
            ),
        )

        ui.on_button_pressed(
            cast(Button.Pressed, SimpleNamespace(button=ui.style_button))
        )

        ui.safe_log.assert_called_once_with("No voices are loaded.", "warning")
        ui.change_voice_lock_state.assert_called_once_with(locked=True)

    def test_textual_resource_footer_only_advertises_ctrl_q_exit(self) -> None:
        """Verify the Textual UI footer no longer advertises CTRL+C exit."""
        celune = cast(
            Celune,
            SimpleNamespace(
                is_in_tutorial=False,
                config={"theme": "dark"},
                backend=SimpleNamespace(current_seed=None),
                input_mode="text_to_speech",
            ),
        )

        pages = ui_resources.resource_pages(celune, "celune")

        exit_page = next(page for page in pages if "CTRL+Q exit" in page)
        self.assertNotIn("CTRL+C", exit_page)
        self.assertNotIn("CTRL+R toggle recording", pages)

    def test_textual_resource_footer_advertises_ctrl_r_only_in_vc_mode(self) -> None:
        """Verify the resource footer advertises recording only while VC mode is active."""
        celune = cast(
            Celune,
            SimpleNamespace(
                is_in_tutorial=False,
                config={"theme": "dark"},
                backend=SimpleNamespace(current_seed=None),
                input_mode="voice_conversion",
            ),
        )

        pages = ui_resources.resource_pages(celune, "celune")

        self.assertIn("CTRL+R toggle recording", pages)

    def test_ctrl_r_toggles_tui_vc_recording(self) -> None:
        """Verify CTRL+R streams VC audio live and flushes the final tail on stop."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None
        queued_segments: list[tuple[np.ndarray, int, str, Optional[int], bool]] = []
        finished_source_ids: list[Optional[int]] = []

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for VC recording tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ) as mock_query_devices,
            mock.patch(
                "celune.ui.app.sd.InputStream",
                side_effect=FakeInputStream,
            ) as mock_input_stream,
            mock.patch(
                "celune.ui.app.queue_streaming_sfx_audio",
                side_effect=lambda engine, audio, sample_rate, label, source_id=None, reset_ready_announcement=False: (
                    queued_segments.append(
                        (
                            np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate,
                            label,
                            source_id,
                            reset_ready_announcement,
                        )
                    )
                    or (1 if source_id is None else source_id)
                ),
            ),
            mock.patch(
                "celune.ui.app.finish_streaming_sfx_audio",
                side_effect=lambda engine, source_id: finished_source_ids.append(
                    source_id
                ),
            ),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                self.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    cast(
                        ui_app._VCAudioCallback,
                        captured_callback,
                    ),
                    np.ones((60000, 2), dtype=np.float32),
                )
                for _ in range(50):
                    if cast(mock.Mock, ui.celune.convert_audio).call_count >= 1:
                        break
                    time.sleep(0.01)
                invoke_captured_callback(
                    cast(
                        ui_app._VCAudioCallback,
                        captured_callback,
                    ),
                    np.ones((60000, 2), dtype=np.float32),
                )

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        mock_query_devices.assert_any_call(
            device="Stereo Mix (Realtek)",
            kind="input",
        )
        self.assertGreaterEqual(mock_query_devices.call_count, 1)
        self.assertEqual(
            mock_input_stream.call_args.kwargs["device"],
            "Stereo Mix (Realtek)",
        )
        for _ in range(50):
            if cast(mock.Mock, ui.celune.convert_audio).call_count >= 1:
                break
            time.sleep(0.01)

        self.assertGreaterEqual(cast(mock.Mock, ui.celune.convert_audio).call_count, 1)
        convert_calls = cast(mock.Mock, ui.celune.convert_audio).call_args_list
        first_convert_args = convert_calls[0].args
        final_convert_args = convert_calls[-1].args
        expected_first_chunk_frames = 60000 - ui._vc_live_overlap_frames(48000)
        self.assertEqual(first_convert_args[1], 48000)
        self.assertEqual(final_convert_args[1], 48000)
        self.assertEqual(
            convert_calls[-1].kwargs["label"],
            "Stereo Mix",
        )
        self.assertGreaterEqual(
            len(first_convert_args[0]),
            expected_first_chunk_frames,
        )
        self.assertGreater(len(final_convert_args[0]), 8)
        self.assertGreaterEqual(len(queued_segments), 1)
        self.assertEqual(queued_segments[0][3], None)
        self.assertTrue(queued_segments[0][4])
        self.assertTrue(all(segment[3] == 1 for segment in queued_segments[1:]))
        self.assertIn(1, finished_source_ids)
        self.assertEqual(
            [source_id for source_id in finished_source_ids if source_id is not None],
            [1],
        )

    def test_vc_recording_feedback_spike_stops_live_stream(self) -> None:
        """Verify a sudden microphone RMS spike stops live VC capture automatically."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                play_audio=mock.Mock(return_value=True),
                is_in_tutorial=False,
                dev=False,
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        class FakeInputStream:
            """Tiny input-stream fake for VC feedback-stop tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            def missing_callback(
                _audio: np.ndarray,
                _frames: int,
                _time_info: Optional[tuple[float, float, float]],
                _status: Optional[sd.CallbackFlags],
            ) -> None:
                raise AssertionError("recording callback was not registered")

            recording_callback = captured_callback or missing_callback

            min_capture_frames = ui._vc_feedback_min_capture_frames(48000)
            safe_frames = max(min_capture_frames - 32, 1)
            recording_callback(
                np.full((safe_frames, 2), 0.08, dtype=np.float32),
                safe_frames,
                None,
                None,
            )
            recording_callback(
                np.full((32, 2), 0.24, dtype=np.float32),
                32,
                None,
                None,
            )
            recording_callback(
                np.full((32, 2), 0.52, dtype=np.float32),
                32,
                None,
                None,
            )

        for _ in range(50):
            if not ui._vc_recording_active():
                break
            time.sleep(0.01)

        self.assertEqual(ui._vc_recording_active(), False)
        for _ in range(50):
            if any(
                "feedback" in call.args[0].lower()
                for call in cast(mock.Mock, ui.safe_log).call_args_list
                if call.args
            ):
                break
            time.sleep(0.01)
        self.assertTrue(
            any(
                "feedback" in call.args[0].lower()
                for call in cast(mock.Mock, ui.safe_log).call_args_list
                if call.args
            )
        )

    def test_ctrl_r_wakes_sleeping_celune_without_starting_recording(self) -> None:
        """Verify CTRL+R wakes a sleeping VC runtime instead of starting capture."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                sleeping=True,
                cur_state="sleeping",
                config={},
            ),
        )
        ui.safe_status = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.wake_from_sleep = mock.Mock()
        ui._cancel_sleep_timer = mock.Mock()
        ui.toggle_vc_recording = mock.Mock(return_value=True)

        start_event = SimpleNamespace(
            key="ctrl+r",
            prevent_default=mock.Mock(),
            stop=mock.Mock(),
        )

        ui.on_key(cast(events.Key, start_event))

        ui._cancel_sleep_timer.assert_called_once()
        ui.safe_status.assert_called_once_with("Waking up")
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.wake_from_sleep.assert_called_once_with()
        ui.toggle_vc_recording.assert_not_called()
        start_event.prevent_default.assert_called_once_with()
        start_event.stop.assert_called_once_with()

    def test_graceful_exit_uses_live_vc_shutdown_path(self) -> None:
        """Verify app exit stops live VC before closing the runtime."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(Celune, SimpleNamespace(close=mock.Mock()))
        ui._shutdown_live_vc_recording = mock.Mock()
        ui.exit = mock.Mock()

        ui._graceful_exit()

        self.assertEqual(ui.cur_state, "exiting")
        ui._shutdown_live_vc_recording.assert_called_once_with()
        cast(mock.Mock, ui.celune.close).assert_called_once_with()
        ui.exit.assert_called_once_with()

    def test_start_vc_recording_logs_ambiguous_input_device_as_warning(self) -> None:
        """Verify ambiguous VC input device matches do not bubble up as exceptions."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                sleeping=False,
                cur_state="ready",
                config={"input_device": "Razer Kraken V4 - Chat"},
            ),
        )
        ui.safe_log = mock.Mock()

        with mock.patch(
            "celune.ui.app.resolve_audio_device_with_info",
            side_effect=ValueError("ambiguous input device"),
        ):
            self.assertEqual(ui._start_vc_recording(), False)

        ui.safe_log.assert_called_once_with("ambiguous input device", "warning")

    def test_gpu_usage_handles_closed_stdout_pipe(self) -> None:
        """Verify resource polling ignores closed-pipe nvidia-smi failures."""
        proc = mock.Mock()
        proc.poll.return_value = 0
        proc.communicate.side_effect = ValueError("I/O operation on closed file.")

        with mock.patch("celune.ui.resources._NVIDIA_SMI", "nvidia-smi"):
            previous_proc = ui_resources._NVIDIA_SMI_PROC
            previous_usage = ui_resources._NVIDIA_SMI_USAGE
            ui_resources._NVIDIA_SMI_PROC = proc
            ui_resources._NVIDIA_SMI_USAGE = 42
            try:
                self.assertIsNone(ui_resources.gpu_usage())
                self.assertIsNone(ui_resources._NVIDIA_SMI_PROC)
                self.assertIsNone(ui_resources._NVIDIA_SMI_USAGE)
            finally:
                ui_resources._NVIDIA_SMI_PROC = previous_proc
                ui_resources._NVIDIA_SMI_USAGE = previous_usage

    def test_textual_input_lock_update_with_persona_on_ui_thread(self) -> None:
        """Verify input state updates update with Persona."""
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

        self.assertEqual(ui.input_box.placeholder, string("ui.input_placeholder"))
        self.assertEqual(ui.style_button.disabled, False)
        available.assert_called_once_with()
        thread_cls.return_value.start.assert_called_once()

    def test_change_input_state_reveals_vc_buttons_after_backend_switch(self) -> None:
        """Verify VC controls appear when the UI unlocks into voice conversion mode."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = Button("Voice")
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui.resources = cast(Label, None)
        ui._input_locked = True
        ui.vc_mode_button.display = False
        ui.vc_pitch_button.display = False
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                vc_backend=SimpleNamespace(),
                vc_f0_condition=False,
                vc_pitch_shift=0,
            ),
        )

        with mock.patch("celune.ui.app.threading.Thread") as thread_cls:
            ui.change_input_state(locked=False)

        self.assertEqual(ui.vc_mode_button.display, True)
        self.assertEqual(ui.vc_pitch_button.display, True)
        self.assertEqual(ui.vc_mode_button.disabled, False)
        self.assertEqual(ui.vc_pitch_button.disabled, False)
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

        ui._persona_available = ui.persona_loaded()
        self.assertEqual(ui.normal_input_placeholder(), string("ui.input_placeholder"))

        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={
                    "vram": "high",
                    "persona": cast(JSONSerializable, persona_config),
                },
                vision=SimpleNamespace(),
            ),
        )
        ui._persona_available = ui.persona_loaded()
        self.assertEqual(ui.normal_input_placeholder(), string("ui.say_placeholder"))

    def test_placeholder_uses_voice_changer_text_when_vc_backend_is_selected(
        self,
    ) -> None:
        """Verify the input placeholder follows VC backend selection."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = Button("Voice")
        ui.resources = cast(Label, None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                vc_backend=SimpleNamespace(),
                vision=SimpleNamespace(),
            ),
        )

        self.assertEqual(
            ui.normal_input_placeholder(),
            string("ui.voice_changer_placeholder"),
        )

    def test_refresh_vc_controls_hides_buttons_outside_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC-only buttons collapse away while the UI is in TTS mode."""
        ui = CeluneUI()
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui._cancel_vc_recording = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                vc_backend=None,
                vc_f0_condition=False,
                vc_pitch_shift=0,
            ),
        )

        ui.refresh_vc_controls()

        self.assertEqual(ui.vc_mode_button.display, False)
        self.assertEqual(ui.vc_pitch_button.display, False)
        self.assertEqual(ui.vc_mode_button.disabled, True)
        self.assertEqual(ui.vc_pitch_button.disabled, True)
        ui._cancel_vc_recording.assert_called_once_with(announce=False)

    def test_refresh_vc_controls_shows_buttons_in_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC-only buttons return when a VC backend is active."""
        ui = CeluneUI()
        ui._input_locked = False
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui._cancel_vc_recording = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                vc_backend=SimpleNamespace(),
                vc_f0_condition=True,
                vc_pitch_shift=3,
            ),
        )

        ui.refresh_vc_controls()

        self.assertEqual(ui.vc_mode_button.display, True)
        self.assertEqual(ui.vc_pitch_button.display, True)
        self.assertEqual(ui.vc_mode_button.label, string("ui.vc_mode_sing"))
        self.assertEqual(
            ui.vc_pitch_button.label,
            string("ui.vc_pitch_button", value="+3"),
        )
        self.assertEqual(ui.vc_mode_button.disabled, False)
        self.assertEqual(ui.vc_pitch_button.disabled, False)
        ui._cancel_vc_recording.assert_not_called()

    def test_runtime_logger_warning_is_routed_into_ui_logs(self) -> None:
        """Verify external Python logger warnings are routed into the UI logs."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("torch.utils.flop_counter")

        ui.install_runtime_log_redirects()
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

    def test_runtime_warning_capture_routes_py_warnings_triton_message(self) -> None:
        """Verify Python warnings formatting is normalized for Triton warnings."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("py.warnings")
        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning(
            "C:\\path\\flop_counter.py:29: UserWarning: triton not found; flop "
            "counting will not work for triton kernels"
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

    def test_runtime_huggingface_logger_error_is_routed_into_ui_logs(self) -> None:
        """Verify Hugging Face logger errors are routed into the UI log widget."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("huggingface_hub")
        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.error("download failed because the connection dropped")

        self.assertEqual(
            captured,
            [
                (
                    "Internal runtime error: download failed because the "
                    "connection dropped",
                    "error",
                )
            ],
        )

    def test_runtime_global_log_redirect_captures_unlisted_external_logger(
        self,
    ) -> None:
        """Verify arbitrary external loggers are captured without per-backend wiring."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("some.third_party.backend")

        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning("backend emitted a warning")

        self.assertEqual(
            captured,
            [("Internal runtime warning: backend emitted a warning", "warning")],
        )

    def test_runtime_logger_suppresses_filtered_messages(self) -> None:
        """Verify runtime logger redirection honors the shared suppression list."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info": captured.append((msg, severity))

        logger = logging.getLogger("some.third_party.backend")

        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning("Loading weights from C:/models/checkpoint.safetensors")

        self.assertEqual(captured, [])

    def test_safe_status_marquees_long_text_for_narrow_status_label(self) -> None:
        """Verify long status text scrolls instead of clipping."""

        class FakeLabel:
            """Tiny fake status label with a constrained width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=14)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        ui.safe_status("Playing C:/Users/user/Music/really_long_filename_demo.wav")
        first = fake_status.rendered
        ui.advance_status_marquee()
        second = fake_status.rendered

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("  "))
        self.assertTrue(second.startswith("  "))

    def test_safe_status_keeps_short_text_static(self) -> None:
        """Verify short status text does not marquee."""

        class FakeLabel:
            """Tiny fake status label with a constrained width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=40)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        ui.safe_status("Playing")
        first = fake_status.rendered
        ui.advance_status_marquee()

        self.assertEqual(first, fake_status.rendered)

    def test_safe_status_repaints_terminal_accent_for_error(self) -> None:
        """Verify error status repaints the terminal shell accent to the error color."""

        class FakeLabel:
            """Simple label test double with mutable styles."""

            def __init__(self) -> None:
                self.size = SimpleNamespace(width=40)
                self.styles = SimpleNamespace(color=None, border=None, border_top=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Capture the rendered label value.

                Args:
                    value: The latest label content.
                """
                self.rendered = value

        class FakeWidget:
            """Simple widget test double with mutable styles."""

            def __init__(self) -> None:
                self.styles = SimpleNamespace(
                    color=None,
                    border=None,
                    border_top=None,
                    background=None,
                    scrollbar_color=None,
                    scrollbar_color_hover=None,
                    scrollbar_color_active=None,
                    scrollbar_background=None,
                    scrollbar_background_hover=None,
                    scrollbar_background_active=None,
                )
                self.rendered = ""

            def update(self, value: str = "") -> None:
                """Capture the rendered label value.

                Args:
                    value: The latest label content.
                """
                self.rendered = value

        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(config={}, backend=FakeBackend, is_in_tutorial=False),
        )
        ui.logs = cast(RichLog, FakeWidget())
        ui.input_box = cast(TextArea, FakeWidget())
        ui.style_button = cast(Button, FakeWidget())
        ui.resources = cast(Label, FakeWidget())
        ui.header = cast(Label, FakeWidget())
        ui.progress_bar = cast(Button, FakeWidget())
        ui.header_lines = (cast(Label, FakeWidget()), cast(Label, FakeWidget()))
        ui.status = cast(Label, FakeLabel())

        ui._fatal_error_active = True
        ui.safe_status("Could not start", "error")

        expected = severity_color(ui.active_theme_name, "error")
        self.assertEqual(ui.theme, "celune_error")
        self.assertIsNone(ui.logs.styles.color)
        self.assertIsNone(ui.logs.styles.border)
        self.assertIsNone(ui.logs.styles.background)
        self.assertIsNone(ui.logs.styles.scrollbar_color)
        self.assertIsNone(ui.input_box.styles.border)
        self.assertIsNone(ui.input_box.styles.background)
        self.assertIsNone(ui.input_box.styles.scrollbar_color)
        self.assertIsNone(ui.style_button.styles.border)
        self.assertIsNone(ui.style_button.styles.background)
        self.assertIsNone(ui.resources.styles.color)
        self.assertIsNone(ui.header.styles.color)
        self.assertIsNone(ui.header_lines[0].styles.border_top)
        self.assertIsNone(ui.progress_bar.styles.color)
        self.assertIsNone(ui.progress_bar.styles.background)
        self.assertEqual(ui.status.styles.color, expected)
        error_theme = ui.get_theme("celune_error")
        self.assertIsNotNone(error_theme)
        self.assertEqual(error_theme.primary, colors.ERROR_DARK_ACCENT)
        self.assertEqual(error_theme.accent, expected)
        self.assertEqual(
            error_theme.foreground,
            colors.ensure_contrast(
                colors.ERROR_HIGHLIGHT,
                colors.ERROR_BACKGROUND,
                7.0,
            ),
        )
        self.assertEqual(error_theme.background, colors.ERROR_BACKGROUND)

    def test_nonfatal_error_status_keeps_normal_theme(self) -> None:
        """Verify ordinary error statuses do not switch into the fatal error theme."""
        ui = CeluneUI()
        ui.status = Label()

        ui.safe_status("Minor issue", "error")

        self.assertEqual(ui.theme, "celune")
        self.assertFalse(ui._fatal_error_active)

    def test_wrapped_fatal_glow_activates_error_theme(self) -> None:
        """Verify the fatal theme only activates through wrapped ``glow.fatal()``."""
        ui = CeluneUI()
        ui.status = Label()
        ui.celune = cast(
            Celune,
            SimpleNamespace(glow=SimpleNamespace(fatal=mock.Mock())),
        )

        ui.wrap_runtime_fatal_glow()
        ui.celune.glow.fatal()

        self.assertTrue(ui._fatal_error_active)
        self.assertEqual(ui.theme, "celune_error")

    def test_fatal_theme_stays_pinned_after_later_nonfatal_status_updates(self) -> None:
        """Verify later routine events cannot clear the fatal UI theme once activated."""
        ui = CeluneUI()
        ui.status = Label()
        ui.celune = cast(
            Celune,
            SimpleNamespace(glow=SimpleNamespace(fatal=mock.Mock())),
        )

        ui.wrap_runtime_fatal_glow()
        ui.celune.glow.fatal()
        ui.safe_status("Idle")
        ui.safe_status("Speaking")

        self.assertTrue(ui._fatal_error_active)
        self.assertEqual(ui.theme, "celune_error")

    def test_fatal_status_text_ignores_later_idle_updates(self) -> None:
        """Verify fatal UI status text is not overwritten by later normal lifecycle events."""
        ui = CeluneUI()
        ui.status = Label()
        ui.safe_status("Celune could not warm up", "error")
        ui._fatal_error_active = True

        ui.safe_status("Idle")
        ui.safe_status("Speaking")

        self.assertEqual(ui._status_text, "Celune could not warm up")
        self.assertEqual(ui.status_severity, "error")

    def test_runtime_error_themes_cover_dark_and_light_modes(self) -> None:
        """Verify both dedicated runtime error themes are registered correctly."""
        ui = CeluneUI()

        ui.register_runtime_error_themes()

        dark_error = ui.get_theme("celune_error")
        light_error = ui.get_theme("celune_light_error")
        self.assertIsNotNone(dark_error)
        self.assertIsNotNone(light_error)
        self.assertEqual(dark_error.background, colors.ERROR_BACKGROUND)
        self.assertEqual(light_error.background, colors.ERROR_LIGHT_BACKGROUND)
        self.assertEqual(dark_error.accent, colors.THEME.error)
        self.assertEqual(light_error.accent, colors.THEME_LIGHT.error)
        self.assertEqual(dark_error.primary, colors.ERROR_DARK_ACCENT)
        self.assertEqual(light_error.primary, colors.ERROR_DARK_ACCENT)
        self.assertEqual(
            dark_error.foreground,
            colors.ensure_contrast(
                colors.ERROR_HIGHLIGHT,
                colors.ERROR_BACKGROUND,
                7.0,
            ),
        )
        self.assertEqual(
            light_error.foreground,
            colors.ensure_contrast(
                colors.ERROR_HIGHLIGHT,
                colors.ERROR_LIGHT_BACKGROUND,
                7.0,
            ),
        )

    def test_resize_repaints_status_after_width_change(self) -> None:
        """Verify widening the status label re-renders the current text immediately."""

        class FakeLabel:
            """Tiny fake status label with a mutable width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=14)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        message = "Playing C:/Users/user/Music/really_long_filename_demo.wav"
        ui.safe_status(message)
        narrow = fake_status.rendered

        fake_status.size = SimpleNamespace(width=96)
        ui.on_resize(cast(events.Resize, SimpleNamespace()))

        self.assertNotEqual(narrow, fake_status.rendered)
        self.assertEqual(fake_status.rendered, f"  {message}")
