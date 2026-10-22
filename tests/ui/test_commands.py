# SPDX-License-Identifier: Apache-2.0
"""Tests for Textual command routing and agent-status presentation."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import tempfile
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from typing import Optional, cast

import numpy as np
from textual import events
from textual.widgets import TextArea

from celune.i18n import string
from celune.celune import Celune
from celune.agent import AgentTaskState
from celune.ui.commands import process_command, attachment_source
from celune.ui.app import (
    CeluneUI,
    VoiceButton,
    ButtonActions,
)
from celune.typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
)

from ..platform import WINDOWS_ONLY
from ..support import FakeBackend, FakeVCBackend, CeluneTestCase


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
            vc_backend=SimpleNamespace(),
        )

    def test_help_includes_cesn_glossary_for_singing_backend(self) -> None:
        """Show CESN syntax help only when the active TTS backend sings."""
        self.ui.celune.backend = SimpleNamespace(name="soulxsinger", singing=True)
        self.ui.celune.vision = None

        self._process_command("help", [])

        messages = [message for message, _severity in self.logs]
        assert string("commands.help_cesn_header") in messages
        assert string("commands.help_cesn_order") in messages
        assert string("commands.help_cesn_notes") in messages
        assert string("commands.help_cesn_ipa") in messages
        assert string("commands.help_cesn_slide") in messages
        assert string("commands.help_cesn_slur") in messages
        assert string("commands.help_cesn_singer") in messages
        assert string("commands.help_cesn_tutorial") in messages

    def test_help_omits_cesn_tutorial_without_singing_backend(self) -> None:
        """Hide the CESN tutorial link when the active backend cannot sing."""
        self.ui.celune.backend = SimpleNamespace(name="mini", singing=False)
        self._process_command("help", [])

        messages = [message for message, _severity in self.logs]
        assert string("commands.help_cesn_tutorial") not in messages

    def _process_command(self, command: str, args: list[str]) -> None:
        """Process one command against the typed UI test double."""
        process_command(cast(CeluneUI, self.ui), command, args)

    def test_restartaudio_command_restarts_host_audio_server(self) -> None:
        """Verify the audio recovery command reports a successful restart."""
        with (
            mock.patch("celune.ui.commands.restart_audio_server") as restart,
            mock.patch("celune.ui.commands.close_stream") as close_stream,
            mock.patch(
                "celune.ui.commands.threading.Thread",
                side_effect=self._thread_runs_immediately,
            ),
        ):
            self._process_command("restartaudio", [])

        restart.assert_called_once_with()
        close_stream.assert_called_once_with(self.ui.celune, abort=True)
        self.ui.celune.try_play_signal.assert_called_once_with("readiness")
        assert self.logs[-1] == (
            "Host audio server restarted.",
            "info",
        )

    def test_vram_command_reports_process_and_component_usage(self) -> None:
        """Verify /vram renders aggregate and per-component CUDA diagnostics."""
        report = {
            "available": True,
            "allocated_bytes": 2 * 1024**3,
            "reserved_bytes": 3 * 1024**3,
            "peak_allocated_bytes": 4 * 1024**3,
            "components": [
                {
                    "name": "tts/qwen3",
                    "loaded": True,
                    "available": True,
                    "device": "cuda:0",
                    "allocated_bytes": 1024**3,
                    "reserved_bytes": 3 * 1024**3,
                    "peak_allocated_bytes": 4 * 1024**3,
                }
            ],
        }
        with mock.patch("celune.ui.commands.runtime_vram_report", return_value=report):
            self._process_command("vram", [])

        assert self.logs == [
            ("--- Memory usage statistics ---", "info"),
            (
                "Text to speech (qwen3): 1.00 GiB/3.00 GiB/4.00 GiB (cuda:0)",
                "info",
            ),
            ("Legend: allocated memory, reserved memory, peak memory", "info"),
        ]

    def test_vram_command_reports_cpu_only_components(self) -> None:
        """Verify /vram reports loaded components on the CPU."""
        report = {
            "available": True,
            "allocated_bytes": 2 * 1024**3,
            "reserved_bytes": 3 * 1024**3,
            "peak_allocated_bytes": 4 * 1024**3,
            "components": [
                {
                    "name": "normalizer",
                    "loaded": True,
                    "available": True,
                    "device": "cpu",
                    "allocated_bytes": 32 * 1024**2,
                    "reserved_bytes": 32 * 1024**2,
                    "peak_allocated_bytes": 32 * 1024**2,
                }
            ],
        }
        with mock.patch("celune.ui.commands.runtime_vram_report", return_value=report):
            self._process_command("vram", [])

        assert self.logs == [
            ("--- Memory usage statistics ---", "info"),
            (
                "Normalizer: 32.00 MiB/-/- (cpu)",
                "info",
            ),
            ("Legend: allocated memory, reserved memory, peak memory", "info"),
        ]

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
        self.ui.celune.set_backend_async = mock.AsyncMock(return_value=True)

        with mock.patch(
            "celune.ui.commands.active_runtime_backend_name",
            return_value="fake-vc",
        ) as active_backend_name:
            self._process_command("backend", ["fake-vc"])

        active_backend_name.assert_called_once_with(self.ui.celune)
        self.ui.celune.set_backend_async.assert_not_called()
        assert self.logs[-1][1] == "warning"

    def test_voice_conversion_mode_command_updates_f0_conditioning(self) -> None:
        """Verify the VC mode command updates engine and backend state."""
        self.ui.celune.input_mode = "voice_conversion"
        self.ui.celune.vc_backend = SimpleNamespace(f0_condition=False)

        self._process_command("vcmode", ["sing"])

        assert self.ui.celune.vc_f0_condition
        assert self.ui.celune.vc_backend.f0_condition
        assert self.logs[-1] == ("Voice conversion mode set to Sing.", "info")
        assert self.ui.refresh_vc_controls.call_count == 1

    def test_vcdebug_command_starts_and_stops_capture(self) -> None:
        """Verify /vcdebug is available only as an explicit VC capture toggle."""
        self.ui.celune.input_mode = "voice_conversion"
        capture = SimpleNamespace(directory=Path("temporary"))
        self.ui._vc_debug_capture = None
        self.ui.stop_vc_debug_capture = mock.Mock(return_value=True)

        with mock.patch("celune.vc.VCDebugCapture", return_value=capture):
            self._process_command("vcdebug", ["on"])

        assert self.ui._vc_debug_capture is capture
        assert self.logs[-1] == (
            "VC debug capture started. Run /vcdebug off to save it.",
            "info",
        )

        self._process_command("vcdebug", ["off"])

        self.ui.stop_vc_debug_capture.assert_called_once_with(announce=True)
        assert self.logs[-1] == (
            "Stopping VC debug capture and saving audio.",
            "info",
        )

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

        self._process_command("vcpitch", ["-2"])
        assert self.logs[-1] == (
            string("commands.unknown_command", command="vcpitch"),
            "warning",
        )

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
