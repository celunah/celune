# SPDX-License-Identifier: MIT
"""Tests for Celune's browser-facing API UI."""

import asyncio
from queue import Queue
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, mock

import numpy as np
from starlette.requests import Request
from starlette.responses import Response

from celune import api
from celune.celune import Celune
from celune.i18n import string
from celune.pipeline import SpeechStreamQueue


class ApiWebUITests(TestCase):
    """Tests for the mounted Gradio browser UI helpers."""

    def setUp(self) -> None:
        self.previous_celune = api.bound_celune
        self.previous_status_text = api.webui_status_text
        self.previous_status_severity = api.webui_status_severity
        self.previous_seeded = api.webui_logs_seeded
        self.previous_resource_page = api.webui_resource_page
        self.previous_last_resource_advance = api.webui_last_resource_advance
        self.previous_last_probed_state = api.webui_last_probed_state
        self.previous_input_locked = api.webui_input_locked
        self.previous_input_placeholder = api.webui_input_placeholder
        self.previous_voice_locked = api.webui_voice_locked
        self.previous_theme_style = api.webui_theme_style
        self.previous_status_source = api.webui_status_source
        self.previous_status_updated_at = api.webui_status_updated_at
        self.previous_auth_token = api.auth_token
        self.previous_logs = list(api.webui_log_lines)
        api.bound_celune = None
        api.auth_token = None
        api.webui_log_lines.clear()
        api.webui_logs_seeded = True
        api.webui_resource_page = 0
        api.webui_last_resource_advance = 0.0
        api.webui_last_probed_state = None
        api.webui_input_locked = True
        api.webui_input_placeholder = "Please wait"
        api.webui_voice_locked = True
        api.webui_status_source = "probe"
        api.webui_status_updated_at = 0.0
        api.set_webui_status("Starting up")

    def tearDown(self) -> None:
        api.bound_celune = self.previous_celune
        api.webui_status_text = self.previous_status_text
        api.webui_status_severity = self.previous_status_severity
        api.webui_logs_seeded = self.previous_seeded
        api.webui_resource_page = self.previous_resource_page
        api.webui_last_resource_advance = self.previous_last_resource_advance
        api.webui_last_probed_state = self.previous_last_probed_state
        api.webui_input_locked = self.previous_input_locked
        api.webui_input_placeholder = self.previous_input_placeholder
        api.webui_voice_locked = self.previous_voice_locked
        api.webui_theme_style = self.previous_theme_style
        api.webui_status_source = self.previous_status_source
        api.webui_status_updated_at = self.previous_status_updated_at
        api.auth_token = self.previous_auth_token
        api.webui_log_lines.clear()
        api.webui_log_lines.extend(self.previous_logs)

    @staticmethod
    async def _passthrough_response(_request: Request) -> Response:
        """Return a simple response for middleware pass-through tests."""
        return Response("ok")

    @staticmethod
    def _request(path: str, method: str = "GET") -> Request:
        """Build one lightweight test request."""
        return Request(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 2060),
                "scheme": "http",
                "server": ("127.0.0.1", 2060),
            }
        )

    def test_root_redirects_to_browser_ui(self) -> None:
        """Verify the fallback root now forwards users to the browser UI."""
        response = api.root()
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/ui")

    def test_browser_ui_requests_bypass_api_security_detection(self) -> None:
        """Verify mounted browser UI paths are recognized by the API middleware."""
        ui_request = self._request("/ui/assets/index.js")
        api_request = self._request("/v1/version")

        self.assertEqual(api.is_browser_ui_request(ui_request), True)
        self.assertEqual(api.is_browser_ui_request(api_request), False)

    def test_api_security_allows_public_read_only_routes_without_token(self) -> None:
        """Verify safe read-only routes stay reachable even when API auth is enabled."""
        api.auth_token = "secret"

        for path in ("/", "/favicon.ico", "/v1", "/v1/version", "/ui"):
            response = asyncio.run(
                api.api_security(
                    self._request(path),
                    self._passthrough_response,
                )
            )
            self.assertEqual(response.status_code, 200, path)

    def test_api_security_requires_token_for_generating_routes(self) -> None:
        """Verify protected API routes still reject unauthenticated requests."""
        api.auth_token = "secret"

        response = asyncio.run(
            api.api_security(
                self._request("/v1/speak", method="POST"),
                self._passthrough_response,
            )
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

    def test_webui_snapshot_uses_bound_celune_state(self) -> None:
        """Verify the browser snapshot mirrors current logs, status, and voice state."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )
        api.webui_log_lines.append(("Ready to speak.", "info"))
        api.set_webui_status("Idle")

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            (
                logs_html,
                status_html,
                resources_html,
                voice_update,
                send_update,
                input_update,
            ) = api.webui_snapshot()

        self.assertIn("Ready to speak.", logs_html)
        self.assertIn("--celune-ui-accent:", status_html)
        self.assertIn("style=", logs_html)
        self.assertIn("Idle", status_html)
        self.assertIn("10.66/11.94", resources_html)
        self.assertEqual(voice_update["value"], "Balanced")
        self.assertEqual(voice_update["interactive"], True)
        self.assertEqual(send_update["interactive"], True)
        self.assertEqual(input_update["interactive"], True)

    def test_webui_wrapped_callbacks_mirror_input_and_voice_lock_state(self) -> None:
        """Verify callback-driven lock state changes are reflected in the browser UI."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                log_callback=lambda msg, severity="info": None,
                status_callback=lambda msg, severity="info": None,
                voice_changed_callback=lambda name: None,
                change_input_state_callback=lambda locked: None,
                change_voice_lock_state_callback=lambda locked: None,
            ),
        )
        api.bind_celune(celune)

        celune.change_input_state_callback(True)
        celune.change_voice_lock_state_callback(True)
        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, _status, _resources, voice_update, send_update, input_update = (
                api.webui_snapshot()
            )
        self.assertEqual(input_update["interactive"], False)
        self.assertEqual(input_update["placeholder"], "Please wait")
        self.assertEqual(send_update["interactive"], False)
        self.assertEqual(voice_update["interactive"], False)

        celune.change_input_state_callback(False)
        celune.change_voice_lock_state_callback(False)
        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            (
                _logs2,
                _status2,
                _resources2,
                voice_update2,
                send_update2,
                input_update2,
            ) = api.webui_snapshot()
        self.assertEqual(input_update2["interactive"], True)
        self.assertEqual(
            input_update2["placeholder"], string("webui.input_placeholder")
        )
        self.assertEqual(send_update2["interactive"], True)
        self.assertEqual(voice_update2["interactive"], True)

    def test_webui_snapshot_shows_tutorial_placeholder(self) -> None:
        """Verify tutorial state uses the tutorial placeholder in the browser UI."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=True,
                locked=True,
                cur_state="idle",
            ),
        )
        api.webui_input_locked = True
        api.webui_input_placeholder = string("webui.tutorial_placeholder")
        api.webui_voice_locked = True

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, _status, _resources, voice_update, send_update, input_update = (
                api.webui_snapshot()
            )
        self.assertEqual(input_update["interactive"], False)
        self.assertEqual(
            input_update["placeholder"], string("webui.tutorial_placeholder")
        )
        self.assertEqual(send_update["interactive"], False)
        self.assertEqual(voice_update["interactive"], False)

    def test_webui_snapshot_uses_voice_changer_placeholder_in_vc_mode(self) -> None:
        """Verify VC mode uses the same voice-changer placeholder as the TUI."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                input_mode="voice_conversion",
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )
        api.webui_input_locked = False

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, _status, _resources, _voice, _send, input_update = (
                api.webui_snapshot()
            )

        self.assertEqual(
            input_update["placeholder"], string("webui.voice_changer_placeholder")
        )

    def test_webui_vc_controls_disable_outside_voice_conversion_mode(self) -> None:
        """Verify VC controls are disabled while Celune is in the normal TTS mode."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="text_to_speech",
            ),
        )

        source_update, pitch_update, mode_update, button_update = (
            api._webui_vc_controls_update()
        )

        self.assertEqual(source_update["interactive"], False)
        self.assertEqual(pitch_update["interactive"], False)
        self.assertEqual(mode_update["interactive"], False)
        self.assertEqual(button_update["interactive"], False)

    def test_webui_vc_controls_enable_in_voice_conversion_mode(self) -> None:
        """Verify VC controls become interactive when the engine is in VC mode."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
            ),
        )

        source_update, pitch_update, mode_update, button_update = (
            api._webui_vc_controls_update()
        )

        self.assertEqual(source_update["interactive"], True)
        self.assertEqual(pitch_update["interactive"], True)
        self.assertEqual(mode_update["interactive"], True)
        self.assertEqual(button_update["interactive"], True)

    def test_webui_snapshot_keeps_failed_no_voice_runtime_locked(self) -> None:
        """Verify a failed no-voice runtime stays in an error/locked browser state."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice=None,
                voices=(),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )
        api.webui_input_locked = False
        api.webui_input_placeholder = string("webui.input_placeholder")
        api.webui_voice_locked = False

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, status_html, _resources, voice_update, send_update, input_update = (
                api.webui_snapshot()
            )

        self.assertIn(f"{api.APP_NAME} could not start", status_html)
        self.assertEqual(voice_update["value"], "No voice set")
        self.assertEqual(voice_update["interactive"], False)
        self.assertEqual(send_update["interactive"], False)
        self.assertEqual(input_update["interactive"], False)

    def test_seeded_logs_strip_persisted_time_prefix(self) -> None:
        """Verify persisted log timestamps do not show up in the browser log view."""
        stripped = api.strip_webui_log_prefix(
            "[2026-06-11T14:22:01] [WARNING] Something happened"
        )
        self.assertEqual(stripped, "Something happened")

    def test_webui_theme_style_uses_cevoice_theme_metadata(self) -> None:
        """Verify browser CSS variables are derived from CEVOICE theme metadata."""
        loader = SimpleNamespace(
            bundle=SimpleNamespace(
                metadata={
                    "theme": {
                        "background": "#112233",
                        "accent": "#aabbcc",
                        "faded_accent": "#556677",
                    }
                }
            )
        )

        with mock.patch("celune.api.default_loader", return_value=loader):
            api.configure_webui_theme()

        self.assertIn("--celune-background: #112233;", api.webui_theme_style)
        self.assertIn("--celune-sleeping: #556677;", api.webui_theme_style)
        self.assertIn("--celune-primary:", api.webui_theme_style)
        self.assertIn("--celune-error:", api.webui_theme_style)
        self.assertIn("--celune-ui-accent:", api.webui_theme_style)
        self.assertIn("--celune-ui-bg:", api.webui_theme_style)
        self.assertIn('rel="icon"', api.WEBUI_HEAD)

    def test_webui_css_keeps_log_panel_flexible(self) -> None:
        """Verify the stylesheet keeps the log panel as the growable shell region."""
        self.assertIn("#celune-log-panel", api.WEBUI_CSS)
        self.assertIn("flex: 1 1 auto;", api.WEBUI_CSS)
        self.assertIn("min-height: 0;", api.WEBUI_CSS)
        self.assertIn('.standard-player input[type="range"]', api.WEBUI_CSS)
        self.assertIn(".minimal-audio-player button:hover", api.WEBUI_CSS)
        self.assertIn(".toast-body.error", api.WEBUI_CSS)
        self.assertIn(".toast-message-text.error::before", api.WEBUI_CSS)
        self.assertIn('content: "Celune is currently unavailable.";', api.WEBUI_CSS)
        self.assertIn(
            "@media (max-width: 768px), (any-pointer: coarse), (hover: none)",
            api.WEBUI_CSS,
        )
        self.assertNotIn("margin-top: auto;", api.WEBUI_CSS)

    def test_webui_audio_waveform_options_follow_primary_color(self) -> None:
        """Verify Gradio audio waveform colors are driven by Celune's primary color."""
        options = api._webui_audio_waveform_options()

        self.assertEqual(
            options.waveform_progress_color,
            api.colors.SEVERITY_COLORS["celune"]["info"],
        )
        self.assertEqual(
            options.trim_region_color,
            api.colors.SEVERITY_COLORS["celune"]["info"],
        )

    def test_webui_head_installs_log_autoscroll(self) -> None:
        """Verify the WebUI head installs auto-scroll behavior for the log pane."""
        self.assertIn("#celune-log-panel pre", api.WEBUI_HEAD)
        self.assertIn("MutationObserver", api.WEBUI_HEAD)
        self.assertIn("scrollTop = logElement.scrollHeight", api.WEBUI_HEAD)

    def test_webui_probe_logs_sleep_transition(self) -> None:
        """Verify the browser log mirrors the sleep transition message."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="sleeping",
                sleeping=True,
            ),
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            logs_html, status_html, _resources, _voice, _send, _input = (
                api.webui_snapshot()
            )

        self.assertIn("currently sleeping. Type anything to wake up.", logs_html)
        self.assertIn("Sleeping", status_html)

    def test_webui_probe_does_not_immediately_override_callback_status(self) -> None:
        """Verify fast callback statuses remain visible through the next probe."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="speaking",
            ),
        )
        api.webui_last_probed_state = "idle"
        api.set_webui_status(
            "Normalizing",
            source="callback",
            updated_at=10.0,
        )

        with (
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: 10.66/11.94 GB available",),
            ),
            mock.patch("celune.api.time.monotonic", return_value=10.1),
        ):
            _logs, status_html, _resources, _voice, _send, _input = api.webui_snapshot()

        self.assertIn("Normalizing", status_html)

    def test_webui_probe_prefers_active_playback_status(self) -> None:
        """Verify active playback status remains visible over generic speaking state."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="speaking",
                _playback_source_statuses={1: "Playing fixture.wav"},
            ),
        )
        api.webui_last_probed_state = "idle"
        api.set_webui_status("Speaking", source="callback", updated_at=10.0)

        with (
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: 10.66/11.94 GB available",),
            ),
            mock.patch("celune.api.time.monotonic", return_value=10.1),
        ):
            _logs, status_html, _resources, _voice, _send, _input = api.webui_snapshot()

        self.assertIn("Playing fixture.wav", status_html)

    def test_webui_probe_reconciles_stale_speaking_status_after_sleep(self) -> None:
        """Verify sleeping runtime state overrides a late speaking callback."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="sleeping",
                sleeping=True,
            ),
        )
        api.webui_last_probed_state = "sleeping"
        api.set_webui_status("Speaking", source="callback")

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            logs_html, status_html, _resources, _voice, _send, _input = (
                api.webui_snapshot()
            )

        self.assertIn("currently sleeping. Type anything to wake up.", logs_html)
        self.assertIn("Sleeping", status_html)

    def test_webui_slash_command_uses_main_ui_command_path(self) -> None:
        """Verify slash commands are delegated into the main UI command handler."""
        ui = SimpleNamespace()
        ui.process_command = mock.Mock()
        ui.call_from_thread = mock.Mock(side_effect=lambda fn, *args: fn(*args))

        with mock.patch("celune.api.CeluneUI._instance", ui):
            updates = list(api.webui_speak("/help"))

        ui.process_command.assert_called_once_with("help", [])
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][0]["value"], "")
        self.assertIsNone(updates[0][1])

    def test_webui_slash_command_warns_without_main_ui(self) -> None:
        """Verify slash commands warn instead of speaking when no main UI exists."""
        with mock.patch("celune.api.CeluneUI._instance", None):
            updates = list(api.webui_speak("/help"))

        self.assertEqual(len(updates), 1)
        self.assertIn("must be running to run commands", updates[0][2])

    def test_webui_speak_returns_browser_audio_after_generation(self) -> None:
        """Verify the browser submit handler returns one browser audio payload."""
        chunks: SpeechStreamQueue = Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)

        def say_stream(_content: str, save: bool = True) -> SpeechStreamQueue:
            _ = save
            return chunks

        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                say_stream=say_stream,
                dev=False,
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            updates = list(api.webui_speak("hello"))

        self.assertGreaterEqual(len(updates), 2)
        first_input, first_audio, *_first_rest = updates[0]
        second_input, second_audio, *_second_rest = updates[1]

        self.assertEqual(first_input["value"], "")
        self.assertIsNone(first_audio)
        self.assertEqual(second_input["value"], "")
        self.assertIsInstance(second_audio, tuple)
        sample_rate, array = cast(tuple[int, np.ndarray], second_audio)
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(array.shape, (8, 2))

    def test_webui_speak_wakes_sleeping_celune_before_speaking(self) -> None:
        """Verify browser submit wakes Celune first, then continues into speech."""
        calls: list[str] = []
        chunks: SpeechStreamQueue = Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        celune = SimpleNamespace(
            dev=False,
            current_voice="balanced",
            voices=("balanced", "calm"),
            is_in_tutorial=False,
            locked=False,
            cur_state="sleeping",
            sleeping=True,
        )

        async def wake_from_sleep_async() -> bool:
            calls.append("wake")
            celune.sleeping = False
            celune.cur_state = "idle"
            return True

        def say_stream(content: str, save: bool = True) -> SpeechStreamQueue:
            _ = save
            calls.append(f"say:{content}")
            return chunks

        celune.wake_from_sleep_async = wake_from_sleep_async
        celune.say_stream = say_stream
        api.bound_celune = cast(Celune, celune)

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            updates = list(api.webui_speak("wake me"))

        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(calls, ["wake", "say:wake me"])
        self.assertEqual(updates[0][0]["value"], "wake me")
        self.assertIsNone(updates[0][1])
        self.assertEqual(updates[-1][0]["value"], "")
        final_audio = updates[-1][1]
        self.assertIsInstance(final_audio, tuple)
        self.assertEqual(cast(tuple[int, np.ndarray], final_audio)[0], 48000)

    def test_webui_cycle_voice_uses_async_runtime_switch(self) -> None:
        """Verify browser voice cycling goes through the async runtime method."""
        calls: list[str] = []

        async def set_voice_async(name: str) -> bool:
            calls.append(name)
            return True

        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                set_voice_async=set_voice_async,
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _snapshot = api._webui_cycle_voice()

        self.assertEqual(calls, ["calm"])

    def test_webui_convert_audio_returns_browser_audio_after_conversion(self) -> None:
        """Verify browser audio conversion returns one playable Celune-format payload."""
        converted_audio = np.ones((10, 2), dtype=np.float32) * 0.5
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                convert_audio=mock.Mock(
                    return_value=SimpleNamespace(
                        audio=converted_audio,
                        sample_rate=24000,
                        label="browser audio input",
                    )
                ),
                dev=False,
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            source_value, browser_audio, *_rest = api._webui_convert_audio(
                (44100, np.zeros((16, 2), dtype=np.float32)),
                6.0,
                "sing",
            )

        self.assertIsNone(source_value)
        self.assertIsInstance(browser_audio, tuple)
        sample_rate, array = cast(tuple[int, np.ndarray], browser_audio)
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(array.shape, (20, 2))
        self.assertEqual(browser_audio[0], 48000)
        celune_convert_audio = cast(mock.Mock, api.bound_celune.convert_audio)
        celune_convert_audio.assert_called_once_with(
            mock.ANY,
            44100,
            label="browser audio input",
            pitch_shift=6,
            f0_condition=True,
        )

    def test_webui_convert_audio_normalizes_integer_browser_audio(self) -> None:
        """Verify browser-side PCM arrays are normalized before VC conversion."""
        captured_audio: dict[str, np.ndarray] = {}
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, **_kwargs: (
                        captured_audio.setdefault("audio", np.asarray(audio)),
                        SimpleNamespace(
                            audio=np.zeros((4, 2), dtype=np.float32),
                            sample_rate=sample_rate,
                            label="browser audio input",
                        ),
                    )[1]
                ),
                dev=False,
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )

        source_pcm = np.array(
            [[32767, -32768], [16384, -16384]],
            dtype=np.int16,
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _source_value, browser_audio, *_rest = api._webui_convert_audio(
                (44100, source_pcm)
            )

        self.assertIsInstance(browser_audio, tuple)
        normalized = captured_audio["audio"]
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.max(np.abs(normalized)) <= 1.0)
        self.assertAlmostEqual(float(normalized[0, 0]), 32767 / 32768, places=5)
        self.assertEqual(float(normalized[0, 1]), -1.0)

    def test_webui_convert_audio_rejects_text_to_speech_mode(self) -> None:
        """Verify browser audio conversion is unavailable outside VC mode."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="text_to_speech",
                dev=False,
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            source_value, browser_audio, logs_html, *_rest = api._webui_convert_audio(
                (44100, np.zeros((8, 2), dtype=np.float32))
            )

        self.assertIsNotNone(source_value)
        self.assertEqual(source_value[0], 44100)
        self.assertIsNone(browser_audio)
        self.assertIn(string("webui.conversion_only_in_vc_mode"), logs_html)

    def test_build_webui_exposes_tts_and_vc_tabs(self) -> None:
        """Verify the browser UI separates TTS and VC into distinct tabs."""
        demo = api._build_webui()
        config = demo.config
        tab_labels = [
            component.get("label") or component.get("props", {}).get("label")
            for component in config.get("components", [])
            if component.get("type") == "tabitem"
        ]

        self.assertEqual(
            tab_labels,
            [string("webui.tts_tab_label"), string("webui.vc_tab_label")],
        )

    def test_webui_snapshot_probes_runtime_status_and_rotates_resources(self) -> None:
        """Verify footer polling refreshes status and rotates the resource page."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="bold",
                voices=("bold", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="speaking",
            ),
        )

        with (
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: first", "Friday, June 11, 2026"),
            ),
            mock.patch("celune.api.time.monotonic", side_effect=[10.0, 12.2]),
        ):
            _logs1, status1, resources1, _voice1, _send1, _input1 = api.webui_snapshot()
            _logs2, status2, resources2, _voice2, _send2, _input2 = api.webui_snapshot()

        self.assertIn("Speaking", status1)
        self.assertIn("Speaking", status2)
        self.assertIn("VRAM: first", resources1)
        self.assertIn("Friday, June 11, 2026", resources2)

    def test_webui_runtime_theme_keeps_normal_palette_for_error_status(self) -> None:
        """Verify browser error statuses no longer switch the full UI palette."""
        api.set_webui_status("I can't speak right now.", "error")

        theme_html = api.webui_theme_html()

        self.assertIn(
            api.colors.THEME.background or api.colors.DEFAULT_BACKGROUND, theme_html
        )
        self.assertNotIn(api.colors.ERROR_BACKGROUND, theme_html)

    def test_webui_nonfatal_error_status_keeps_normal_theme(self) -> None:
        """Verify non-fatal browser errors do not switch the UI into the fatal palette."""
        api.set_webui_status("I can't change my voice right now.", "error")

        theme_html = api.webui_theme_html()

        self.assertIn(
            api.colors.THEME.background or api.colors.DEFAULT_BACKGROUND, theme_html
        )
        self.assertNotIn(api.colors.ERROR_BACKGROUND, theme_html)

    def test_webui_wrapped_fatal_glow_requests_api_shutdown(self) -> None:
        """Verify fatal glow stops the API/WebUI surface instead of tinting it red."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                log_callback=lambda msg, severity="info": None,
                status_callback=lambda msg, severity="info": None,
                voice_changed_callback=lambda name: None,
                change_input_state_callback=lambda locked: None,
                change_voice_lock_state_callback=lambda locked: None,
                glow=SimpleNamespace(fatal=mock.Mock()),
            ),
        )
        api.bound_celune = celune
        api.current_api_server = cast(
            api.StartedServer,
            SimpleNamespace(should_exit=False, force_exit=False),
        )

        api.wrap_celune_callbacks(celune)
        celune.glow.fatal()

        self.assertIsNone(api.bound_celune)
        self.assertTrue(api.current_api_server.should_exit)
        self.assertTrue(api.current_api_server.force_exit)

    @staticmethod
    def test_start_api_reports_when_port_is_already_in_use() -> None:
        """Verify occupied API ports produce a direct warning instead of a runtime error."""
        celune = SimpleNamespace(log=mock.Mock(), dev=False)
        bind_error = OSError(10048, "only one usage of each socket address")

        with mock.patch("celune.api.run_api", side_effect=bind_error):
            thread = api.start_api(cast(Celune, celune), port=2060)
            thread.join(timeout=1.0)

        celune.log.assert_called_once_with(
            string("api.port_in_use", port=2060),
            "warning",
        )
