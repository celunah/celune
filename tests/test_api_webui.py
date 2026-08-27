# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's browser-facing API UI."""

import asyncio
from queue import Queue
from types import SimpleNamespace
from typing import cast
from unittest import mock

import numpy as np
from starlette.requests import Request
from starlette.responses import Response

from celune import api
from celune.i18n import string
from celune.celune import Celune
from celune.pipeline import SpeechStreamQueue

from .support import CeluneTestCase


class TestApiWebUI(CeluneTestCase):
    """Tests for the mounted Gradio browser UI helpers."""

    def setUp(self) -> None:
        """Snapshot global WebUI state before each test."""
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
        self.previous_caption_text = api.webui_caption_text
        self.previous_caption_progress = api.webui_caption_progress
        self.previous_caption_active = api.webui_caption_active
        self.previous_progress_current = api.webui_progress_current
        self.previous_progress_total = api.webui_progress_total
        self.previous_active_theme_name = api.webui_active_theme_name
        self.previous_timed_update_sequence = api.webui_timed_update_sequence
        self.previous_timed_update_received_at = api.webui_timed_update_received_at
        self.previous_timed_update_source = api.webui_timed_update_source
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
        api.webui_caption_text = ""
        api.webui_caption_progress = 0.0
        api.webui_caption_active = False
        api.webui_progress_current = None
        api.webui_progress_total = None
        api.webui_active_theme_name = "celune"
        api.webui_timed_update_sequence = 0
        api.webui_timed_update_received_at = 0.0
        api.webui_timed_update_source = "fallback"
        api.set_webui_status("Starting up")

    def tearDown(self) -> None:
        """Restore global WebUI state after each test."""
        api._unsubscribe_webui_events()
        if api.webui_timed_update_unsubscribe is not None:
            api.webui_timed_update_unsubscribe()
            api.webui_timed_update_unsubscribe = None
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
        api.webui_caption_text = self.previous_caption_text
        api.webui_caption_progress = self.previous_caption_progress
        api.webui_caption_active = self.previous_caption_active
        api.webui_progress_current = self.previous_progress_current
        api.webui_progress_total = self.previous_progress_total
        api.webui_active_theme_name = self.previous_active_theme_name
        api.webui_timed_update_sequence = self.previous_timed_update_sequence
        api.webui_timed_update_received_at = self.previous_timed_update_received_at
        api.webui_timed_update_source = self.previous_timed_update_source
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
        assert response.status_code == 307
        assert response.headers["location"] == "/ui"

    def test_browser_ui_requests_bypass_api_security_detection(self) -> None:
        """Verify mounted browser UI paths are recognized by the API middleware."""
        ui_request = self._request("/ui/assets/index.js")
        api_request = self._request("/v1/version")

        assert api.is_browser_ui_request(ui_request)
        assert not api.is_browser_ui_request(api_request)

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
            assert response.status_code == 200, path

    def test_api_security_requires_token_for_generating_routes(self) -> None:
        """Verify protected API routes still reject unauthenticated requests."""
        api.auth_token = "secret"

        response = asyncio.run(
            api.api_security(
                self._request("/v1/speak", method="POST"),
                self._passthrough_response,
            )
        )

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

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

        assert "Ready to speak." in logs_html
        assert "--celune-ui-accent:" in status_html
        assert "style=" in logs_html
        assert "Idle" in status_html
        assert "10.66/11.94" in resources_html
        assert voice_update["value"] == "Balanced"
        assert voice_update["interactive"]
        assert send_update["interactive"]
        assert input_update["interactive"]

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
        assert not input_update["interactive"]
        assert input_update["placeholder"] == "Please wait"
        assert not send_update["interactive"]
        assert not voice_update["interactive"]

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
        assert input_update2["interactive"]
        assert input_update2["placeholder"] == string("ui.input_placeholder")
        assert send_update2["interactive"]
        assert voice_update2["interactive"]

    def test_webui_wrapped_callbacks_preserve_legacy_log_signatures(self) -> None:
        """Verify WebUI callback wrapping still supports two-argument handlers."""
        log_calls: list[tuple[str, str]] = []
        status_calls: list[tuple[str, str]] = []

        def legacy_log(message: str, severity: str = "info") -> None:
            log_calls.append((message, severity))

        def legacy_status(message: str, severity: str = "info") -> None:
            status_calls.append((message, severity))

        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced",),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                log_callback=legacy_log,
                status_callback=legacy_status,
                voice_changed_callback=lambda _name: None,
                change_input_state_callback=lambda _locked: None,
                change_voice_lock_state_callback=lambda _locked: None,
            ),
        )
        api.bind_celune(celune)

        celune.log_callback("legacy log", "warning", loglevel="debug")
        celune.status_callback("legacy status", "info", loglevel="verbose")

        assert log_calls == [("legacy log", "warning")]
        assert status_calls == [("legacy status", "info")]

    def test_webui_wrapped_callbacks_mirror_generation_lifecycle(self) -> None:
        """Verify idle, queue, caption, progress, and error callbacks reach WebUI state."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=True,
                cur_state="speaking",
                log_callback=lambda *_args, **_kwargs: None,
                status_callback=lambda *_args, **_kwargs: None,
                error_callback=lambda _message: None,
                idle_callback=lambda: None,
                queue_avail_callback=lambda: None,
                progress_callback=lambda *_args: None,
                caption_progress_callback=lambda *_args: None,
                caption_callback=lambda _caption: None,
                caption_timing_callback=lambda *_args: None,
                voice_changed_callback=lambda _name: None,
                change_input_state_callback=lambda _locked: None,
                change_voice_lock_state_callback=lambda _locked: None,
            ),
        )
        with mock.patch(
            "celune.api.main_window_log_path",
            return_value=mock.Mock(exists=lambda: False),
        ):
            api.bind_celune(celune)

        celune.queue_avail_callback()
        assert not api.webui_input_locked
        assert api.webui_status_text == string("status.speaking")

        celune.caption_callback("Hello there")
        celune.caption_progress_callback(1.0, 2.0)
        api.webui_progress_current = 1.0
        api.webui_progress_total = 1.0
        _status = api._webui_status_html()
        assert "Hello there" in _status
        assert "50%" in _status
        assert "100%" not in _status

        celune.error_callback("backend failed")
        assert "backend failed" in api._webui_status_html()
        assert any(
            message == "backend failed" and severity == "error"
            for message, severity in api.webui_log_lines
        )

        celune.idle_callback()
        assert api.webui_caption_text == ""
        assert api.webui_progress_total is None

    def test_webui_persona_input_uses_think_path(self) -> None:
        """Verify browser input follows the Persona path when talkback is enabled."""
        chunks = Queue()
        think = mock.Mock(return_value=True)
        say_stream = mock.Mock(return_value=chunks)
        celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                current_voice="balanced",
                voices=("balanced", "calm"),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                sleeping=False,
                think=think,
                say_stream=say_stream,
            ),
        )
        api.bound_celune = celune
        with (
            mock.patch("celune.api.persona_talkback_enabled", return_value=True),
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: first",),
            ),
        ):
            list(api.webui_speak("hello"))

        think.assert_called_once_with("hello")
        say_stream.assert_not_called()

    def test_webui_slash_command_uses_shared_handler_without_tui(self) -> None:
        """Verify slash commands do not require a mounted Textual singleton."""
        celune = cast(Celune, SimpleNamespace())
        api.bound_celune = celune
        with (
            mock.patch("celune.api.CeluneUI._instance", None),
            mock.patch("celune.ui.commands.process_command") as process_command,
        ):
            assert api._webui_run_command("/help")

        process_command.assert_called_once()
        assert process_command.call_args.args[1:] == ("help", [])

    def test_webui_timed_updates_follow_cedts_sequence(self) -> None:
        """Verify newer TUI timed updates drive browser resource and theme state."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced",),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )
        api.bound_celune = celune
        update = api.UiTimedUpdate(
            runtime_id=str(id(celune)),
            sequence=4,
            emitted_at=1.0,
            resource_page=3,
            theme_name="celune_light",
            status_text="Status from TUI",
            status_severity="info",
            status_marquee_offset=2,
        )
        api._receive_webui_timed_update(update)
        assert api.webui_resource_page == 3
        assert api.webui_active_theme_name == "celune_light"
        assert api.webui_timed_update_sequence == 4
        assert api.webui_status_text == "Status from TUI"
        assert api.webui_timed_update_source == "cedts"

    def test_webui_seeded_multiline_log_retains_record_severity(self, tmp_path) -> None:
        """Verify continuation lines inherit the persisted record severity."""
        log_path = tmp_path / "celune.log"
        log_path.write_text(
            "[2026-08-25T12:00:00] [WARNING] first line\nsecond line\n"
            "[2026-08-25T12:00:01] [INFO] next line\n",
            encoding="utf-8",
        )
        api.webui_logs_seeded = False
        with mock.patch("celune.api.main_window_log_path", return_value=log_path):
            api._seed_webui_logs()

        assert list(api.webui_log_lines) == [
            ("first line\nsecond line", "warning"),
            ("next line", "info"),
        ]

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
        api.webui_input_placeholder = string("ui.tutorial_placeholder")
        api.webui_voice_locked = True

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, _status, _resources, voice_update, send_update, input_update = (
                api.webui_snapshot()
            )
        assert not input_update["interactive"]
        assert input_update["placeholder"] == string("ui.tutorial_placeholder")
        assert not send_update["interactive"]
        assert not voice_update["interactive"]

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

        assert input_update["placeholder"] == string("ui.voice_changer_placeholder")

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

        assert not source_update["interactive"]
        assert not pitch_update["interactive"]
        assert not mode_update["interactive"]
        assert not button_update["interactive"]

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

        assert source_update["interactive"]
        assert pitch_update["interactive"]
        assert mode_update["interactive"]
        assert button_update["interactive"]

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
        api.webui_input_placeholder = string("ui.input_placeholder")
        api.webui_voice_locked = False

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, status_html, _resources, voice_update, send_update, input_update = (
                api.webui_snapshot()
            )

        assert f"{api.APP_NAME} could not start" in status_html
        assert voice_update["value"] == "No voice set"
        assert not voice_update["interactive"]
        assert not send_update["interactive"]
        assert not input_update["interactive"]

    def test_webui_vc_state_uses_core_predicate_and_disables_text_input(self) -> None:
        """Verify a backend-reported VC state is authoritative in the browser."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced", "calm"),
                input_mode="text_to_speech",
                is_voice_conversion_mode=mock.Mock(return_value=True),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
            ),
        )
        api.bound_celune = celune
        api.webui_input_locked = False

        input_update = api._input_update()
        send_update = api._send_button_update()

        assert not input_update["interactive"]
        assert not send_update["interactive"]

    def test_seeded_logs_strip_persisted_time_prefix(self) -> None:
        """Verify persisted log timestamps do not show up in the browser log view."""
        stripped = api.strip_webui_log_prefix(
            "[2026-06-11T14:22:01] [WARNING] Something happened"
        )
        assert stripped == "Something happened"

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

        assert "--celune-background: #112233;" in api.webui_theme_style
        assert "--celune-sleeping: #556677;" in api.webui_theme_style
        assert "--celune-primary:" in api.webui_theme_style
        assert "--celune-error:" in api.webui_theme_style
        assert "--celune-ui-accent:" in api.webui_theme_style
        assert "--celune-ui-bg:" in api.webui_theme_style
        assert 'rel="icon"' in api.WEBUI_HEAD

    def test_webui_css_keeps_log_panel_flexible(self) -> None:
        """Verify the stylesheet keeps the log panel as the growable shell region."""
        assert "#celune-log-panel" in api.WEBUI_CSS
        assert "flex: 1 1 auto;" in api.WEBUI_CSS
        assert "min-height: 0;" in api.WEBUI_CSS
        assert '.standard-player input[type="range"]' in api.WEBUI_CSS
        assert ".minimal-audio-player button:hover" in api.WEBUI_CSS
        assert ".toast-body.error" in api.WEBUI_CSS
        assert ".toast-message-text.error::before" in api.WEBUI_CSS
        assert 'content: "Celune is currently unavailable.";' in api.WEBUI_CSS
        assert (
            "@media (max-width: 768px), (any-pointer: coarse), (hover: none)"
            in api.WEBUI_CSS
        )
        assert "margin-top: auto;" not in api.WEBUI_CSS

    def test_webui_audio_waveform_options_follow_primary_color(self) -> None:
        """Verify Gradio audio waveform colors are driven by Celune's primary color."""
        options = api._webui_audio_waveform_options()

        assert (
            options.waveform_progress_color
            == api.colors.SEVERITY_COLORS["celune"]["info"]
        )
        assert options.trim_region_color == api.colors.SEVERITY_COLORS["celune"]["info"]

    def test_webui_head_installs_log_autoscroll(self) -> None:
        """Verify the WebUI head installs auto-scroll behavior for the log pane."""
        assert "#celune-log-panel pre" in api.WEBUI_HEAD
        assert "MutationObserver" in api.WEBUI_HEAD
        assert "isNearLogBottom" in api.WEBUI_HEAD
        assert "__celuneLogAutoscrollFollow" in api.WEBUI_HEAD
        assert "scrollTop = logElement.scrollHeight" in api.WEBUI_HEAD

    def test_webui_log_buffer_deduplicates_adjacent_callback_entries(self) -> None:
        """Verify overlapping log forwarding cannot print one WebUI line twice."""
        api._append_webui_log("same line", "warning")
        api._append_webui_log("same line", "warning")

        assert list(api.webui_log_lines) == [("same line", "warning")]

    def test_webui_persona_placeholder_matches_tui_logic(self) -> None:
        """Verify Persona talkback uses the main TUI's input placeholder."""
        celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced",),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                persona_ready=True,
                config={},
            ),
        )

        with (
            mock.patch("celune.api.persona_enabled", return_value=True),
            mock.patch("celune.api.persona_talkback_enabled", return_value=True),
        ):
            placeholder = api._webui_input_placeholder(celune, False, True)

        assert placeholder == string("ui.say_placeholder")

    def test_webui_recording_hint_uses_the_existing_resource_snapshot(self) -> None:
        """Verify the ALT+R hint is rendered with the timed footer update."""
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                current_voice="balanced",
                voices=("balanced",),
                is_in_tutorial=False,
                locked=False,
                cur_state="idle",
                persona_ready=True,
                config={},
            ),
        )
        api.webui_input_locked = False

        with (
            mock.patch.object(api.CeluneUI, "_instance", object()),
            mock.patch("celune.api.persona_enabled", return_value=True),
            mock.patch("celune.api.persona_talkback_enabled", return_value=True),
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: available",),
            ),
        ):
            resources_html = api._webui_resources_html()

        assert string("webui.recording_voice_hint") in resources_html

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

        assert "currently sleeping. Type anything to wake up." in logs_html
        assert "Sleeping" in status_html

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

        assert "Normalizing" in status_html

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

        assert "Playing fixture.wav" in status_html

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

        assert "currently sleeping. Type anything to wake up." in logs_html
        assert "Sleeping" in status_html

    def test_webui_slash_command_uses_main_ui_command_path(self) -> None:
        """Verify slash commands are delegated into the main UI command handler."""
        ui = SimpleNamespace()
        ui.process_command = mock.Mock()
        ui.call_from_thread = mock.Mock(side_effect=lambda fn, *args: fn(*args))

        with mock.patch("celune.api.CeluneUI._instance", ui):
            updates = list(api.webui_speak("/help"))

        ui.process_command.assert_called_once_with("help", [])
        assert len(updates) == 1
        assert updates[0][0]["value"] == ""
        assert updates[0][1] is None

    def test_webui_slash_command_reports_missing_core(self) -> None:
        """Verify slash commands report a missing core instead of requiring TUI."""
        with mock.patch("celune.api.CeluneUI._instance", None):
            updates = list(api.webui_speak("/help"))

        assert len(updates) == 1
        assert "not currently available" in updates[0][2]

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

        assert len(updates) >= 2
        first_input, first_audio, *_first_rest = updates[0]
        second_input, second_audio, *_second_rest = updates[1]

        assert first_input["value"] == ""
        assert first_audio is None
        assert second_input["value"] == ""
        assert isinstance(second_audio, tuple)
        sample_rate, array = cast(tuple[int, np.ndarray], second_audio)
        assert sample_rate == 48000
        assert array.shape == (8, 2)

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

        assert len(updates) >= 3
        assert calls == ["wake", "say:wake me"]
        assert updates[0][0]["value"] == "wake me"
        assert updates[0][1] is None
        assert updates[-1][0]["value"] == ""
        final_audio = updates[-1][1]
        assert isinstance(final_audio, tuple)
        assert cast(tuple[int, np.ndarray], final_audio)[0] == 48000

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

        assert calls == ["calm"]

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

        assert source_value is None
        assert isinstance(browser_audio, tuple)
        sample_rate, array = cast(tuple[int, np.ndarray], browser_audio)
        assert sample_rate == 48000
        assert array.shape == (20, 2)
        assert browser_audio[0] == 48000
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

        assert isinstance(browser_audio, tuple)
        normalized = captured_audio["audio"]
        assert normalized.dtype == np.float32
        assert np.max(np.abs(normalized)) <= 1.0
        assert round(abs(float(normalized[0, 0]) - 32767 / 32768), 5) == 0
        assert float(normalized[0, 1]) == -1.0

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

        assert source_value is not None
        assert source_value[0] == 44100
        assert browser_audio is None
        assert string("webui.conversion_only_in_vc_mode") in logs_html

    def test_build_webui_exposes_tts_and_vc_tabs(self) -> None:
        """Verify the browser UI separates TTS and VC into distinct tabs."""
        demo = api._build_webui()
        config = demo.config
        tab_labels = [
            component.get("label") or component.get("props", {}).get("label")
            for component in config.get("components", [])
            if component.get("type") == "tabitem"
        ]

        assert tab_labels == [
            string("webui.tts_tab_label"),
            string("webui.vc_tab_label"),
        ]

        component_ids = {
            component.get("props", {}).get("elem_id")
            for component in config.get("components", [])
        }
        assert "celune-voice-menu" not in component_ids
        assert "celune-style" in component_ids
        assert "celune-record" not in component_ids
        assert "celune-stop" not in component_ids
        assert "celune-settings" not in component_ids
        assert "celune-record-hotkey" in component_ids
        assert "Usage may differ. Some Celune features may not be available." in str(
            config
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

        assert "Speaking" in status1
        assert "Speaking" in status2
        assert "VRAM: first" in resources1
        assert "Friday, June 11, 2026" in resources2

    def test_webui_runtime_theme_keeps_normal_palette_for_error_status(self) -> None:
        """Verify browser error statuses no longer switch the full UI palette."""
        api.set_webui_status("I can't speak right now.", "error")

        theme_html = api.webui_theme_html()

        assert (
            api.colors.THEME.background or api.colors.DEFAULT_BACKGROUND
        ) in theme_html
        assert api.colors.ERROR_BACKGROUND not in theme_html

    def test_webui_nonfatal_error_status_keeps_normal_theme(self) -> None:
        """Verify non-fatal browser errors do not switch the UI into the fatal palette."""
        api.set_webui_status("I can't change my voice right now.", "error")

        theme_html = api.webui_theme_html()

        assert (
            api.colors.THEME.background or api.colors.DEFAULT_BACKGROUND
        ) in theme_html
        assert api.colors.ERROR_BACKGROUND not in theme_html

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

        assert api.bound_celune is None
        assert api.current_api_server.should_exit
        assert api.current_api_server.force_exit

    @staticmethod
    def test_start_api_reports_when_port_is_already_in_use() -> None:
        """Verify occupied API ports produce a direct warning instead of a runtime error."""
        celune = SimpleNamespace(log=mock.Mock(), dev=False)
        bind_error = OSError(10048, "only one usage of each socket address")

        with mock.patch("celune.api.run_api", side_effect=bind_error):
            thread = api.start_api(cast(Celune, celune), port=2060)
            thread.join(timeout=1.0)

        celune.log.assert_called_once_with(
            "API port 2060 is already in use.",
            "warning",
        )
