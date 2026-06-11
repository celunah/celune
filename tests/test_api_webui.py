# SPDX-License-Identifier: MIT
"""Tests for Celune's browser-facing API UI."""

from types import SimpleNamespace
from queue import Queue
from typing import cast
from unittest import TestCase, mock

import numpy as np
from starlette.requests import Request

from celune import api
from celune.celune import Celune
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
        self.previous_logs = list(api.webui_log_lines)
        api.bound_celune = None
        api.webui_log_lines.clear()
        api.webui_logs_seeded = True
        api.webui_resource_page = 0
        api.webui_last_resource_advance = 0.0
        api.webui_last_probed_state = None
        api.webui_input_locked = True
        api.webui_input_placeholder = "Waiting for Celune to finish loading"
        api.webui_voice_locked = True
        api.webui_status_source = "probe"
        api.webui_status_updated_at = 0.0
        api._set_webui_status("Starting up")

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
        api.webui_log_lines.clear()
        api.webui_log_lines.extend(self.previous_logs)

    def test_root_redirects_to_browser_ui(self) -> None:
        """Verify the fallback root now forwards users to the browser UI."""
        response = api.root()
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/ui")

    def test_browser_ui_requests_bypass_api_security_detection(self) -> None:
        """Verify mounted browser UI paths are recognized by the API middleware."""
        ui_request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/ui/assets/index.js",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 2060),
                "scheme": "http",
                "server": ("127.0.0.1", 2060),
            }
        )
        api_request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/v1/version",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 2060),
                "scheme": "http",
                "server": ("127.0.0.1", 2060),
            }
        )

        self.assertEqual(api._is_browser_ui_request(ui_request), True)
        self.assertEqual(api._is_browser_ui_request(api_request), False)

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
        api._set_webui_status("Idle")

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
            ) = api._webui_snapshot()

        self.assertIn("Ready to speak.", logs_html)
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
                api._webui_snapshot()
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
            ) = api._webui_snapshot()
        self.assertEqual(input_update2["interactive"], True)
        self.assertEqual(input_update2["placeholder"], "Enter text to speak here")
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
        api.webui_input_placeholder = "Currently in tutorial mode"
        api.webui_voice_locked = True

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            _logs, _status, _resources, voice_update, send_update, input_update = (
                api._webui_snapshot()
            )
        self.assertEqual(input_update["interactive"], False)
        self.assertEqual(input_update["placeholder"], "Currently in tutorial mode")
        self.assertEqual(send_update["interactive"], False)
        self.assertEqual(voice_update["interactive"], False)

    def test_seeded_logs_strip_persisted_time_prefix(self) -> None:
        """Verify persisted log timestamps do not show up in the browser log view."""
        stripped = api._strip_webui_log_prefix(
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
            api._configure_webui_theme()

        self.assertIn("--celune-background: #112233;", api.webui_theme_style)
        self.assertIn("--celune-sleeping: #556677;", api.webui_theme_style)
        self.assertIn("--celune-primary:", api.webui_theme_style)
        self.assertIn('rel="icon"', api.WEBUI_HEAD)

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
                api._webui_snapshot()
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

        with (
            mock.patch(
                "celune.api.ui_resources.resource_pages",
                return_value=("VRAM: 10.66/11.94 GB available",),
            ),
            mock.patch("celune.api.time.monotonic", side_effect=[10.0, 10.1]),
        ):
            api._set_webui_status("Normalizing", source="callback")
            _logs, status_html, _resources, _voice, _send, _input = (
                api._webui_snapshot()
            )

        self.assertIn("Normalizing", status_html)

    def test_webui_slash_command_uses_main_ui_command_path(self) -> None:
        """Verify slash commands are delegated into the main UI command handler."""
        ui = SimpleNamespace()
        ui.process_command = mock.Mock()
        ui.call_from_thread = mock.Mock(side_effect=lambda fn, *args: fn(*args))

        with mock.patch("celune.api.CeluneUI._instance", ui):
            updates = list(api._webui_speak("/help"))

        ui.process_command.assert_called_once_with("help", [])
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][0]["value"], "")
        self.assertIsNone(updates[0][1])

    def test_webui_slash_command_warns_without_main_ui(self) -> None:
        """Verify slash commands warn instead of speaking when no main UI exists."""
        with mock.patch("celune.api.CeluneUI._instance", None):
            updates = list(api._webui_speak("/help"))

        self.assertEqual(len(updates), 1)
        self.assertIn("Slash commands require the main Celune window", updates[0][2])

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
            updates = list(api._webui_speak("hello"))

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

        def wake_from_sleep() -> bool:
            calls.append("wake")
            celune.sleeping = False
            celune.cur_state = "idle"
            return True

        def say_stream(content: str, save: bool = True) -> SpeechStreamQueue:
            _ = save
            calls.append(f"say:{content}")
            return chunks

        celune.wake_from_sleep = wake_from_sleep
        celune.say_stream = say_stream
        api.bound_celune = cast(Celune, celune)

        with mock.patch(
            "celune.api.ui_resources.resource_pages",
            return_value=("VRAM: 10.66/11.94 GB available",),
        ):
            updates = list(api._webui_speak("wake me"))

        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(calls, ["wake", "say:wake me"])
        self.assertEqual(updates[0][0]["value"], "wake me")
        self.assertIsNone(updates[0][1])
        self.assertEqual(updates[-1][0]["value"], "")
        final_audio = updates[-1][1]
        self.assertIsInstance(final_audio, tuple)
        self.assertEqual(cast(tuple[int, np.ndarray], final_audio)[0], 48000)

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
            _logs1, status1, resources1, _voice1, _send1, _input1 = (
                api._webui_snapshot()
            )
            _logs2, status2, resources2, _voice2, _send2, _input2 = (
                api._webui_snapshot()
            )

        self.assertIn("Speaking", status1)
        self.assertIn("Speaking", status2)
        self.assertIn("VRAM: first", resources1)
        self.assertIn("Friday, June 11, 2026", resources2)
