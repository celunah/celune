# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime validation and lightweight UI commands."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import sys
import time
import queue
import asyncio
import tempfile
import warnings
import threading
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from typing import Optional, cast
from collections.abc import Callable

import pytest
import numpy as np
from textual import events
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Label, Static, RichLog, TextArea, ProgressBar

from celune.ui.app import (
    Button,
    CeluneUI,
    VoiceButton,
    UILogMessage,
    ButtonActions,
    ProgressLabel,
    CeluneLoadingScreen,
)
from celune.theme import colors
from celune.celune import Celune
from celune.utils import discard
from celune.ui import app as ui_app
from celune.i18n import string, tagged_string
from celune.ui import terminal as ui_terminal
from celune.ui import resources as ui_resources
from celune.ui.headless import CeluneHeadlessUI
from tests.support import FakeBackend, CeluneTestCase
from celune.constants import APP_NAME, COST_EQUIVALENTS, ExitCodes


class TestUIStartup(CeluneTestCase):
    """Tests for UI startup guard rails."""

    def tearDown(self) -> None:
        """Reset singleton UI guards after each test."""
        CeluneUI._instance = None
        CeluneHeadlessUI._instance = None

    def test_deferred_startup_catches_system_exit_in_the_ui_worker(self) -> None:
        """Verify startup worker exits become UI-handled errors."""
        ui = CeluneUI(
            startup_loader=lambda: cast(Celune, (_ for _ in ()).throw(SystemExit(4)))
        )
        ui.call_from_thread = mock.Mock()

        ui._load_deferred_runtime()

        ui.call_from_thread.assert_called_once()
        callback, error = ui.call_from_thread.call_args.args
        assert callback.__self__ is ui
        assert getattr(callback, "__func__", None) is getattr(
            ui._handle_deferred_runtime_error,
            "__func__",
            None,
        )
        assert isinstance(error, SystemExit)
        assert error.code == 4

    def test_linux_terminal_stream_does_not_need_lazy_discard_export(self) -> None:
        """Verify Linux terminal setup works before deferred UI imports complete."""
        ui = CeluneUI()
        output_stream = mock.Mock()
        fake_stderr = SimpleNamespace(
            fileno=lambda: 2,
            encoding="utf-8",
            errors="strict",
        )
        missing_discard = mock.sentinel.missing_discard
        saved_discard = ui_app.__dict__.pop("discard", missing_discard)

        try:
            with (
                mock.patch.object(sys, "__stderr__", fake_stderr),
                mock.patch("celune.ui.runtime.os.name", "posix"),
                mock.patch("celune.ui.runtime.os.dup", return_value=17) as dup,
                mock.patch(
                    "celune.ui.runtime.os.fdopen", return_value=output_stream
                ) as fdopen,
            ):
                assert ui._prepare_terminal_output_stream() is output_stream

            dup.assert_called_once_with(2)
            fdopen.assert_called_once_with(
                17,
                "w",
                encoding="utf-8",
                errors="strict",
                buffering=1,
            )
        finally:
            ui._terminal_output_stream = None
            if saved_discard is not missing_discard:
                ui_app.discard = saved_discard

    def test_missing_dependency_startup_waits_for_ctrl_q(self) -> None:
        """Verify missing dependencies remain visible until the user quits."""
        ui = CeluneUI()
        ui.refresh_css = mock.Mock()
        ui._show_loading_error = mock.Mock()
        ui._write_terminal_title = mock.Mock()
        missing = ModuleNotFoundError("No module named 'dateutil'")
        missing.name = "dateutil"

        ui._handle_deferred_runtime_error(missing)

        assert ui.cur_state == "error"
        assert ui._fatal_error_active
        assert ui._startup_error_exit_code == ExitCodes.EXIT_MISSING_DEPENDENCIES.value
        ui._write_terminal_title.assert_called_once_with(
            (
                APP_NAME,
                string("osc.state_error"),
                string("osc.action_missing_dependency"),
            )
        )
        ui._show_loading_error.assert_called_once_with(
            tagged_string("ui.init_error", "INIT ERROR"),
            status_message=string("status.early_initialization_failed"),
            footer_message=string("status.missing_dependency"),
        )

        ui._shutdown_runtime = mock.Mock()
        ui.exit = mock.Mock()
        asyncio.run(ui.action_quit())

        ui.exit.assert_called_once_with(
            return_code=ExitCodes.EXIT_MISSING_DEPENDENCIES.value
        )

    def test_loading_screen_stores_latest_log_before_mount(self) -> None:
        """Verify startup log text can arrive before loading widgets mount."""
        screen = CeluneLoadingScreen()

        screen.set_status_message("Loading backend")
        screen.set_latest_log_message("Backend initialized")
        screen.show_error("Backend failed")

        assert screen._status_message == string("status.failed_to_start")
        assert screen._latest_log_message == "Backend failed"
        assert screen._wait_message == string(
            "ui.loading_cannot_continue",
            app_name=APP_NAME,
        )
        assert screen._footer_message == string(
            "ui.app_could_not_start",
            app_name=APP_NAME,
        )

    def test_main_ui_selects_celune_theme_before_rendering_without_engine(self) -> None:
        """Verify the single UI app has its theme before the engine is attached."""
        ui = CeluneUI()
        ui.prepare_theme()

        self.assertIsNone(ui.celune)
        self.assertEqual(ui.theme, "celune")
        self.assertEqual(ui.current_theme.name, "celune")
        self.assertEqual(
            ui.theme_variables["background"].lower(), colors.THEME.background
        )

    def test_main_ui_prepares_the_lightweight_theme_before_runtime_imports(
        self,
    ) -> None:
        """Verify the loading frame uses Celune colors without the engine."""
        ui = CeluneUI()

        with mock.patch("celune.ui.app._load_ui_runtime_dependencies") as load_runtime:
            ui._prepare_loading_theme()

        load_runtime.assert_not_called()
        self.assertEqual(ui.theme, "celune")
        self.assertEqual(ui.current_theme.background, "#1d1826")
        self.assertEqual(ui.current_theme.primary, "#cebaff")

    def test_main_ui_prepares_configured_theme_before_rendering(self) -> None:
        """Verify the main UI applies the configured theme before ``run``."""
        ui = CeluneUI()
        ui.celune = cast(Celune, SimpleNamespace(config={"theme": "light"}))

        with mock.patch("celune.ui.app.default_loader", return_value=None):
            ui.prepare_theme()

        self.assertEqual(ui.theme, "celune_light")
        self.assertEqual(ui.current_theme.name, "celune_light")

    def test_settings_menu_flattens_nested_configuration_values(self) -> None:
        """Verify the configuration manager exposes nested YAML leaf values."""
        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={"backend": "mini", "persona": {"context_size": 8192}},
            ),
        )

        with mock.patch.object(ui, "_show_menu") as show_menu:
            ui.open_settings_menu()

        menu = show_menu.call_args.args[0]
        assert [option.label for option in menu.options] == [
            "Backend",
            "Persona context size",
        ]
        assert menu.options[0].explanation == (
            "Choose the speech backend Celune should use."
        )
        assert menu.return_value is False
        assert ui._settings_paths == (("backend",), ("persona", "context_size"))

    def test_settings_menu_saves_values_and_requests_pending_restart(self) -> None:
        """Verify ENTER's settings result writes YAML and returns exit code 7."""
        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={"backend": "mini", "persona": {"context_size": 8192}}
            ),
        )
        menu = ui_terminal.SelectMenuWidget(
            "Configuration manager",
            [
                ui_terminal.SelectMenuOption("backend", "qwen3"),
                ui_terminal.SelectMenuOption("persona.context_size", 16384),
            ],
            return_value=False,
        )
        ui._settings_paths = (("backend",), ("persona", "context_size"))
        ui.safe_log = mock.Mock()

        with tempfile.TemporaryDirectory() as temporary:
            config_file = Path(temporary) / "config.yaml"
            with (
                mock.patch.object(
                    ui_app,
                    "config_path",
                    return_value=config_file,
                    create=True,
                ),
                mock.patch.object(ui_app, "yaml", create=True) as yaml_module,
                mock.patch.object(ui, "_run_shutdown_step") as shutdown_step,
                mock.patch.object(ui, "_set_terminal_status") as set_status,
                mock.patch.object(ui, "_graceful_exit") as graceful_exit,
            ):
                ui._save_settings(menu)
                shutdown_step.call_args_list[0].args[0]()
                set_status.assert_called_once_with(
                    "restarting",
                    string("osc.action_restarting"),
                )

            yaml_module.safe_dump.assert_called_once()
            assert ui.celune.config["backend"] == "qwen3"
            assert ui.celune.config["persona"]["context_size"] == 16384
            assert ui.cur_state == "restarting"
            ui.safe_log.assert_not_called()
            shutdown_step.assert_called_once()
            graceful_exit.assert_called_once_with(return_code=7)

    def test_settings_restart_preserves_restart_state_through_graceful_exit(
        self,
    ) -> None:
        """Verify a settings restart uses the normal fade while retaining its exit code."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.cur_state = "restarting"
        ui.exit = mock.Mock()
        ui._shutdown_runtime = mock.Mock()
        screen = SimpleNamespace(
            styles=SimpleNamespace(opacity=1.0),
            refresh=mock.Mock(),
            query=mock.Mock(return_value=()),
        )
        fade_complete: list[Callable[[], None]] = []

        def capture_fade(*_args, on_complete=None, **_kwargs) -> None:
            if on_complete is not None:
                fade_complete.append(on_complete)

        with (
            mock.patch.object(
                CeluneUI,
                "screen",
                new_callable=mock.PropertyMock,
                return_value=screen,
            ),
            mock.patch.object(ui, "_animate_opacity", side_effect=capture_fade),
            mock.patch.object(
                ui,
                "call_after_refresh",
                side_effect=lambda callback: callback(),
            ),
        ):
            ui._graceful_exit(return_code=ExitCodes.EXIT_PENDING_RESTART.value)

            assert ui.cur_state == "restarting"
            assert len(fade_complete) == 1
            fade_complete[0]()

        ui._shutdown_runtime.assert_called_once_with()
        ui.exit.assert_called_once_with(
            return_code=ExitCodes.EXIT_PENDING_RESTART.value
        )

    def test_loading_screen_mounts_with_canonical_textual_css(self) -> None:
        """Verify the loading screen composes and updates in a real Textual app."""

        screen = CeluneLoadingScreen()

        class Harness(App):
            """Minimal Textual host for the loading screen smoke test."""

            CSS = ui_app.CELUNE_CSS

            def compose(self):
                """Compose the loading overlay as the root widget."""
                yield screen

        async def run_smoke_test() -> None:
            app = Harness()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                assert str(screen.query_one("#loading-brand", Static).render()) == (
                    APP_NAME
                )
                screen.set_latest_log_message("Backend initialized")
                self.assertEqual(
                    str(screen.query_one("#loading-log-message", Static).render()),
                    "Backend initialized",
                )
                screen.set_startup_messages(
                    [
                        string("ui.startup_checking_dependencies"),
                        string("ui.startup_loading_core"),
                        string("ui.startup_initializing_core"),
                    ]
                )
                self.assertEqual(
                    str(screen.query_one("#loading-diagnostics", Static).render()),
                    "Checking dependencies...\nLoading core...\nInitializing core...",
                )
                self.assertIn(
                    str(screen.query_one("#loading-spinner", Static).render()),
                    {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
                )
                log = screen.query_one("#loading-log")
                log_message = screen.query_one("#loading-log-message")
                footer = screen.query_one("#loading-footer")
                starting = screen.query_one("#loading-footer-starting")
                quit_hint = screen.query_one("#loading-footer-quit", Static)
                assert log_message.region.y - log.region.y == 2
                assert log.region.bottom - log_message.region.bottom == 2
                assert footer.region.right <= 80
                assert starting.region.x >= 2
                assert quit_hint.region.right <= 78
                assert str(quit_hint.render()) == "CTRL+Q to quit"

                screen.show_error("Backend failed")
                assert str(
                    screen.query_one("#loading-state-label", Static).render()
                ) == string("status.failed_to_start")
                assert (
                    str(screen.query_one("#loading-log-message", Static).render())
                    == "Backend failed"
                )
                assert not screen.query_one("#loading-spinner", Static).display
                assert str(screen.query_one("#loading-wait", Static).render()) == (
                    string("ui.loading_cannot_continue", app_name=APP_NAME)
                )
                assert str(
                    screen.query_one("#loading-footer-starting", Static).render()
                ) == string("ui.app_could_not_start", app_name=APP_NAME)

        asyncio.run(run_smoke_test())

    def test_loading_screen_receives_only_info_level_log_lines(self) -> None:
        """Verify verbose and debug lines do not replace the loading log message."""
        ui = CeluneUI()
        ui.celune = cast(Celune, SimpleNamespace(log_level="debug"))
        ui.logs = None
        loading_screen = mock.Mock()
        ui._loading_screen = loading_screen

        ui.safe_log("debug line", loglevel="debug")
        ui.safe_log("verbose line", loglevel="verbose")
        ui.safe_log("info line", loglevel="info")

        loading_screen.set_latest_log_message.assert_called_once_with("info line")

    def test_deferred_startup_diagnostics_follow_runtime_stages(self) -> None:
        """Verify verbose startup diagnostics are emitted as deferred work advances."""
        ui = CeluneUI(
            startup_loader=mock.Mock(return_value=mock.sentinel.celune),
            startup_log_level="verbose",
        )
        ui.receive_startup_diagnostic = mock.Mock()
        ui.call_from_thread = mock.Mock()

        with mock.patch.object(ui_app, "_load_ui_runtime_dependencies"):
            ui._load_deferred_runtime()

        ui.receive_startup_diagnostic.assert_not_called()
        ui.call_from_thread.assert_called_once_with(
            ui.attach_celune,
            mock.sentinel.celune,
        )

    def test_deferred_startup_diagnostics_are_quiet_at_info_level(self) -> None:
        """Verify normal startup does not add verbose stage diagnostics."""
        ui = CeluneUI(startup_loader=mock.Mock(return_value=mock.sentinel.celune))
        ui.receive_startup_diagnostic = mock.Mock()
        ui.call_from_thread = mock.Mock()

        with mock.patch.object(ui_app, "_load_ui_runtime_dependencies"):
            ui._load_deferred_runtime()

        ui.receive_startup_diagnostic.assert_not_called()

    def test_startup_diagnostics_update_terminal_title_transitions(self) -> None:
        """Verify each loading-screen startup stage publishes a concise title."""
        ui = CeluneUI()
        ui._set_terminal_status = mock.Mock()

        ui.receive_startup_diagnostic(string("ui.startup_initializing_core"))

        ui._set_terminal_status.assert_called_once_with(
            "initializing",
            string("osc.action_initializing_core"),
        )

    def test_status_messages_use_concise_terminal_title_actions(self) -> None:
        """Verify core status transitions do not place full status text in titles."""
        ui = CeluneUI()
        status_cases = {
            "status.downloading_audio": ("speaking", "osc.action_downloading"),
            "status.generating": ("speaking", "osc.action_generating_audio"),
            "status.normalizing": ("thinking", "osc.action_normalizing"),
            "status.waiting_for_model": (
                "initializing",
                "osc.action_waiting_for_model",
            ),
            "status.warming_up": ("initializing", "osc.action_warming_up"),
            "status.reloading_backend": (
                "reloading",
                "osc.action_loading_backend",
            ),
            "status.reloading_character": (
                "reloading",
                "osc.action_loading_voice",
            ),
            "status.restoring_backend": ("reloading", "osc.action_restoring"),
            "status.missing_dependency": (
                "error",
                "osc.action_missing_dependency",
            ),
        }

        for status_key, expected in status_cases.items():
            state, action_key = expected
            assert ui._terminal_status_for(string(status_key), "info") == (
                state,
                string(action_key),
            )

        for status_key in ("pipeline.playing_label", "pipeline.revoicing_label"):
            assert ui._terminal_status_for(
                string(status_key, label="long technical label"),
                "info",
            ) == ("speaking", string("osc.action_playing_audio"))

        assert ui._terminal_status_for("untranslated failure", "error") == (
            "error",
            string("osc.action_error"),
        )
        assert ui._terminal_status_for("untranslated warning", "warning") == (
            "warning",
            string("osc.action_warning"),
        )

    def test_pre_attach_logs_use_configured_startup_log_level(self) -> None:
        """Verify deferred full-mode startup logs use the configured level before attach."""
        ui = CeluneUI(startup_log_level="info")
        ui.logs = None
        loading_screen = mock.Mock()
        ui._loading_screen = loading_screen

        ui.safe_log("hidden verbose line", loglevel="verbose")
        ui.safe_log("visible info line", loglevel="info")

        loading_screen.set_latest_log_message.assert_called_once_with(
            "visible info line"
        )

    def test_start_background_init_shows_loading_screen_before_loading_tts(
        self,
    ) -> None:
        """Verify the loading screen is pushed before the engine load worker starts."""
        ui = CeluneUI()

        with (
            mock.patch.object(ui, "_show_loading_screen") as show_loading_screen,
            mock.patch.object(ui, "load_tts") as load_tts,
        ):
            ui.start_background_init()

        show_loading_screen.assert_called_once_with()
        load_tts.assert_called_once_with()

    def test_start_background_init_does_not_emit_redundant_diagnostic(self) -> None:
        """Verify deferred model work does not add another startup checkpoint."""
        ui = CeluneUI(startup_log_level="verbose")
        ui._emit_startup_diagnostic = mock.Mock()
        ui.load_tts = mock.Mock()

        with mock.patch.object(ui, "_show_loading_screen"):
            ui.start_background_init()

        ui._emit_startup_diagnostic.assert_not_called()
        ui.load_tts.assert_called_once_with()

    def test_dismiss_loading_screen_fades_then_hides_overlay(self) -> None:
        """Verify successful initialization hides the overlay in place."""
        ui = CeluneUI()
        loading_screen = mock.Mock()
        main_container = mock.Mock()
        ui._loading_screen = loading_screen

        def finish_fade(*_args, on_complete=None, **_kwargs) -> None:
            if on_complete is not None:
                on_complete()

        def run_after_refresh(callback) -> None:
            callback()

        with (
            mock.patch.object(
                loading_screen,
                "animate",
                side_effect=finish_fade,
            ) as fade,
            mock.patch.object(
                ui,
                "_animate_opacity",
                side_effect=finish_fade,
            ) as main_fade,
            mock.patch.object(
                ui,
                "query_one",
                return_value=main_container,
            ),
            mock.patch.object(ui, "_refresh_logs") as refresh_logs,
            mock.patch.object(
                ui,
                "call_after_refresh",
                side_effect=run_after_refresh,
            ),
        ):
            ui._dismiss_loading_screen()

        assert not loading_screen.display
        main_container.refresh.assert_called_once_with(layout=True, repaint=True)
        fade.assert_called_once_with(
            "opacity",
            0.0,
            on_complete=mock.ANY,
            duration=ui_app._LOADING_FADE_SECONDS,
            easing="out_cubic",
        )
        main_fade.assert_called_once_with(
            main_container,
            1.0,
            duration=ui_app._MAIN_UI_FADE_SECONDS,
        )
        assert loading_screen.display is False
        refresh_logs.assert_called_once_with()

    def test_tts_log_preserves_backend_log_level_filtering(self) -> None:
        """Verify isolated backend debug logs stay hidden at the info level."""
        ui = CeluneUI()
        ui.celune = cast(Celune, SimpleNamespace(log_level="info"))
        ui.safe_log = mock.Mock()

        ui.tts_log("hidden", loglevel="debug")
        assert ui.safe_log.call_args_list == [
            mock.call("hidden", "info", loglevel="debug")
        ]
        ui.safe_log.reset_mock()

        ui.tts_log("visible", loglevel="info")
        assert ui.safe_log.call_args_list == [
            mock.call("visible", "info", loglevel="info")
        ]

    def test_headless_log_preserves_backend_log_level_filtering(self) -> None:
        """Verify headless isolated backend logs honor the configured level."""
        ui = CeluneHeadlessUI({"headless_nocolor": True})
        ui.celune = cast(Celune, SimpleNamespace(log_level="info"))

        with mock.patch("builtins.print") as print_output:
            ui.headless_log("hidden", loglevel="debug")
            print_output.assert_not_called()

            ui.headless_log("visible", loglevel="info")
            print_output.assert_called_once()

    def test_crossfade_vc_overlap_keeps_mono_audio_one_dimensional(self) -> None:
        """Verify mono live VC overlap crossfades stay valid 1D audio."""
        ui = CeluneUI()
        previous = np.linspace(-1.0, -0.25, 4, dtype=np.float32)
        current = np.linspace(0.25, 1.0, 4, dtype=np.float32)

        blended = ui._crossfade_vc_overlap(previous, current)

        assert blended.shape == (4,)
        assert blended.dtype == np.float32

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

        assert blended.shape == (4, 2)
        assert blended.dtype == np.float32

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
            assert queued is not None
            if queued is None:
                pytest.fail("expected queued VC chunks")
            queued_labels.append(queued[2])

        assert queued_labels == ["newer", "fresh"]

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
        assert queued is not None
        if queued is None:
            pytest.fail("expected a queued VC chunk")
        assert queued[2] == "fresh"

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
        assert queued_final is not None
        if queued_final is None:
            pytest.fail("expected a final VC chunk before the stop marker")
        assert queued_final[2] == "final"
        assert queued_final[3]
        assert queued_stop is None

    def test_vc_input_has_voice_uses_rms_threshold(self) -> None:
        """Verify live VC VAD only treats sufficiently loud audio as speech."""
        quiet_audio = np.full((256, 2), 0.002, dtype=np.float32)
        voiced_audio = np.full((256, 2), 0.05, dtype=np.float32)

        assert not CeluneUI._vc_input_has_voice(quiet_audio)
        assert CeluneUI._vc_input_has_voice(voiced_audio)

    def test_textual_ui_can_mount_before_engine_attachment(self) -> None:
        """Verify the loading frame can mount while the engine is still absent."""
        ui = CeluneUI()
        fake_widgets = {
            "#container": Vertical(),
            "#logs": RichLog(),
            "#input": TextArea(),
            "#status": Label(),
            "#resources": Label(),
            "#caption": Label(),
            "#style": Button(),
            "#vc-mode": Button(),
            "#vc-pitch": Button(),
            "#progress": SimpleNamespace(update=lambda **_: None),
            "#progress-label": Label(),
            "#header": Label(),
            "#loading-overlay": CeluneLoadingScreen(),
        }
        with (
            mock.patch("celune.ui.app.colors.configure_theme"),
            mock.patch.object(
                ui,
                "query_one",
                side_effect=lambda selector, *_args: fake_widgets[selector],
            ),
            mock.patch.object(ui, "query", return_value=[]),
            mock.patch.object(ui, "set_interval"),
            mock.patch.object(ui, "set_focus"),
            mock.patch.object(ui, "call_after_refresh") as call_after_refresh,
            mock.patch.object(ui, "safe_status"),
            mock.patch.object(ui, "_refresh_status"),
            mock.patch.object(ui, "_refresh_theme_text"),
            mock.patch.object(ui, "_refresh_logs"),
        ):
            ui.on_mount()

        self.assertIsNone(ui.celune)
        self.assertIsNotNone(ui._loading_screen)
        if ui._loading_screen is not None:
            self.assertTrue(ui._loading_screen.display)
        call_after_refresh.assert_not_called()

    def test_progress_label_switches_between_percentage_time_and_hidden(self) -> None:
        """Verify the progress readout uses the applicable display mode."""
        label = ProgressLabel()

        label.set_progress(1, 4)
        assert label.display
        assert str(label.render()) == " 25%"

        label.set_progress(48000, 96000, audio_playing=True, sample_rate=48000)
        assert str(label.render()) == "00:01"

        label.set_progress(None, None)
        assert not label.display
        assert str(label.render()) == ""

    def test_safe_progress_completes_late_idle_indeterminate_updates(self) -> None:
        """Verify stale startup progress cannot leave the idle bar indeterminate."""

        class FakeProgressBar:
            """Small progress bar test double."""

            def __init__(self) -> None:
                self.updates: list[dict[str, Optional[float]]] = []
                self.display = True
                self.parent = None
                self.styles = SimpleNamespace(opacity=1.0)

            def update(self, **values: Optional[float]) -> None:
                """Capture progress updates."""
                self.updates.append(values)

        ui = CeluneUI()
        progress_bar = FakeProgressBar()
        ui.progress_bar = cast(ProgressBar, progress_bar)
        ui.progress_label = ProgressLabel()
        ui.celune_ready = True
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                cur_state="idle",
                persona_loading=False,
            ),
        )

        ui.safe_progress(1, 1)
        ui.safe_progress(None, None)

        assert progress_bar.updates[-1] == {"total": 1, "progress": 1}
        assert ui.progress_label.display
        assert str(ui.progress_label.render()) == "100%"

    def test_attach_celune_starts_runtime_after_loading_frame(self) -> None:
        """Verify engine attachment schedules normal loading in the same app."""
        ui = CeluneUI()
        fake_celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                backend=SimpleNamespace(is_fake=True),
                log_callback=None,
                status_callback=None,
                error_callback=None,
                idle_callback=None,
                queue_avail_callback=None,
                voice_changed_callback=None,
                change_input_state_callback=None,
                change_voice_lock_state_callback=None,
                progress_callback=None,
                caption_callback=None,
                caption_timing_callback=None,
                close=lambda: None,
                glow=SimpleNamespace(fatal=lambda: None),
            ),
        )
        ui._loading_screen = mock.Mock()
        ui._loading_screen.display = True
        ui._runtime_intervals_started = True

        with (
            mock.patch("celune.ui.app.default_loader", return_value=None),
            mock.patch("celune.ui.app.ui_resources") as resources,
            mock.patch.object(ui, "_enable_runtime_log_capture"),
            mock.patch.object(ui, "refresh_vc_controls"),
            mock.patch.object(ui, "update_resources"),
            mock.patch.object(ui, "start_background_init") as start_init,
            mock.patch.object(ui, "_refresh_theme_text"),
            mock.patch.object(ui, "call_after_refresh") as call_after_refresh,
        ):
            ui.attach_celune(fake_celune)

        self.assertIs(ui.celune, fake_celune)
        self.assertEqual(fake_celune.log_callback, ui.tts_log)
        resources.prime_usage.assert_called_once_with()
        call_after_refresh.assert_called_once_with(start_init)

    def test_callback_binding_does_not_chain_same_callback(self) -> None:
        """Verify an already-installed callback is not invoked through a wrapper."""
        calls: list[str] = []

        def callback(message: str) -> None:
            calls.append(message)

        ui = CeluneUI()
        fake_celune = cast(
            Celune,
            SimpleNamespace(log_callback=callback),
        )
        ui.celune = fake_celune

        ui._chain_runtime_callback("log_callback", callback)
        fake_celune.log_callback("message")

        assert calls == ["message"]
        assert fake_celune.log_callback is callback

    def test_log_binding_does_not_chain_startup_sink_again(self) -> None:
        """Verify startup log delivery does not duplicate the UI log callback."""
        calls: list[str] = []

        def ui_callback(message: str) -> None:
            calls.append(message)

        def buffered_callback(message: str) -> None:
            ui_callback(message)

        ui = CeluneUI()
        fake_celune = cast(
            Celune,
            SimpleNamespace(
                log_callback=buffered_callback,
                _startup_log_sink=ui_callback,
            ),
        )
        ui.celune = fake_celune

        ui._chain_runtime_callback("log_callback", ui_callback)
        fake_celune.log_callback("message")

        assert calls == ["message"]
        assert fake_celune.log_callback is buffered_callback

    def test_textual_ui_mount_enables_stdio_redirects_before_runtime_load(self) -> None:
        """Verify mount captures startup stdio before Celune begins loading."""
        ui = CeluneUI()
        fake_widgets = {
            "#container": Vertical(),
            "#logs": RichLog(),
            "#input": TextArea(),
            "#status": Label(),
            "#resources": Label(),
            "#caption": Label(),
            "#style": Button(),
            "#vc-mode": Button(),
            "#vc-pitch": Button(),
            "#progress": SimpleNamespace(update=lambda **_: None),
            "#progress-label": Label(),
            "#header": Label(),
            "#loading-overlay": CeluneLoadingScreen(),
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

            assert sys.stdout is ui._log_stdout
            assert sys.stderr is ui._log_stderr
            assert ui._runtime_log_capture_enabled
            set_focus.assert_called_once_with(None)
        finally:
            ui.disable_runtime_log_capture()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_textual_ui_mount_binds_callbacks_for_attached_runtime(self) -> None:
        """Verify an attached runtime adopts the Textual UI callbacks on mount."""
        ui = CeluneUI()
        fake_widgets = {
            "#container": Vertical(),
            "#logs": RichLog(),
            "#input": TextArea(),
            "#status": Label(),
            "#resources": Label(),
            "#caption": Label(),
            "#style": Button(),
            "#vc-mode": Button(),
            "#vc-pitch": Button(),
            "#progress": SimpleNamespace(update=lambda **_: None),
            "#progress-label": Label(),
            "#header": Label(),
            "#loading-overlay": CeluneLoadingScreen(),
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
                caption_callback=None,
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

            assert ui.celune.log_callback.__self__ is ui
            assert ui.celune.log_callback.__func__ is CeluneUI.tts_log
            assert ui.celune.status_callback.__self__ is ui
            assert ui.celune.status_callback.__func__ is CeluneUI.safe_status
            assert ui.celune.error_callback.__self__ is ui
            assert ui.celune.error_callback.__func__ is CeluneUI.error
            assert ui.celune.idle_callback.__self__ is ui
            assert ui.celune.idle_callback.__func__ is CeluneUI.tts_idle
            assert ui.celune.queue_avail_callback.__self__ is ui
            assert ui.celune.queue_avail_callback.__func__ is CeluneUI.tts_queue_avail
            assert ui.celune.voice_changed_callback.__self__ is ui
            assert (
                ui.celune.voice_changed_callback.__func__ is CeluneUI.tts_voice_changed
            )
            assert ui.celune.change_input_state_callback.__self__ is ui
            assert (
                ui.celune.change_input_state_callback.__func__
                is CeluneUI.change_input_state
            )
            assert ui.celune.change_voice_lock_state_callback.__self__ is ui
            assert (
                ui.celune.change_voice_lock_state_callback.__func__
                is CeluneUI.change_voice_lock_state
            )
            assert ui.celune.progress_callback.__self__ is ui
            assert ui.celune.progress_callback.__func__ is CeluneUI.safe_progress
            assert ui.celune.caption_callback.__self__ is ui
            assert ui.celune.caption_callback.__func__ is CeluneUI.tts_caption
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
                assert ui._runtime_log_capture_enabled
                assert sys.stdout is ui._log_stdout
                assert sys.stderr is ui._log_stderr

                ui.disable_runtime_log_capture()

            assert not ui._runtime_log_capture_enabled
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
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

            assert ui._old_stdout is terminal
            assert ui._old_stderr is terminal
            assert ui._log_stdout is not None
            assert ui._log_stderr is not None
            assert ui._log_stdout.underlying_stdout is terminal
            assert ui._log_stdout.underlying_stderr is terminal
            assert ui._log_stderr.underlying_stdout is terminal
            assert ui._log_stderr.underlying_stderr is terminal
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    @staticmethod
    def test_log_redirect_ansi_forwards_and_flushes_underlying_stdout() -> None:
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

        assert captured == [
            (
                "C:/tmp/hub.py:110: FutureWarning: TRANSFORMERS_CACHE is deprecated",
                "warning",
            )
        ]

    def test_log_redirect_preserves_severity_across_multiline_records(self) -> None:
        """Verify continuation lines inherit the first line's inferred severity."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
        )

        redirect.write("FutureWarning: first line\nsecond line\n")

        assert captured == [
            ("FutureWarning: first line", "warning"),
            ("second line", "warning"),
        ]

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

        assert not captured

    def test_runtime_log_filters_gpt_sovits_text2semantic_loading(self) -> None:
        """Verify GPT-SoVITS checkpoint loading chatter stays out of the UI log."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
            filter_messages=ui_app._RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )

        redirect.write("Loading Text2Semantic weights from C:/models/custom-e20.ckpt\n")

        assert not captured

    def test_runtime_log_filters_transformers_ignored_generation_flags(self) -> None:
        """Verify harmless Transformers generation-flag warnings stay out of the UI log."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="warning",
            filter_messages=ui_app._RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )

        redirect.write(
            "Internal runtime warning: The following generation flags are not valid "
            "and may be ignored: ['temperature', 'top_k'].\n"
        )

        assert not captured

    def test_runtime_log_filters_firered_frontend_notice(self) -> None:
        """Verify FireRedTTS3's redundant frontend notice stays out of the UI log."""
        stream = mock.Mock()
        stream.isatty.return_value = True
        captured: list[tuple[str, str]] = []
        redirect = ui_terminal.LogRedirect(
            stdout=stream,
            stderr=stream,
            write_callback=lambda msg, severity: captured.append((msg, severity)),
            default_severity="info",
            filter_messages=ui_app._RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )

        redirect.write("[INFO] FireRedTTS3 (text front-end) loaded\n")

        assert not captured

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

        assert not captured

    @staticmethod
    def test_load_tts_writes_terminal_title_to_original_stdout() -> None:
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
                use_normalization=True,
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

            terminal.write.assert_called_with(f"\x1b]0;{APP_NAME} ・ Ready ・ Idle\x07")
            terminal.flush.assert_called()
            ui.safe_progress.assert_called_once_with(1, 1)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_sleep_timer_callbacks_ignore_textual_shutdown(self) -> None:
        """Verify background sleep callbacks tolerate Textual stopping concurrently."""
        ui = CeluneUI()
        ui.call_from_thread = mock.Mock(side_effect=RuntimeError("App is not running"))

        schedule_thread = threading.Thread(target=ui._schedule_sleep_timer)
        cancel_thread = threading.Thread(target=ui._cancel_sleep_timer)
        schedule_thread.start()
        cancel_thread.start()
        schedule_thread.join(timeout=1)
        cancel_thread.join(timeout=1)

        self.assertFalse(schedule_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())
        self.assertEqual(ui.call_from_thread.call_count, 2)

    def test_headless_ui_warns_without_attached_celune(self) -> None:
        """Verify headless mode warns before doing nothing without Celune."""
        ui = CeluneHeadlessUI({"headless_nocolor": True})
        with (
            warnings.catch_warnings(record=True) as caught,
            mock.patch("celune.ui.headless.signal.signal"),
            mock.patch("celune.ui.headless.time.sleep", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            warnings.simplefilter("always")
            ui.run()

        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)
        assert f"CeluneHeadlessUI has no attached {APP_NAME} instance" in str(
            caught[0].message
        )

    def test_load_tts_marks_ui_error_when_startup_returns_false(self) -> None:
        """Verify handled startup failures leave the UI in an error state."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "No Voice Set",
            actions=ButtonActions(press=False, hold=False),
        )
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
        assert ui.cur_state == "error"
        assert ui.input_box.placeholder == "Please wait"
        assert ui.style_button.actions == ButtonActions(press=False, hold=False)
        assert not ui._fatal_error_active

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
        assert ui.cur_state != "error"

    def test_load_tts_publishes_the_first_loaded_voice_in_agent_test_mode(self) -> None:
        """Verify agent test startup does not leave the voice button on its placeholder."""
        ui = CeluneUI()
        ui.safe_status = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui.tts_voice_changed = mock.Mock()
        ui._dismiss_loading_screen = mock.Mock()
        ui._finish_test_startup = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                load=lambda: True,
                voices=("balanced", "bold"),
                current_voice=None,
                use_normalization=False,
                backend_mode="agent_test",
            ),
        )

        load_tts = getattr(CeluneUI.load_tts, "__wrapped__", CeluneUI.load_tts)
        load_tts(ui)

        ui.tts_voice_changed.assert_called_once_with("balanced")

    def test_finished_agent_test_reconciles_all_visible_state(self) -> None:
        """Verify test completion refreshes logs, voice, caption, status, and locks."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                test_finished=True,
                current_voice=None,
                voices=("balanced",),
                cur_state="stopped",
            ),
        )
        ui.celune_ready = True
        ui._hide_caption_widgets = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.change_voice_lock_state = mock.Mock()
        ui.tts_voice_changed = mock.Mock()
        ui.safe_status = mock.Mock()
        ui._refresh_logs = mock.Mock()

        ui._apply_test_finished_state()

        ui._hide_caption_widgets.assert_called_once_with()
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.change_voice_lock_state.assert_called_once_with(locked=True)
        ui.tts_voice_changed.assert_called_once_with("balanced")
        self.assertEqual(ui.input_box.placeholder, string("ui.stopped_placeholder"))
        ui.safe_status.assert_called_once_with(string("status.stopped"), "sleeping")
        ui._refresh_logs.assert_called_once_with()
        self.assertEqual(
            ui._terminal_status_for(string("status.stopped"), "sleeping"),
            ("stopped", string("osc.action_stopped")),
        )

    def test_log_refresh_does_not_duplicate_pending_background_message(self) -> None:
        """Verify a refresh and its queued message render one log entry."""
        ui = CeluneUI()
        logs = mock.Mock()
        logs.scroll_offset = SimpleNamespace(x=0, y=0)
        logs.auto_scroll = True
        ui.logs = cast(RichLog, logs)
        ui.log_history = [("Test result", "info")]

        ui._refresh_logs()
        ui.on_uilog_message(UILogMessage("Test result", "info"))

        logs.write.assert_called_once()
        self.assertEqual(ui._rendered_log_count, 1)

    def test_queued_log_reconciles_stale_render_cursor(self) -> None:
        """Verify a queued log is restored when the render cursor got ahead of the widget."""
        ui = CeluneUI()
        logs = mock.Mock()
        logs.lines = []
        logs._size_known = True
        logs.auto_scroll = True
        ui.logs = cast(RichLog, logs)
        ui.log_history = [("Backend ready", "info")]
        ui._rendered_log_count = 1

        ui.on_uilog_message(UILogMessage("Backend ready", "info"))

        logs.clear.assert_called_once_with()
        logs.write.assert_called_once()
        self.assertEqual(ui._rendered_log_count, 1)

    def test_log_refresh_preserves_rendered_history(self) -> None:
        """Verify a layout refresh does not clear already rendered log entries."""
        ui = CeluneUI()
        logs = mock.Mock()
        logs.lines = [mock.Mock()]
        logs.auto_scroll = True
        logs._size_known = True
        ui.logs = cast(RichLog, logs)
        ui.log_history = [("Existing log", "info")]
        ui._rendered_log_count = 1

        ui._refresh_logs()

        logs.clear.assert_not_called()
        logs.write.assert_not_called()
        logs.scroll_end.assert_called_once_with(
            animate=False,
            immediate=True,
            force=True,
        )

    def test_finished_agent_test_ignores_late_speaking_status(self) -> None:
        """Verify late playback callbacks cannot overwrite the stopped status."""
        ui = CeluneUI()
        ui.status = Label()
        ui.celune = cast(Celune, SimpleNamespace(test_finished=True))

        ui.safe_status(string("status.speaking"))

        self.assertNotEqual(ui._status_text, string("status.speaking"))

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

        assert handled
        assert ui.input_box.text == ""
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
        ui.style_button = VoiceButton(
            "No Voice Set",
            actions=ButtonActions(press=False, hold=False),
        )
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

        assert ui.cur_state == "error"
        assert ui.input_box.placeholder == "Please wait"
        assert ui.style_button.actions == ButtonActions(press=False, hold=False)

    def test_tts_idle_keeps_controls_locked_while_runtime_is_reloading(self) -> None:
        """Verify idle playback callbacks do not unlock the UI mid-reload."""
        ui = CeluneUI()
        ui.celune_ready = True
        ui.cur_state = "idle"
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Balanced",
            actions=ButtonActions(press=False, hold=False),
        )
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

        assert ui.celune.locked
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.change_voice_lock_state.assert_called_once_with(locked=True)
        ui.safe_status.assert_not_called()

    @staticmethod
    def test_on_button_pressed_ignores_voice_switch_when_no_voices_loaded() -> None:
        """Verify voice cycling is blocked cleanly when startup left no voices loaded."""
        ui = CeluneUI()
        ui.celune_ready = False
        ui.style_button = VoiceButton(
            "No Voice Set",
            actions=ButtonActions(press=False, hold=False),
        )
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
        assert "CTRL+C" not in exit_page
        assert "CTRL+R toggle recording" not in pages

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

        assert "CTRL+R toggle recording" in pages

    def test_textual_resource_footer_rotates_cost_equivalent_pages(self) -> None:
        """Verify the resource footer includes saved-cost pages for each provider."""
        celune = cast(
            Celune,
            SimpleNamespace(
                is_in_tutorial=False,
                config={"theme": "dark"},
                backend=SimpleNamespace(current_seed=None),
                input_mode="text_to_speech",
                total_generated_speech_seconds=120.0,
                historical_generated_speech_seconds=180.0,
            ),
        )

        pages = ui_resources.resource_pages(celune, "celune")
        cost_pages = [
            page
            for page in pages
            if "(this session): $" in page or "(overall): $" in page
        ]

        assert len(cost_pages) == len(COST_EQUIVALENTS) * 2
        assert "gemini-flash-tts (this session): $0.03" in cost_pages
        assert "openai-realtime (this session): $0.19" in cost_pages
        assert "gemini-flash-tts (overall): $0.08" in cost_pages
        assert "openai-realtime (overall): $0.48" in cost_pages

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

        def queue_streaming_segment(
            _engine: Celune,
            audio: np.ndarray,
            sample_rate: int,
            label: str,
            source_id: Optional[int] = None,
            generation: Optional[int] = None,
            reset_ready_announcement: bool = False,
        ) -> int:
            """Capture one converted segment submitted for playback."""
            discard(_engine)
            discard(generation)
            queued_segments.append(
                (
                    np.asarray(audio, dtype=np.float32).copy(),
                    sample_rate,
                    label,
                    source_id,
                    reset_ready_announcement,
                )
            )
            return 1 if source_id is None else source_id

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
                side_effect=queue_streaming_segment,
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
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.ones((60000, 2), dtype=np.float32),
                )
                for _ in range(50):
                    if cast(mock.Mock, ui.celune.convert_audio).call_count >= 1:
                        break
                    time.sleep(0.01)
                invoke_captured_callback(
                    captured_callback,
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
        assert mock_query_devices.call_count >= 1
        assert mock_input_stream.call_args.kwargs["device"] == "Stereo Mix (Realtek)"
        for _ in range(50):
            if cast(mock.Mock, ui.celune.convert_audio).call_count >= 1:
                break
            time.sleep(0.01)
