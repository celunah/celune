# SPDX-License-Identifier: Apache-2.0
"""Tests for Textual status, command, and runtime presentation behavior."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import sys
import asyncio
import subprocess
from typing import cast
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from collections.abc import Callable, Awaitable, Coroutine

from textual import events
from textual.widget import Widget

from celune.i18n import string
from celune.ui import app as ui_app
from celune.constants import APP_NAME
from celune.terminal import set_terminal_title
from celune.typing.common import JSONSerializable
from celune.ui.app import (
    CeluneUI,
    VoiceButton,
    ButtonActions,
)

from ..support import CeluneTestCase


class TestUIRuntime(CeluneTestCase):
    """Verify Textual runtime presentation and lazy startup behavior."""

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
        assert not button.has_class("-actions-disabled")

        CeluneUI.change_voice_lock_state(
            ui,
            locked=False,
            can_open_menu=False,
        )

        assert button.actions == ButtonActions(press=True, hold=False)
        assert not button.has_class("-actions-disabled")

    def test_singing_backend_locks_voice_cycle_and_keeps_menu_available(self) -> None:
        """Keep style cycling locked while allowing the singing-pack menu."""
        button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=True, hold=True),
        )
        ui = cast(
            CeluneUI,
            SimpleNamespace(
                celune=SimpleNamespace(backend=SimpleNamespace(singing=True)),
                celune_ready=True,
                style_button=button,
                update_resources=mock.Mock(),
                _run_on_ui_thread=lambda callback: callback(),
            ),
        )

        CeluneUI.change_voice_lock_state(
            ui,
            locked=False,
            can_open_menu=True,
        )

        assert button.actions == ButtonActions(press=False, hold=True)

        ui.celune.cur_state = "reloading"
        CeluneUI.change_voice_lock_state(ui, locked=True)
        assert button.actions == ButtonActions(press=False, hold=False)

    def test_singing_voice_menu_lists_only_one_non_searchable_singing_entry(
        self,
        tmp_path: Path,
    ) -> None:
        """Offer only CECHAR singing roles as single, non-searchable menu rows."""
        from celune import cevoice

        singing_path = tmp_path / "singer.cechar"
        speaking_path = tmp_path / "speaker.cevoice"
        singing_path.touch()
        speaking_path.touch()

        def open_pack(path: Path) -> SimpleNamespace:
            return SimpleNamespace(
                path=path,
                voice_order=("balanced", "bold"),
                singing_voice="sing" if path == singing_path else None,
            )

        show_menu = mock.Mock()
        ui = cast(
            CeluneUI,
            SimpleNamespace(
                celune=SimpleNamespace(backend=SimpleNamespace(singing=True)),
                _active_menu=None,
                _voice_menu_paths={},
                safe_log=mock.Mock(),
                _show_menu=show_menu,
                _menu_footer=CeluneUI._menu_footer,
            ),
        )

        with (
            mock.patch.object(cevoice, "bundled_voices_dir", return_value=tmp_path),
            mock.patch.object(cevoice, "active_bundle_path", return_value=singing_path),
            mock.patch.object(cevoice.CEVoice, "open", side_effect=open_pack),
            mock.patch.object(
                cevoice,
                "bundle_display_name",
                side_effect=lambda bundle: bundle.path.stem.title(),
            ),
        ):
            CeluneUI.open_voice_menu(ui)

        menu = show_menu.call_args.args[0]
        assert len(menu.options) == 1
        option = menu.options[0]
        assert option.label == "Singer"
        assert option.value == "sing"
        assert option.confirm_value == "sing"
        assert not option.editable
        assert option.autocomplete_values is None
        assert string("ui.menu_hint_search") not in menu.footer_builder(option)
        assert ui._voice_menu_paths == {"Singer": singing_path}
        assert show_menu.call_args.args[1] == "voice"

    def test_singing_voice_menu_selection_reloads_only_the_character_pack(self) -> None:
        """Select a singing-capable pack without unlocking voice-style switching."""
        from celune import cevoice

        bundle_path = Path("singer.cechar")
        calls: list[Path] = []
        set_voice = mock.Mock(return_value=True)

        def set_bundle(path: Path) -> bool:
            calls.append(path)
            return True

        fake_celune = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            sleeping=False,
            set_cevoice_and_wait=set_bundle,
            set_voice_and_wait=set_voice,
            voices=("balanced", "bold"),
            current_voice="balanced",
        )
        ui = SimpleNamespace(
            celune=fake_celune,
            _voice_menu_paths={"Singer": bundle_path},
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

        with mock.patch.object(
            cevoice, "active_bundle_path", return_value=Path("old.cechar")
        ):
            asyncio.run(
                apply_voice_selection(
                    cast(CeluneUI, ui),
                    {"pack": "Singer", "entry": "sing"},
                )
            )

        assert calls == [bundle_path]
        set_voice.assert_not_called()
        assert ui.celune_styles == ("balanced", "bold")
        ui.change_voice_lock_state.assert_called_once_with(
            locked=True,
            can_open_menu=True,
        )

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

    def test_voice_button_long_press_reaches_voice_handler(self) -> None:
        """Verify the real hold timer dispatches the voice-specific message."""

        class Harness(ui_app.App[None]):
            """Minimal Textual host for the voice hold dispatch test."""

            CSS = ui_app.CELUNE_CSS

            def __init__(self) -> None:
                super().__init__()
                self.held = 0

            def compose(self) -> ui_app.ComposeResult:
                """Mount one enabled voice button."""
                yield VoiceButton(
                    "Voice",
                    id="voice",
                    actions=ButtonActions(press=False, hold=True),
                )

            def on_voice_button_held(self, _event: VoiceButton.Held) -> None:
                """Record the message that the application should receive."""
                self.held += 1

        async def run_hold_test() -> None:
            app = Harness()
            async with app.run_test(size=(40, 10)) as pilot:
                assert await pilot.mouse_down("#voice")
                await pilot.pause(0.7)
                assert await pilot.mouse_up("#voice")
                await pilot.pause()
                assert app.held == 1

        asyncio.run(run_hold_test())

    def test_voice_button_press_is_gated_without_disabling_the_button(self) -> None:
        """Verify an unavailable press action leaves the native button enabled."""
        button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=True),
        )
        button.post_message = mock.Mock()

        button.press()

        assert not button.has_class("-actions-disabled")
        button.post_message.assert_not_called()

    def test_buttons_use_action_capabilities_and_disabled_appearance(self) -> None:
        """Verify ordinary buttons default to press-only capabilities."""
        button = ui_app.Button(
            "Action",
            actions=ButtonActions(press=False),
        )

        assert button.actions == ButtonActions(press=False)
        assert button.has_class("-actions-disabled")

        button.actions = ButtonActions()

        assert not button.has_class("-actions-disabled")

    def test_button_click_clears_focus_after_release(self) -> None:
        """Verify clicking a button does not leave its focus highlight latched."""

        class Harness(ui_app.App):
            """Minimal Textual host for the action button."""

            CSS = ui_app.CELUNE_CSS

            def compose(self) -> ui_app.ComposeResult:
                """Mount one button for the focus regression test."""
                yield ui_app.Button("Action", id="action")

        async def run_focus_test() -> None:
            app = Harness()
            async with app.run_test(size=(40, 10)) as pilot:
                await pilot.click("#action")
                await pilot.pause()
                assert app.focused is None

        asyncio.run(run_focus_test())

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
        project_root = Path(__file__).resolve().parents[2]
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
        project_root = Path(__file__).resolve().parents[2]
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
