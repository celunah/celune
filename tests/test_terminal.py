# SPDX-License-Identifier: Apache-2.0
"""Tests for terminal and Textual selection menus."""

import pytest
from textual.app import App
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, Static

from celune.ui.app import CeluneUI, SelectMenuOverlay
from celune.ui.terminal import SelectMenuOption, SelectMenuWidget
from celune.ui.theme import CELUNE_CSS


def test_select_menu_widget_aligns_values_after_longest_label() -> None:
    """Editable values share a column two cells after the longest label."""
    menu = SelectMenuWidget(
        "Configuration manager",
        [
            SelectMenuOption("Backend", "mini"),
            SelectMenuOption("Voice", "calm"),
            SelectMenuOption("VRAM target", "medium"),
            SelectMenuOption("Theme", "dark"),
            SelectMenuOption("Information", editable=False),
        ],
    )

    lines = menu.render().plain.splitlines()
    value_positions = [line.index("[") for line in lines[2:6]]

    assert value_positions == [16, 16, 16, 16]
    assert lines[2].startswith("-> Backend")
    assert lines[6] == "   Information"


def test_configuration_labels_preserve_project_names() -> None:
    """Configuration labels are readable without losing project abbreviations."""
    assert CeluneUI._config_label(("api", "enabled")) == "API enabled"
    assert (
        CeluneUI._config_label(("gpt_sovits_t2s_weights_path",))
        == "GPT-SoVITS T2S weights path"
    )
    assert CeluneUI._config_label(("persona", "memory", "enabled")) == (
        "Persona memory enabled"
    )


def test_select_menu_widget_truncates_values_to_terminal_cells() -> None:
    """Long values use an ellipsis without changing their returned value."""
    value = r"C:\Users\user\AppData\Local\Celune\runtime\weights.safetensors"
    menu = SelectMenuWidget("Configuration manager", [SelectMenuOption("Path", value)])

    assert menu._truncate_cells(value, 18).endswith("…")
    assert menu.selected_value == value


def test_select_menu_widget_can_show_only_the_current_value() -> None:
    """Current-only value mode follows the highlighted editable row."""
    menu = SelectMenuWidget(
        "Voice list",
        [
            SelectMenuOption("Celune", "calm"),
            SelectMenuOption("Lune", "soft"),
            SelectMenuOption("Celine", "distorted"),
        ],
        value_display="current",
    )

    assert "[ calm ]" in menu.render().plain
    assert "[ soft ]" not in menu.render().plain
    menu.on_key(Key("down", None))
    rendered = menu.render().plain
    assert "[ calm ]" not in rendered
    assert "[ soft ]" in rendered


def test_select_menu_widget_rebuilds_footer_for_selected_row() -> None:
    """A dynamic footer follows the currently highlighted option."""
    menu = SelectMenuWidget(
        "Voice list",
        [
            SelectMenuOption(
                "Editable",
                "calm",
                autocomplete_values=("calm", "soft"),
            ),
            SelectMenuOption("Locked", editable=False),
        ],
        footer_builder=lambda option: option.label,
    )

    assert menu.render().plain.endswith("Editable")
    menu.on_key(Key("down", None))
    assert menu.render().plain.endswith("Locked")


def test_select_menu_widget_places_explanation_above_footer_hints() -> None:
    """A selected row explanation is rendered immediately above its hints."""
    menu = SelectMenuWidget(
        "Configuration manager",
        [SelectMenuOption("API enabled", True, explanation="Edit the API setting.")],
        footer_builder=lambda option: "ENTER save\nESC cancel",
    )

    lines = menu.render().plain.splitlines()
    assert lines[-2:] == ["ENTER save", "ESC cancel"]
    assert lines[-3] == "Edit the API setting."


def test_select_menu_widget_can_confirm_a_static_row_value() -> None:
    """A static row can return a fixed value without showing brackets."""
    menu = SelectMenuWidget(
        "Select voice",
        [
            SelectMenuOption(
                "Celune",
                "balanced",
                editable=False,
                confirm_value="balanced",
            )
        ],
    )

    assert "[ balanced ]" not in menu.render().plain
    assert menu.selected_value == "balanced"


def test_select_menu_widget_keybinds_and_editable_values() -> None:
    """Keybinds select rows and editable values can be changed externally."""
    menu = SelectMenuWidget(
        "Configuration manager",
        [
            SelectMenuOption("Backend", "mini", keybind="b"),
            SelectMenuOption("Voice", "calm", keybind="v"),
            SelectMenuOption("Read-only", "ignored", editable=False),
        ],
    )

    menu.on_key(Key("v", "v"))
    assert menu.selected_index == 1
    assert menu.selected_value == "calm"
    menu.set_value(1, "soft")
    assert menu.selected_value == "soft"
    with pytest.raises(ValueError, match="non-editable"):
        menu.set_value(2, "changed")


def test_select_menu_widget_cycles_and_searches_autocomplete_values() -> None:
    """Left/Right cycle candidates and typing searches their original types."""
    menu = SelectMenuWidget(
        "Voice list",
        [
            SelectMenuOption(
                "Voice",
                "calm",
                autocomplete_values=("calm", "soft", "distorted"),
            ),
            SelectMenuOption("Locked", "fixed", editable=False),
        ],
    )

    menu.on_key(Key("right", None))
    assert menu.selected_value == "soft"
    menu.on_key(Key("left", None))
    assert menu.selected_value == "calm"
    menu.on_key(Key("s", "s"))
    assert menu.selected_value == "soft"
    menu.on_key(Key("backspace", None))
    assert menu.selected_value == "calm"

    menu.on_key(Key("down", None))
    menu.on_key(Key("right", None))
    menu.on_key(Key("x", "x"))
    assert menu.selected_value is None

    typed = SelectMenuWidget(
        "VRAM target",
        [SelectMenuOption("Target", 8, autocomplete_values=(4, 8, 16))],
    )
    typed.on_key(Key("4", "4"))
    assert typed.selected_value == 4
    assert isinstance(typed.selected_value, int)

    boolean = SelectMenuWidget(
        "Boolean",
        [SelectMenuOption("Enabled", True, autocomplete_values=(True, False))],
    )
    for character in "ssssss":
        boolean.on_key(Key(character, character))
    assert boolean.selected_value is True
    boolean.on_key(Key("f", "f"))
    assert not boolean.selected_value

    integer = SelectMenuWidget("Integer", [SelectMenuOption("Count", 12)])
    integer.on_key(Key("x", "x"))
    assert integer.selected_value == 12
    assert isinstance(integer.selected_value, int)

    optional = SelectMenuWidget(
        "Optional",
        [SelectMenuOption("Locale", None, autocomplete_values=(None, "en-US"))],
    )
    optional.on_key(Key("x", "x"))
    assert optional.selected_value is None
    optional.on_key(Key("e", "e"))
    optional.on_key(Key("n", "n"))
    assert optional.selected_value == "en-US"


def test_select_menu_widget_rejects_invalid_keybinds() -> None:
    """Navigation and duplicate keybinds fail during widget construction."""
    with pytest.raises(ValueError, match="navigation keys"):
        SelectMenuWidget("Menu", [SelectMenuOption("One")], keybinds={"up": 0})
    with pytest.raises(ValueError, match="duplicate option keybind"):
        SelectMenuWidget(
            "Menu",
            [SelectMenuOption("One", keybind="x"), SelectMenuOption("Two")],
            keybinds={"X": 1},
        )


class _SelectMenuHarness(App[None]):
    """Minimal Textual host used to observe selection messages."""

    def __init__(self, menu: SelectMenuWidget) -> None:
        super().__init__()
        self.menu = menu
        self.received: list[Message] = []

    def compose(self):
        """Mount the menu as the only focusable widget."""
        yield self.menu

    def on_mount(self) -> None:
        """Give keyboard focus to the menu for pilot input."""
        self.set_focus(self.menu)

    def on_select_menu_widget_confirmed(
        self,
        message: SelectMenuWidget.Confirmed,
    ) -> None:
        """Keep confirmation messages for assertions."""
        self.received.append(message)

    def on_select_menu_widget_cancelled(
        self,
        message: SelectMenuWidget.Cancelled,
    ) -> None:
        """Keep cancellation messages for assertions."""
        self.received.append(message)


class _OverlayHarness(App[None]):
    """Minimal host used to verify the menu overlay geometry."""

    CSS = CELUNE_CSS

    def __init__(self, menu: SelectMenuWidget) -> None:
        super().__init__()
        self.menu = menu
        self.underlying_clicks = 0

    def compose(self):
        """Mount the menu through the centering overlay."""
        yield Static("Underlying", id="underlying-background")
        yield Button("Underlying", id="underlying")

    def on_mount(self) -> None:
        """Push the menu screen after the underlying screen has rendered."""
        self.push_screen(SelectMenuOverlay(self.menu))

    def on_button_pressed(self, _event: Button.Pressed) -> None:
        """Record clicks that would indicate the overlay leaked input."""
        self.underlying_clicks += 1

    def on_select_menu_widget_cancelled(
        self, _event: SelectMenuWidget.Cancelled
    ) -> None:
        """Dismiss the modal screen without exiting the application."""
        self.screen.dismiss()


@pytest.mark.anyio
async def test_select_menu_widget_confirms_value_and_cancels() -> None:
    """ENTER emits the selected value and ESC emits a cancellation message."""
    menu = SelectMenuWidget(
        "Voice list",
        [SelectMenuOption("Celune", "calm"), SelectMenuOption("Lune", "soft")],
    )
    app = _SelectMenuHarness(menu)

    async with app.run_test() as pilot:
        await pilot.press("down", "enter")
        await pilot.pause()
        assert isinstance(app.received[0], SelectMenuWidget.Confirmed)
        assert app.received[0].value == "soft"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.received[1], SelectMenuWidget.Cancelled)
        assert app.received[1].option_index == 1
        assert app.received[1].value is None


@pytest.mark.anyio
async def test_select_menu_widget_can_confirm_without_a_value() -> None:
    """The confirmation contract can intentionally omit the row value."""
    menu = SelectMenuWidget(
        "Menu",
        [SelectMenuOption("One", "value")],
        return_value=False,
    )
    app = _SelectMenuHarness(menu)

    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()

    assert isinstance(app.received[0], SelectMenuWidget.Confirmed)
    assert app.received[0].value is None


@pytest.mark.anyio
async def test_select_menu_overlay_centers_content_sized_menu() -> None:
    """The overlay centers a menu without expanding it to the screen size."""
    menu = SelectMenuWidget(
        "Select voice",
        [SelectMenuOption("Celune", "balanced")],
        footer="ENTER apply",
    )
    app = _OverlayHarness(menu)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        overlay = app.screen
        assert isinstance(overlay, SelectMenuOverlay)
        assert menu.region.width < overlay.region.width
        assert menu.region.height < overlay.region.height
        assert menu.region.x > 0
        assert menu.region.y > 0
        assert abs(menu.region.center[0] - overlay.region.center[0]) <= 1
        assert abs(menu.region.center[1] - overlay.region.center[1]) <= 1
        await pilot.click(offset=(1, 1))
        assert app.underlying_clicks == 0
        assert overlay.styles.background.a == 0

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not overlay
        assert app.is_running


@pytest.mark.anyio
async def test_select_menu_keeps_configuration_rows_after_layout() -> None:
    """A long configuration menu must not collapse to its selected row."""
    menu = SelectMenuWidget(
        "Configuration manager",
        [SelectMenuOption(f"setting.{index}", index) for index in range(20)],
        footer_builder=lambda option: "UP/DOWN select ・ ENTER save ・ ESC cancel",
    )
    app = _OverlayHarness(menu)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.pause()
        rendered = menu.render().plain.splitlines()
        option_lines = [line for line in rendered if line.startswith(("   ", "-> "))]
        assert len(option_lines) == 15
