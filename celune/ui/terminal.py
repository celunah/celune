# SPDX-License-Identifier: Apache-2.0
"""Terminal UI helpers."""

import re
import sys
import logging
from dataclasses import dataclass, replace
from collections.abc import Callable, Collection, Mapping
from typing import Literal, Optional

import readchar
from rich.cells import cell_len
from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widget import Widget
from textual.dom import NoScreen

from ..typing.common import JSONSerializable


class SelectMenu:
    """A selection menu.

    Args:
        choices: Human-readable choice names.
        raw_choices: Internal choice values.
        prompt: Selection prompt to override the default one.
    """

    def __init__(
        self,
        choices: list[str],
        raw_choices: list[JSONSerializable],
        prompt: str = "Select an option",
    ) -> None:
        if not choices:
            raise ValueError("choices must not be empty")
        if len(choices) != len(raw_choices):
            raise ValueError("choices and raw_choices must have same length")

        self.choices = choices
        self.raw_choices = raw_choices
        self.prompt = prompt
        self.idx = 0

    def render(self) -> None:
        """Render available selections."""
        sys.stdout.write("\r")
        for n, choice in enumerate(self.choices):
            if n == self.idx:
                sys.stdout.write(f"\033[7m -> {choice} \033[0m\n")
            else:
                sys.stdout.write(f"    {choice} \n")

        sys.stdout.write(f"\033[{len(self.choices)}A")
        sys.stdout.flush()

    def start(self) -> JSONSerializable:
        """Start the selection menu.

        Returns:
            str: The chosen option.
        """
        sys.stdout.write("\033[?25l")
        sys.stdout.write(self.prompt)
        sys.stdout.write("\n\n")

        try:
            while True:
                self.render()
                key = readchar.readkey()
                if key == readchar.key.UP:
                    self.idx = (self.idx - 1) % len(self.choices)
                elif key == readchar.key.DOWN:
                    self.idx = (self.idx + 1) % len(self.choices)
                elif key == readchar.key.ENTER:
                    return self.raw_choices[self.idx]
        finally:
            sys.stdout.write(f"\033[{len(self.choices)}B")
            sys.stdout.write("\033[?25h\n")


@dataclass(frozen=True)
class SelectMenuOption:
    """Describe one row in a :class:`SelectMenuWidget`.

    Args:
        label: Text displayed on the left side of the row.
        value: Value displayed in brackets for an editable row.
        editable: Whether the row has a value and can be updated.
        keybind: Optional printable key that selects this row directly.
        autocomplete_values: Optional values that Left/Right can cycle and
            typing can search. Values keep their original JSON-compatible
            types when selected.
        display_value: Optional text to show in brackets instead of the
            returned value.
        show_value: Whether an editable value is rendered in brackets.
        confirm_value: Optional value returned for a non-editable row.
        explanation: Optional line shown above the footer hints.
    """

    label: str
    value: JSONSerializable = None
    editable: bool = True
    keybind: Optional[str] = None
    autocomplete_values: Optional[Collection[JSONSerializable]] = None
    display_value: Optional[str] = None
    show_value: bool = True
    confirm_value: Optional[JSONSerializable] = None
    explanation: Optional[str] = None


class SelectMenuWidget(Widget):
    """Render an interactive, value-aware selection menu in Textual.

    The widget keeps the existing ``SelectMenu`` row highlight convention: the
    selected row is rendered with reverse video. Values are aligned two cells
    after the longest label, and can be shown for every editable row or only
    for the selected row.

    Args:
        title: Header text rendered above the options.
        options: Rows that can be selected.
        footer: Optional hint or status text rendered below the options.
        footer_builder: Optional callback that creates a dynamic footer from
            the currently selected row.
        value_display: Show editable values for ``"all"`` rows or only the
            ``"current"`` row.
        return_value: Whether confirmation messages contain the selected row's
            value. Non-editable rows produce ``None`` unless they define a
            ``confirm_value``.
        keybinds: Optional mapping from a key to an option index. A keybind on
            an option is merged with this mapping.
        widget_id: Optional Textual widget id.

    Raises:
        ValueError: If no options are supplied, labels contain line breaks, or
            a keybind points to an invalid or duplicate option.
    """

    can_focus = True

    class Confirmed(Message):
        """Message emitted when the current option is confirmed."""

        def __init__(
            self,
            menu: "SelectMenuWidget",
            option_index: int,
            value: JSONSerializable,
        ) -> None:
            super().__init__()
            self.menu = menu
            self.option_index = option_index
            self.option = menu.options[option_index]
            self.value = value

    class Cancelled(Message):
        """Message emitted when the menu receives ESC."""

        def __init__(self, menu: "SelectMenuWidget", option_index: int) -> None:
            super().__init__()
            self.menu = menu
            self.option_index = option_index
            self.option = menu.options[option_index]
            self.value = None

    def __init__(
        self,
        title: str,
        options: Collection[SelectMenuOption],
        footer: Optional[str] = None,
        value_display: Literal["all", "current"] = "all",
        return_value: bool = True,
        keybinds: Optional[Mapping[str, int]] = None,
        widget_id: Optional[str] = None,
        footer_builder: Optional[Callable[[SelectMenuOption], Optional[str]]] = None,
    ) -> None:
        super().__init__(id=widget_id)
        self.title = title
        self.options = tuple(options)
        if not self.options:
            raise ValueError("options must not be empty")
        if value_display not in {"all", "current"}:
            raise ValueError("value_display must be 'all' or 'current'")

        if "\n" in title or "\r" in title:
            raise ValueError("title must be single-line")
        for option in self.options:
            if "\n" in option.label or "\r" in option.label:
                raise ValueError("option labels must be single-line")
        if footer is not None and ("\n" in footer or "\r" in footer):
            raise ValueError("footer must be single-line")

        self.footer = footer
        self.footer_builder = footer_builder
        self.value_display = value_display
        self.return_value = return_value
        self.selected_index = 0
        self._edit_buffer: Optional[str] = None
        self._keybinds = self._build_keybinds(keybinds)

    def render(self) -> Text:
        """Render the header, rows, and optional footer."""
        value_column = 3 + max(cell_len(option.label) for option in self.options) + 2
        content_width = (
            self.content_region.width
            if self.content_region.width > 0
            else self.size.width
        )
        rendered = Text(self.title)
        rendered.append("\n\n")

        footer = self._footer_text()
        visible_indices = self._visible_option_indices()
        for visible_index, index in enumerate(visible_indices):
            option = self.options[index]
            prefix = "-> " if index == self.selected_index else "   "
            line = Text(prefix + option.label)
            if self._shows_value(index, option):
                padding = value_column - cell_len(prefix + option.label)
                line.append(" " * max(0, padding))
                display_value = (
                    option.display_value
                    if option.display_value is not None
                    else self._value_text(option.value)
                )
                if content_width > 0:
                    display_value = self._truncate_cells(
                        display_value,
                        max(1, content_width - value_column - 4),
                    )
                line.append(f"[ {display_value} ]")
            if index == self.selected_index:
                line.stylize("reverse")
            rendered.append(line)
            if visible_index < len(visible_indices) - 1:
                rendered.append("\n")

        if footer is not None:
            rendered.append("\n\n")
            rendered.append(footer)
        return rendered

    def _footer_text(self) -> Optional[str]:
        """Return the static or selected-row-specific footer text."""
        footer = (
            self.footer_builder(self.selected_option)
            if self.footer_builder is not None
            else self.footer
        )
        explanation = self.selected_option.explanation
        if explanation and footer:
            return f"{explanation}\n{footer}"
        return explanation or footer

    def _visible_option_indices(self) -> tuple[int, ...]:
        """Return the option window that fits while keeping the footer visible."""
        # ``content_region.height`` can describe the current rendered content
        # while Textual is settling an auto-sized, scrollable widget. Using it
        # here creates a feedback loop where the second render may decide that
        # only the selected row fits. The screen is the stable viewport.
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            screen_height = 0
        viewport_height = screen_height or self.region.height or self.size.height
        if viewport_height <= 0:
            return tuple(range(len(self.options)))

        footer = self._footer_text()
        footer_lines = footer.count("\n") + 1 if footer is not None else 0
        fixed_lines = 2 + (2 + footer_lines if footer is not None else 0)
        option_capacity = max(1, viewport_height - fixed_lines - 4)
        if option_capacity >= len(self.options):
            return tuple(range(len(self.options)))

        start = min(
            max(0, self.selected_index - option_capacity // 2),
            len(self.options) - option_capacity,
        )
        return tuple(range(start, start + option_capacity))

    def on_key(self, event: Key) -> None:
        """Handle navigation, value editing, confirmation, and cancellation."""
        if event.key == "up":
            self._select((self.selected_index - 1) % len(self.options))
        elif event.key == "down":
            self._select((self.selected_index + 1) % len(self.options))
        elif event.key == "left":
            self._cycle_value(-1)
        elif event.key == "right":
            self._cycle_value(1)
        elif event.key == "backspace":
            self._edit_text(backspace=True)
        elif event.key == "escape":
            self.post_message(self.Cancelled(self, self.selected_index))
        elif event.key == "enter":
            value = self.selected_value if self.return_value else None
            self.post_message(self.Confirmed(self, self.selected_index, value))
        else:
            option_index = self._keybinds.get(event.key.casefold())
            if option_index is not None:
                self._select(option_index)
            elif event.character and event.character.isprintable():
                self._edit_text(character=event.character)
            else:
                return
        event.stop()

    def set_value(self, option_index: int, value: JSONSerializable) -> None:
        """Update an editable row and repaint the menu.

        Args:
            option_index: Zero-based row index to update.
            value: New value displayed in the row.

        Raises:
            IndexError: If ``option_index`` is outside the option list.
            ValueError: If the selected row is non-editable.
        """
        if option_index < 0 or option_index >= len(self.options):
            raise IndexError("option index is out of range")
        option = self.options[option_index]
        if not option.editable:
            raise ValueError("cannot set the value of a non-editable option")
        self._set_option_value(option_index, value)

    @property
    def selected_option(self) -> SelectMenuOption:
        """Return the currently highlighted row."""
        return self.options[self.selected_index]

    @property
    def selected_value(self) -> JSONSerializable:
        """Return the selected value, including an optional static-row result."""
        option = self.selected_option
        if option.editable:
            return option.value
        return option.confirm_value

    def _build_keybinds(self, keybinds: Optional[Mapping[str, int]]) -> dict[str, int]:
        """Validate and combine option and explicit keybinds."""
        bindings: dict[str, int] = {}
        for index, option in enumerate(self.options):
            if option.keybind is not None:
                self._add_keybind(bindings, option.keybind, index)
        if keybinds is not None:
            for key, option_index in keybinds.items():
                self._add_keybind(bindings, key, option_index)
        return bindings

    def _add_keybind(
        self,
        bindings: dict[str, int],
        key: str,
        option_index: int,
    ) -> None:
        """Add one normalized keybind after checking its target and uniqueness."""
        normalized_key = key.casefold()
        if not normalized_key:
            raise ValueError("keybinds must not contain empty keys")
        if option_index < 0 or option_index >= len(self.options):
            raise ValueError("keybind option index is out of range")
        if normalized_key in {
            "up",
            "down",
            "left",
            "right",
            "backspace",
            "escape",
            "enter",
        }:
            raise ValueError("navigation keys cannot be used as option keybinds")
        if normalized_key in bindings:
            raise ValueError(f"duplicate option keybind: {key}")
        bindings[normalized_key] = option_index

    def _select(self, option_index: int) -> None:
        """Highlight one option and request a repaint."""
        self.selected_index = option_index
        self._edit_buffer = None
        self.refresh()

    def _cycle_value(self, direction: int) -> None:
        """Cycle the selected editable row through its autocomplete values."""
        option = self.selected_option
        if not option.editable or not option.autocomplete_values:
            return

        values = tuple(option.autocomplete_values)
        current_index = next(
            (index for index, value in enumerate(values) if value == option.value),
            -1,
        )
        next_index = (current_index + direction) % len(values)
        self._set_option_value(self.selected_index, values[next_index])

    def _edit_text(
        self,
        *,
        character: Optional[str] = None,
        backspace: bool = False,
    ) -> None:
        """Edit or search the selected editable row's value."""
        option = self.selected_option
        if not option.editable:
            return

        if self._edit_buffer is None:
            self._edit_buffer = (
                "" if character is not None else self._value_text(option.value)
            )
        if character is not None:
            self._edit_buffer += character
        elif backspace:
            self._edit_buffer = self._edit_buffer[:-1]

        query = self._edit_buffer
        matches = self._matching_values(option, query)
        reset_editor = False
        if matches:
            value = matches[0]
        elif option.autocomplete_values is not None:
            value = option.value
            reset_editor = True
        else:
            value = self._coerce_text_value(query, option.value)
            reset_editor = not self._is_valid_text_value(query, option.value)
        self._set_option_value(
            self.selected_index,
            value,
            reset_editor=reset_editor,
        )

    @staticmethod
    def _matching_values(
        option: SelectMenuOption,
        query: str,
    ) -> tuple[JSONSerializable, ...]:
        """Return autocomplete candidates matching the current query."""
        if not option.autocomplete_values:
            return ()
        normalized_query = query.casefold()
        if isinstance(option.value, bool):
            if not normalized_query:
                return ()
            return tuple(
                value
                for value in option.autocomplete_values
                if SelectMenuWidget._value_text(value)
                .casefold()
                .startswith(normalized_query)
            )
        return tuple(
            value
            for value in option.autocomplete_values
            if normalized_query in SelectMenuWidget._value_text(value).casefold()
        )

    @staticmethod
    def _coerce_text_value(text: str, current: JSONSerializable) -> JSONSerializable:
        """Preserve simple scalar return types while editing text."""
        normalized = text.casefold().strip()
        if isinstance(current, bool):
            if normalized in {"true", "false"}:
                return normalized == "true"
            return current
        if current is None and normalized == "none":
            return None
        if isinstance(current, int):
            try:
                return int(text)
            except ValueError:
                return current
        if isinstance(current, float):
            try:
                return float(text)
            except ValueError:
                return current
        return text

    @staticmethod
    def _is_valid_text_value(text: str, current: JSONSerializable) -> bool:
        """Return whether text can be represented by the current scalar type."""
        normalized = text.casefold().strip()
        if isinstance(current, bool):
            return normalized in {"true", "false"}
        if current is None:
            return normalized == "none"
        if isinstance(current, int):
            try:
                int(text)
            except ValueError:
                return False
            return True
        if isinstance(current, float):
            try:
                float(text)
            except ValueError:
                return False
            return True
        return True

    def _set_option_value(
        self,
        option_index: int,
        value: JSONSerializable,
        *,
        reset_editor: bool = True,
    ) -> None:
        """Replace an editable row value and repaint the widget."""
        option = self.options[option_index]
        updated = replace(option, value=value)
        self.options = (
            *self.options[:option_index],
            updated,
            *self.options[option_index + 1 :],
        )
        if reset_editor:
            self._edit_buffer = None
        self.refresh()

    def _shows_value(self, index: int, option: SelectMenuOption) -> bool:
        """Return whether one option should render its bracketed value."""
        if not option.editable:
            return False
        return option.show_value and (
            self.value_display == "all" or index == self.selected_index
        )

    @staticmethod
    def _value_text(value: JSONSerializable) -> str:
        """Convert a JSON-compatible option value into one display cell."""
        return str(value).replace("\n", " ").replace("\r", " ")

    @staticmethod
    def _truncate_cells(value: str, max_cells: int) -> str:
        """Shorten a display value without splitting its terminal cell budget."""
        if max_cells <= 0:
            return ""
        if cell_len(value) <= max_cells:
            return value

        ellipsis = "…"
        target_cells = max(0, max_cells - cell_len(ellipsis))
        truncated = ""
        for character in value:
            if cell_len(truncated + character) > target_cells:
                break
            truncated += character
        return truncated + ellipsis


class LogRedirect:
    """Redirect logs to the logger."""

    def __init__(
        self,
        stdout,
        stderr,
        write_callback: Callable[[str, str], None],
        default_severity: str = "info",
        filter_messages: Optional[Collection[str]] = None,
    ) -> None:
        self.write_callback = write_callback
        self.default_severity = default_severity
        self._buffer = ""
        self.underlying_stdout = stdout
        self.underlying_stderr = stderr
        self.filter_messages = (
            filter_messages  # these messages will be filtered out by the logger
        )

    def _is_filtered_message(self, message: str) -> bool:
        """Return whether one redirected message should be suppressed."""
        if self.filter_messages is None:
            return False

        return any(
            filtered_message in message for filtered_message in self.filter_messages
        )

    @staticmethod
    def _severity_for_message(message: str, default_severity: str) -> str:
        """Infer log severity from one redirected text line."""
        lowered = message.casefold()

        if "[error]" in lowered:
            return "error"
        if "[warning]" in lowered:
            return "warning"
        if "traceback (most recent call last):" in lowered:
            return "error"
        if re.search(
            r"\b(?:error|exception|fatal(?: error)?)\b",
            lowered,
        ):
            return "error"
        if re.search(
            (
                r"\b(?:warning|futurewarning|deprecationwarning|"
                r"pendingdeprecationwarning|runtimewarning|resourcewarning|"
                r"userwarning|syntaxwarning|importwarning|unicodewarning|"
                r"byteswarning)\b"
            ),
            lowered,
        ):
            return "warning"
        return default_severity

    def write(self, text: str) -> None:
        """Write text to the logger.

        Args:
            text: The raw text chunk captured from redirected output.
        """
        if not text:
            return

        # strip any incoming ANSI, but keep TTY specific input
        ansi_regex = re.compile(
            r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
        )
        text = re.sub(ansi_regex, "", text)

        self._buffer += text

        while "\n" in self._buffer or "\r" in self._buffer:
            newline_pos = self._buffer.find("\n") if "\n" in self._buffer else 10**9
            cr_pos = self._buffer.find("\r") if "\r" in self._buffer else 10**9
            pos = min(newline_pos, cr_pos)

            chunk = self._buffer[:pos].strip()
            self._buffer = self._buffer[pos + 1 :]

            if chunk and not self._is_filtered_message(chunk):
                self.write_callback(
                    chunk,
                    self._severity_for_message(chunk, self.default_severity),
                )

    def ansi(self, escape: str) -> None:
        """Write ANSI escape code(s) to the terminal directly.

        Args:
            escape: The ANSI escape code(s) to process.
        """
        ansi_regex = re.compile(
            r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
        )
        any_ansi = re.findall(ansi_regex, escape)
        if any_ansi:
            escapes = "".join(any_ansi)
            self.underlying_stdout.write(escapes)
            self.underlying_stdout.flush()

    def flush(self) -> None:
        """Flush the buffers."""
        if self._buffer.strip():
            chunk = self._buffer.strip()
            if not self._is_filtered_message(chunk):
                self.write_callback(
                    chunk,
                    self._severity_for_message(chunk, self.default_severity),
                )
        self._buffer = ""

    def isatty(self) -> bool:
        """Return if the underlying terminal is a TTY.

        Returns:
            bool: Whether the underlying terminal is a TTY.
        """
        return self.underlying_stdout.isatty()


class UILogHandler(logging.Handler):
    """Route Python logging records into Celune's UI log callback."""

    def __init__(
        self,
        write_callback: Callable[[str, str], None],
        filter_messages: Optional[Collection[str]] = None,
    ) -> None:
        super().__init__()
        self.write_callback = write_callback
        self.filter_messages = filter_messages

    def _is_filtered_message(self, message: str) -> bool:
        """Return whether one logging message should be suppressed."""
        if self.filter_messages is None:
            return False

        return any(
            filtered_message in message for filtered_message in self.filter_messages
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one Python logging record into the UI log stream.

        Args:
            record: The logging record to be emitted.
        """
        # noinspection PyBroadException
        try:
            message = record.getMessage().strip()
        except Exception:
            self.handleError(record)
            return

        if not message:
            return

        if (
            "triton not found; flop counting will not work for triton kernels"
            in message
        ):
            message = "triton not found; flop counting will not work for triton kernels"

        if self._is_filtered_message(message):
            return

        if record.levelno >= logging.ERROR:
            severity = "error"
            prefix = "Internal runtime error:"
        elif record.levelno >= logging.WARNING:
            severity = "warning"
            prefix = "Internal runtime warning:"
        else:
            severity = "info"
            prefix = "Internal runtime notice:"

        self.write_callback(f"{prefix} {message}", severity)


def is_celune_log_record(record: logging.LogRecord) -> bool:
    """Return whether a logging record belongs to Celune itself.

    Args:
        record: The logging record to classify.

    Returns:
        bool: ``True`` when the record originated from Celune loggers.
    """
    return record.name == "celune" or record.name.startswith("celune.")
