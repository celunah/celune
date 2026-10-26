# SPDX-License-Identifier: Apache-2.0
"""Read-only runtime inspector for the Textual debug view."""

from __future__ import annotations

import sys
import time
import types
import contextlib
import threading

from enum import Enum
from pathlib import Path
from collections import deque
from datetime import datetime
from types import CodeType, ModuleType
from dataclasses import dataclass, field
from collections.abc import Callable, Iterator
from typing import ClassVar, Optional, Protocol, cast

from rich.text import Text
from textual.app import ComposeResult
from textual import binding, containers, screen, widgets

from ..i18n import string

_CELUNE_SOURCE_ROOT = Path(__file__).resolve().parent.parent


class _MonitorEvents(Protocol):
    """Event constants used by the optional execution monitor."""

    PY_START: int
    PY_RESUME: int
    PY_RETURN: int
    PY_YIELD: int
    PY_UNWIND: int
    NO_EVENTS: int


class _MonitoringAPI(Protocol):
    """Small typed surface of Python's execution monitoring API."""

    events: _MonitorEvents

    def get_tool(self, tool_id: int, /) -> Optional[str]: ...

    def use_tool_id(self, tool_id: int, name: str, /) -> None: ...

    def free_tool_id(self, tool_id: int, /) -> None: ...

    def register_callback(
        self,
        tool_id: int,
        event: int,
        callback: Optional[Callable[..., object]],
        /,
    ) -> Optional[Callable[..., object]]: ...

    def set_events(self, tool_id: int, event_set: int, /) -> None: ...

    def set_local_events(
        self,
        tool_id: int,
        code: CodeType,
        event_set: int,
        /,
    ) -> None: ...


@dataclass
class DebugCallEntry:
    """One observed Celune function activation."""

    entry_id: int
    timestamp: str
    name: str
    call_path: tuple[str, ...]
    thread_id: int
    source_path: str
    source_line: int
    started_ns: int
    duration_ms: Optional[float] = None
    result: str = "running"
    exception_type: Optional[str] = None


@dataclass
class _ActiveCall:
    """A function activation tracked on one executing thread."""

    code: CodeType
    name: str
    started_ns: int
    entry: Optional[DebugCallEntry]


class DebugCallTrace:
    """Capture bounded calls to loaded Celune Python functions."""

    _MAX_CALL_DEPTH = 24
    _TRACE_LIMIT = 500
    _EVENTS: ClassVar[tuple[str, ...]] = (
        "PY_START",
        "PY_RESUME",
        "PY_RETURN",
        "PY_YIELD",
        "PY_UNWIND",
    )

    def __init__(self, ui: object) -> None:
        self._ui = ui
        self._monitoring = cast(
            Optional[_MonitoringAPI], getattr(sys, "monitoring", None)
        )
        self._tool_id: Optional[int] = None
        self._labels: dict[CodeType, str] = {}
        self._entries: deque[DebugCallEntry] = deque(maxlen=self._TRACE_LIMIT)
        self._stacks: dict[int, list[_ActiveCall]] = {}
        self._lock = threading.RLock()
        self._next_entry_id = 0
        self._recording = True
        self._enabled = False
        self._failed = False

    @property
    def available(self) -> bool:
        """Return whether an execution monitor is active."""
        return self._tool_id is not None and not self._failed

    @property
    def paused(self) -> bool:
        """Return whether new call entries are currently being recorded."""
        return not self._recording

    def sync_enabled(self) -> None:
        """Keep call monitoring aligned with the active debug log level."""
        self.set_enabled(_is_debug_enabled(self._ui))

    def set_enabled(self, enabled: bool) -> None:
        """Start or stop recording as debug logging is enabled or disabled."""
        if not enabled:
            if self._enabled:
                self._enabled = False
                self.stop()
                self.clear()
                self._recording = True
            return

        if not self._enabled:
            self._enabled = True
            self._failed = False
            self._recording = True
            self.clear()
        if self.available:
            self.refresh_code_objects()
            return
        if self._tool_id is not None:
            self.stop()
        self._failed = False
        self.start()

    def start(self) -> bool:
        """Install per-code call monitoring for the loaded Celune modules."""
        if self._monitoring is None:
            return False
        if self._tool_id is not None:
            return not self._failed
        self._failed = False

        tool_id = next(
            (
                candidate
                for candidate in range(6)
                if self._monitoring.get_tool(candidate) is None
            ),
            None,
        )
        if tool_id is None:
            return False

        try:
            self._monitoring.use_tool_id(tool_id, "celune-debug-inspector")
            self._tool_id = tool_id
            callbacks = {
                "PY_START": self._on_start,
                "PY_RESUME": self._on_start,
                "PY_RETURN": self._on_return,
                "PY_YIELD": self._on_return,
                "PY_UNWIND": self._on_unwind,
            }
            for event_name, callback in callbacks.items():
                event = getattr(self._monitoring.events, event_name)
                self._monitoring.register_callback(tool_id, event, callback)
            self._monitoring.set_events(
                tool_id,
                self._monitoring.events.PY_UNWIND,
            )
            self.refresh_code_objects()
        except Exception:
            self._failed = True
            self.stop()
            return False
        if not self.available:
            self.stop()
            return False
        return self.available

    def stop(self) -> None:
        """Remove this monitor's callbacks and per-code events."""
        tool_id = self._tool_id
        monitoring = self._monitoring
        self._tool_id = None
        if tool_id is None or monitoring is None:
            return

        with self._lock:
            codes = tuple(self._labels)
            self._labels.clear()
            self._stacks.clear()
        for code in codes:
            with contextlib.suppress(Exception):
                monitoring.set_local_events(tool_id, code, monitoring.events.NO_EVENTS)
        with contextlib.suppress(Exception):
            monitoring.set_events(tool_id, monitoring.events.NO_EVENTS)
        for event_name in self._EVENTS:
            with contextlib.suppress(Exception):
                monitoring.register_callback(
                    tool_id,
                    getattr(monitoring.events, event_name),
                    None,
                )
        with contextlib.suppress(Exception):
            monitoring.free_tool_id(tool_id)

    def refresh_code_objects(self) -> None:
        """Enable monitoring for newly imported Celune modules."""
        tool_id = self._tool_id
        monitoring = self._monitoring
        if tool_id is None or monitoring is None or self._failed:
            return

        local_events = (
            monitoring.events.PY_START
            | monitoring.events.PY_RESUME
            | monitoring.events.PY_RETURN
            | monitoring.events.PY_YIELD
        )
        try:
            discovered: dict[CodeType, str] = {}
            seen_codes: set[CodeType] = set()
            for module_name, module in tuple(sys.modules.items()):
                if (
                    not module_name.startswith("celune.")
                    or module_name == "celune.ui.debug"
                    or not isinstance(module, ModuleType)
                ):
                    continue
                for value in vars(module).values():
                    if (
                        isinstance(value, (types.FunctionType, type))
                        and value.__module__ != module_name
                    ):
                        continue
                    for code in _codes_from_value(value, seen_codes):
                        if not _is_celune_code(code):
                            continue
                        discovered.setdefault(
                            code,
                            f"{module_name}.{code.co_qualname}",
                        )

            for code, name in discovered.items():
                if code in self._labels:
                    continue
                monitoring.set_local_events(tool_id, code, local_events)
                self._labels[code] = name
        except Exception:
            self._failed = True
            self.stop()

    def set_paused(self, paused: bool) -> None:
        """Pause or resume recording while keeping call stacks coherent."""
        self._recording = not paused

    def clear(self) -> None:
        """Remove all visible entries from the rolling trace."""
        with self._lock:
            self._entries.clear()

    def entries(self) -> tuple[DebugCallEntry, ...]:
        """Return a stable snapshot of the rolling call buffer."""
        with self._lock:
            return tuple(self._entries)

    def _on_start(self, code: CodeType, instruction_offset: int) -> None:
        del instruction_offset
        if self._failed:
            return
        name = self._labels.get(code)
        if name is None:
            return

        started_ns = time.perf_counter_ns()
        thread_id = threading.get_ident()
        try:
            with self._lock:
                stack = self._stacks.setdefault(thread_id, [])
                path = tuple(call.name for call in stack) + (name,)
                entry: Optional[DebugCallEntry] = None
                if self._recording:
                    self._next_entry_id += 1
                    entry = DebugCallEntry(
                        entry_id=self._next_entry_id,
                        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        name=name,
                        call_path=path[-self._MAX_CALL_DEPTH :],
                        thread_id=thread_id,
                        source_path=code.co_filename,
                        source_line=code.co_firstlineno,
                        started_ns=started_ns,
                    )
                    self._entries.append(entry)
                stack.append(_ActiveCall(code, name, started_ns, entry))
        except Exception:
            self._failed = True

    def _on_return(
        self,
        code: CodeType,
        instruction_offset: int,
        return_value: object,
    ) -> None:
        del instruction_offset, return_value
        self._finish_call(code, "returned", None)

    def _on_unwind(
        self,
        code: CodeType,
        instruction_offset: int,
        exception: BaseException,
    ) -> None:
        del instruction_offset
        self._finish_call(code, "raised", type(exception).__name__)

    def _finish_call(
        self,
        code: CodeType,
        result: str,
        exception_type: Optional[str],
    ) -> None:
        if self._failed:
            return
        thread_id = threading.get_ident()
        finished_ns = time.perf_counter_ns()
        try:
            with self._lock:
                stack = self._stacks.get(thread_id)
                if not stack:
                    return
                matching_index = next(
                    (
                        index
                        for index in range(len(stack) - 1, -1, -1)
                        if stack[index].code is code
                    ),
                    None,
                )
                if matching_index is None:
                    return
                completed = stack[matching_index:]
                del stack[matching_index:]
                if not stack:
                    self._stacks.pop(thread_id, None)
                for active in completed:
                    entry = active.entry
                    if entry is None:
                        continue
                    entry.duration_ms = (finished_ns - active.started_ns) / 1_000_000
                    entry.result = result
                    entry.exception_type = exception_type
        except Exception:
            self._failed = True


@dataclass
class _InspectorComponent:
    """One selectable component backed by an in-process object."""

    path: str
    label: str
    value: object
    children: list[_InspectorComponent] = field(default_factory=list)


_UI_MODULE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("celune.ui.runtime", "debug.component_ui_runtime"),
    ("celune.ui.commands", "debug.component_ui_commands"),
    ("celune.ui.capture", "debug.component_ui_capture"),
    ("celune.ui.interaction", "debug.component_ui_interaction"),
)
_ENGINE_MODULE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("celune.core.lifecycle", "debug.component_lifecycle"),
    ("celune.core.speech", "debug.component_speech"),
    ("celune.pipeline", "debug.component_pipeline"),
    ("celune.pipeline.submission", "debug.component_pipeline_submission"),
    ("celune.pipeline.generation", "debug.component_pipeline_generation"),
    ("celune.pipeline.buffering", "debug.component_pipeline_buffering"),
    ("celune.pipeline.playback", "debug.component_pipeline_playback"),
    ("celune.pipeline.output", "debug.component_pipeline_output"),
    ("celune.pipeline.captions", "debug.component_pipeline_captions"),
    ("celune.pipeline.timing", "debug.component_pipeline_timing"),
    ("celune.audio.server", "debug.component_audio_server"),
    ("celune.cedts.worker", "debug.component_worker"),
)
_COMPONENT_ACRONYMS = frozenset(
    {
        "api",
        "asr",
        "cechar",
        "cedts",
        "cevoice",
        "cli",
        "cpu",
        "cuda",
        "dsp",
        "gpu",
        "http",
        "https",
        "ipa",
        "json",
        "kv",
        "llm",
        "onnx",
        "rest",
        "rms",
        "tts",
        "tui",
        "ui",
        "url",
        "uuid",
        "vc",
        "vad",
        "vram",
        "webui",
        "yaml",
    }
)


class DebugInspectorScreen(screen.ModalScreen[None]):
    """Show live Celune and UI state in a debug-only modal."""

    BINDINGS: ClassVar[list[binding.BindingType]] = [
        ("escape", "close_inspector", "Close inspector"),
        ("f", "focus_filter", "Filter components"),
    ]
    DEFAULT_CSS = """
    DebugInspectorScreen {
        align: center middle;
        background: $background 92%;
        color: $foreground;
    }
    #debug-panel {
        width: 98%;
        height: 96%;
        border: round $primary;
        background: $background;
        padding: 0 2;
    }
    #debug-heading {
        height: 2;
        border-bottom: solid $secondary;
        margin-bottom: 1;
    }
    #debug-status {
        width: 1fr;
        height: 1;
        color: $foreground;
        text-align: left;
    }
    #debug-main {
        height: 1fr;
        min-height: 6;
    }
    #debug-components-pane {
        width: 30%;
        min-width: 22;
        border: round $secondary;
        padding: 1;
        margin-right: 1;
    }
    #debug-components-title,
    #debug-state-title,
    #debug-attributes-title,
    #debug-live-title,
    #debug-call-detail-title {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    #debug-components-title {
        margin-bottom: 1;
    }
    #debug-filter {
        height: 3;
        border: round $secondary;
        margin: 0 0 1 0;
    }
    #debug-components {
        height: 1fr;
        padding: 0;
    }
    #debug-details-pane {
        width: 1fr;
        border: round $secondary;
        padding: 1 2;
    }
    #debug-selected-component {
        height: 1;
        color: $foreground;
        text-style: bold;
        margin-bottom: 1;
    }
    #debug-state-title {
        margin-top: 0;
    }
    #debug-state-table {
        height: 1;
        min-height: 1;
        max-height: 8;
    }
    #debug-attributes-title {
        margin-top: 1;
    }
    #debug-attributes-table {
        height: 1fr;
        min-height: 3;
    }
    #debug-live-pane {
        height: 9;
        min-height: 7;
        border: round $secondary;
        margin-top: 1;
        padding: 0 1;
    }
    #debug-live-heading {
        height: 3;
        align: left middle;
    }
    #debug-live-title {
        width: 1fr;
    }
    #debug-pause,
    #debug-clear {
        min-width: 10;
        height: 3;
        padding: 0 1;
        border: round $primary;
        background: $surface;
        color: $primary;
        margin-left: 1;
    }
    #debug-live-content {
        height: 1fr;
        min-height: 4;
    }
    #debug-call-table {
        width: 68%;
        height: 1fr;
        border-right: solid $secondary;
    }
    #debug-call-detail {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    #debug-footer {
        height: 2;
        width: 1fr;
        color: $secondary;
        border-top: solid $secondary;
        padding: 0 1;
    }
    """

    def __init__(self, ui: object, trace: DebugCallTrace) -> None:
        super().__init__()
        self.ui = ui
        self.trace = trace
        self._selected_component: Optional[_InspectorComponent] = None
        self._component_signature: tuple[tuple[str, int], ...] = ()
        self._line_by_path: dict[str, int] = {}
        self._selected_call_id: Optional[int] = None
        self._shown_trace_signature: tuple[tuple[int, Optional[float], str], ...] = ()
        self._shown_trace_ids: tuple[int, ...] = ()
        self._shown_call_detail_id: Optional[int] = None
        self._shown_call_detail: Optional[str] = None
        self._rendered_table_values: dict[str, tuple[tuple[str, str], ...]] = {}
        self._rendered_table_components: dict[str, str] = {}
        self._rendered_state_table_height = 1
        self._component_tree: Optional[widgets.Tree[_InspectorComponent]] = None

    def compose(self) -> ComposeResult:
        """Build the component selector, snapshot, and call trace panes."""
        with containers.Vertical(id="debug-panel"):
            with containers.Horizontal(id="debug-heading"):
                yield widgets.Static("", id="debug-status")
            with containers.Horizontal(id="debug-main"):
                with containers.Vertical(id="debug-components-pane"):
                    yield widgets.Static(
                        string("debug.components"), id="debug-components-title"
                    )
                    yield widgets.Input(
                        placeholder=string("debug.filter_placeholder"),
                        id="debug-filter",
                    )
                    tree: widgets.Tree[_InspectorComponent] = widgets.Tree(
                        string("debug.components"),
                        id="debug-components",
                    )
                    tree.show_root = False
                    self._component_tree = tree
                    yield tree
                with containers.Vertical(id="debug-details-pane"):
                    yield widgets.Static("", id="debug-selected-component")
                    yield widgets.Static(string("debug.state"), id="debug-state-title")
                    yield widgets.DataTable(
                        cursor_type="row",
                        show_cursor=False,
                        show_header=False,
                        show_row_labels=False,
                        id="debug-state-table",
                    )
                    yield widgets.Static(
                        string("debug.attributes"), id="debug-attributes-title"
                    )
                    yield widgets.DataTable(
                        cursor_type="row",
                        show_cursor=False,
                        show_header=False,
                        show_row_labels=False,
                        id="debug-attributes-table",
                    )
            with containers.Vertical(id="debug-live-pane"):
                with containers.Horizontal(id="debug-live-heading"):
                    yield widgets.Static(
                        string("debug.live_calls"), id="debug-live-title"
                    )
                    yield widgets.Button(
                        string("debug.pause"), id="debug-pause", compact=True
                    )
                    yield widgets.Button(
                        string("debug.clear"), id="debug-clear", compact=True
                    )
                with containers.Horizontal(id="debug-live-content"):
                    yield widgets.DataTable(
                        cursor_type="row",
                        show_header=True,
                        show_row_labels=False,
                        id="debug-call-table",
                    )
                    with containers.Vertical(id="debug-call-detail"):
                        yield widgets.Static(
                            string("debug.call_details"),
                            id="debug-call-detail-title",
                        )
                        yield widgets.Static(
                            string("debug.select_call"),
                            id="debug-call-detail-body",
                        )
            yield widgets.Static(
                string("debug.footer"),
                id="debug-footer",
            )

    def on_mount(self) -> None:
        """Initialize inspector tables and populate their first snapshots."""
        table = self.query_one("#debug-state-table", widgets.DataTable)
        table.add_columns(
            (string("debug.field"), "debug-field"),
            (string("debug.value"), "debug-value"),
        )
        table = self.query_one("#debug-attributes-table", widgets.DataTable)
        table.add_columns(
            (string("debug.field"), "debug-field"),
            (string("debug.value"), "debug-value"),
        )
        table = self.query_one("#debug-call-table", widgets.DataTable)
        table.add_columns(
            string("debug.time"),
            string("debug.call"),
            (string("debug.duration"), "debug-duration"),
        )
        self._refresh_snapshot()
        self._refresh_live_calls()
        self.set_interval(0.25, self._refresh_snapshot)
        self.set_interval(0.25, self._refresh_live_calls)

    def on_tree_node_selected(
        self,
        event: widgets.Tree.NodeSelected[_InspectorComponent],
    ) -> None:
        """Show the snapshot for the component selected in the tree."""
        component = event.node.data
        if component is None:
            return
        self._selected_component = component
        self._render_component()

    def on_input_changed(self, event: widgets.Input.Changed) -> None:
        """Filter component names while preserving the selected path."""
        if event.input.id == "debug-filter":
            self._populate_tree(event.value)

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Apply the live-trace pause and clear controls."""
        if event.button.id == "debug-pause":
            self.trace.set_paused(not self.trace.paused)
            self._update_pause_button()
        elif event.button.id == "debug-clear":
            self.trace.clear()
            self._selected_call_id = None
            self._refresh_live_calls()

    def on_data_table_row_highlighted(
        self,
        event: widgets.DataTable.RowHighlighted,
    ) -> None:
        """Show the call path for the highlighted trace entry."""
        if event.data_table.id != "debug-call-table":
            return
        row_key = event.row_key.value
        try:
            self._selected_call_id = int(row_key) if row_key is not None else None
        except (TypeError, ValueError):
            self._selected_call_id = None
        self._render_call_detail()

    def action_close_inspector(self) -> None:
        """Close the modal and stop its call trace."""
        self.app.pop_screen()

    def action_focus_filter(self) -> None:
        """Focus the component filter input."""
        self.query_one("#debug-filter", widgets.Input).focus()

    def _refresh_snapshot(self) -> None:
        if not _is_debug_enabled(self.ui):
            self.app.pop_screen()
            return
        roots = self._build_component_roots()
        signature = _component_signature(roots)
        if signature != self._component_signature:
            self._component_signature = signature
            self._populate_tree(
                self.query_one("#debug-filter", widgets.Input).value,
                roots=roots,
            )
        elif self._selected_component is not None:
            for component in _iter_components(roots):
                if component.path == self._selected_component.path:
                    self._selected_component = component
                    break
        self.query_one("#debug-status", widgets.Static).update(self._format_status())
        self._render_component()
        self._update_pause_button()

    def _refresh_live_calls(self) -> None:
        table = self.query_one("#debug-call-table", widgets.DataTable)
        entries = self.trace.entries()
        ids = tuple(entry.entry_id for entry in entries)
        signature = tuple(
            (entry.entry_id, entry.duration_ms, entry.result) for entry in entries
        )
        previous_ids = self._shown_trace_ids
        previous_id_set = set(previous_ids)
        previous_signature = {
            entry_id: (duration, result)
            for entry_id, duration, result in self._shown_trace_signature
        }
        previous_selection = self._selected_call_id
        was_at_bottom = table.is_vertical_scroll_end
        removed_ids = tuple(
            entry_id for entry_id in previous_ids if entry_id not in ids
        )
        added_ids = tuple(
            entry_id for entry_id in ids if entry_id not in previous_id_set
        )

        for entry_id in removed_ids:
            table.remove_row(str(entry_id))

        for entry in entries:
            row_key = str(entry.entry_id)
            if entry.entry_id not in previous_id_set:
                table.add_row(
                    entry.timestamp,
                    entry.name,
                    _format_duration(entry),
                    key=row_key,
                )
            elif (
                previous_signature.get(entry.entry_id)
                != (
                    entry.duration_ms,
                    entry.result,
                )
                or entry.result == "running"
            ):
                table.update_cell(
                    row_key,
                    "debug-duration",
                    _format_duration(entry),
                )

        self._shown_trace_signature = signature
        self._shown_trace_ids = ids
        if not ids:
            self._selected_call_id = None
        elif previous_selection not in ids:
            if previous_selection is None or not previous_ids or was_at_bottom:
                self._selected_call_id = ids[-1]
            else:
                previous_row = (
                    previous_ids.index(previous_selection)
                    if previous_selection in previous_ids
                    else 0
                )
                self._selected_call_id = ids[min(previous_row, len(ids) - 1)]
        elif added_ids and was_at_bottom:
            self._selected_call_id = ids[-1]

        selection_changed = self._selected_call_id != previous_selection
        if self._selected_call_id is not None and (selection_changed or removed_ids):
            table.move_cursor(
                row=ids.index(self._selected_call_id),
                column=0,
                animate=False,
                scroll=was_at_bottom and bool(added_ids),
            )
        self._render_call_detail()

    def _render_call_detail(self) -> None:
        body = self.query_one("#debug-call-detail-body", widgets.Static)
        entry = next(
            (
                candidate
                for candidate in self.trace.entries()
                if candidate.entry_id == self._selected_call_id
            ),
            None,
        )
        if (
            entry is not None
            and entry.duration_ms is None
            and self._shown_call_detail_id == entry.entry_id
        ):
            return
        if entry is None:
            content = string("debug.select_call")
        else:
            lines = [
                f"{string('debug.call_path')}: {' → '.join(entry.call_path)}",
                f"{string('debug.duration')}: {_format_duration(entry)}",
                f"{string('debug.thread')}: {entry.thread_id}",
                f"{string('debug.source')}: {entry.source_path}:{entry.source_line}",
            ]
            if entry.exception_type is not None:
                lines.append(f"{string('debug.exception')}: {entry.exception_type}")
            content = "\n".join(lines)
        if content == self._shown_call_detail:
            return

        body.update(Text(content, no_wrap=False) if entry is not None else content)
        self._shown_call_detail = content
        self._shown_call_detail_id = self._selected_call_id

    def _build_component_roots(self) -> list[_InspectorComponent]:
        ui_component = _InspectorComponent(
            "ui",
            string("debug.component_ui"),
            self.ui,
        )
        ui_attributes = _object_attributes(self.ui)
        ui_state_names = (
            ("_binding_state", "debug.component_binding"),
            ("_widgets", "debug.component_widgets"),
            ("_theme_state", "debug.component_theme"),
            ("_interaction_state", "debug.component_interaction"),
            ("_log_capture_state", "debug.component_log_capture"),
        )
        for attribute, label_key in ui_state_names:
            value = ui_attributes.get(attribute)
            if value is not None:
                ui_component.children.append(
                    _InspectorComponent(
                        path=f"ui.{attribute}",
                        label=string(label_key),
                        value=value,
                    )
                )
        for module_name, label_key in _UI_MODULE_COMPONENTS:
            module = sys.modules.get(module_name)
            if isinstance(module, ModuleType):
                ui_component.children.append(
                    _InspectorComponent(
                        path=module_name,
                        label=string(label_key),
                        value=module,
                    )
                )
        ui_component.children.sort(key=lambda component: component.label.casefold())

        roots = [ui_component]
        engine = getattr(self.ui, "celune", None)
        if engine is None:
            return roots

        engine_component = _InspectorComponent(
            "celune",
            string("debug.component_engine"),
            engine,
        )
        for name, value in sorted(
            _object_attributes(engine).items(),
            key=lambda item: item[0].casefold(),
        ):
            if name == "__dict__" or _is_sensitive_name(name):
                continue
            if not _is_component_object(value):
                continue
            child_name = _component_label(name)
            engine_component.children.append(
                _InspectorComponent(
                    path=f"celune.{name}",
                    label=child_name or name,
                    value=value,
                )
            )
        for module_name, label_key in _ENGINE_MODULE_COMPONENTS:
            module = sys.modules.get(module_name)
            if isinstance(module, ModuleType):
                engine_component.children.append(
                    _InspectorComponent(
                        path=module_name,
                        label=string(label_key),
                        value=module,
                    )
                )
        engine_component.children.sort(key=lambda component: component.label.casefold())
        roots.append(engine_component)
        return roots

    def _populate_tree(
        self,
        filter_text: str,
        *,
        roots: Optional[list[_InspectorComponent]] = None,
    ) -> None:
        tree = self._component_tree
        if tree is None:
            return
        if roots is None:
            roots = self._build_component_roots()
        selected_path = (
            self._selected_component.path
            if self._selected_component is not None
            else "ui"
        )
        needle = filter_text.strip().casefold()
        tree.clear()
        line_by_path: dict[str, int] = {}
        cursor_line = 0
        for root in roots:
            root_matches = not needle or needle in root.label.casefold()
            matching_children = [
                child
                for child in root.children
                if root_matches or not needle or needle in child.label.casefold()
            ]
            if needle and not root_matches and not matching_children:
                continue
            node = tree.root.add(
                Text(root.label),
                root,
                expand=bool(matching_children),
                allow_expand=bool(matching_children),
            )
            line_by_path[root.path] = cursor_line
            cursor_line += 1
            for child in matching_children:
                node.add(
                    Text(child.label),
                    child,
                    allow_expand=False,
                )
                line_by_path[child.path] = cursor_line
                cursor_line += 1
        tree.root.expand()
        self._line_by_path = line_by_path
        if selected_path not in line_by_path:
            selected_path = next(iter(line_by_path), "")
        if selected_path:
            tree.cursor_line = line_by_path[selected_path]
            selected = next(
                (
                    component
                    for component in _iter_components(roots)
                    if component.path == selected_path
                ),
                None,
            )
            self._selected_component = selected
        elif needle:
            self._selected_component = None
            self.query_one("#debug-selected-component", widgets.Static).update(
                string("debug.filter_empty")
            )
        self._render_component()

    def _render_component(self) -> None:
        component = self._selected_component
        if component is None:
            return
        self.query_one("#debug-selected-component", widgets.Static).update(
            Text(component.label)
        )
        attributes = _object_attributes(component.value)
        states: list[tuple[str, str]] = []
        other_attributes: list[tuple[str, str]] = []
        if not attributes and _is_simple_value(component.value):
            states.append((string("debug.value"), _format_value("", component.value)))
        for name, value in sorted(
            attributes.items(),
            key=lambda item: item[0].casefold(),
        ):
            formatted = _format_value(name, value)
            target = states if _is_state_name(name) else other_attributes
            target.append((name, formatted))
        self._render_attribute_table(
            "#debug-state-table",
            states,
            string("debug.no_state"),
            component.path,
        )
        self._render_attribute_table(
            "#debug-attributes-table",
            other_attributes,
            string("debug.no_attributes"),
            component.path,
        )

    def _render_attribute_table(
        self,
        selector: str,
        values: list[tuple[str, str]],
        empty_text: str,
        component_path: str,
    ) -> None:
        table = self.query_one(selector, widgets.DataTable)
        rows = tuple(values) if values else ((empty_text, ""),)
        previous_rows = self._rendered_table_values.get(selector, ())
        previous_values = dict(previous_rows)
        current_values = dict(rows)
        if self._rendered_table_components.get(selector) != component_path:
            table.clear(columns=False)
            for name, value in rows:
                table.add_row(
                    Text(name),
                    Text(value, no_wrap=False),
                    key=name,
                )
        else:
            for name in previous_values.keys() - current_values.keys():
                table.remove_row(name)
            for name, value in rows:
                if name not in previous_values:
                    table.add_row(
                        Text(name),
                        Text(value, no_wrap=False),
                        key=name,
                    )
                elif previous_values[name] != value:
                    table.update_cell(
                        name,
                        "debug-value",
                        Text(value, no_wrap=False),
                    )
        self._rendered_table_values[selector] = rows
        self._rendered_table_components[selector] = component_path
        if selector == "#debug-state-table":
            table_height = min(max(len(rows), 1), 8)
            if table_height != self._rendered_state_table_height:
                table.styles.height = table_height
                self._rendered_state_table_height = table_height

    def _format_status(self) -> str:
        engine = getattr(self.ui, "celune", None)
        if engine is None:
            operation = string("debug.not_attached")
            activity = string("debug.not_attached")
            backend = string("debug.unknown")
        else:
            operation = _short_value(getattr(engine, "operation_state", None)).upper()
            activity = _short_value(getattr(engine, "cur_state", None))
            backend_value = getattr(engine, "backend", None)
            if backend_value is None:
                backend_value = getattr(engine, "tts_backend", None)
            backend = _short_value(backend_value)
        trace_state = (
            string("debug.trace_unavailable")
            if not self.trace.available
            else string("debug.trace_paused")
            if self.trace.paused
            else string("debug.trace_active")
        )
        return string(
            "debug.status_format",
            operation=operation,
            activity=activity,
            backend=backend,
            trace=trace_state,
        )

    def _update_pause_button(self) -> None:
        button = self.query_one("#debug-pause", widgets.Button)
        if not self.trace.available:
            button.disabled = True
            button.label = string("debug.trace_unavailable")
            return
        button.disabled = False
        button.label = string("debug.resume" if self.trace.paused else "debug.pause")


def _codes_from_value(
    value: object,
    seen: set[CodeType],
) -> Iterator[CodeType]:
    if isinstance(value, types.FunctionType):
        yield from _walk_code(value.__code__, seen)
        wrapped = getattr(value, "__wrapped__", None)
        if isinstance(wrapped, types.FunctionType) and wrapped.__code__ not in seen:
            yield from _codes_from_value(wrapped, seen)
    elif isinstance(value, (staticmethod, classmethod)):
        yield from _codes_from_value(value.__func__, seen)
    elif isinstance(value, property):
        for function in (value.fget, value.fset, value.fdel):
            if function is not None:
                yield from _codes_from_value(function, seen)
    elif isinstance(value, type):
        for member in vars(value).values():
            yield from _codes_from_value(member, seen)


def _is_celune_code(code: CodeType) -> bool:
    """Return whether a code object was compiled from the Celune package."""
    try:
        return Path(code.co_filename).resolve().is_relative_to(_CELUNE_SOURCE_ROOT)
    except (OSError, RuntimeError, ValueError):
        return False


def _walk_code(code: CodeType, seen: set[CodeType]) -> Iterator[CodeType]:
    if code in seen:
        return
    seen.add(code)
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _walk_code(constant, seen)


def _object_attributes(value: object) -> dict[str, object]:
    try:
        attributes = vars(value)
    except (TypeError, AttributeError):
        return {}
    return {
        name: attribute
        for name, attribute in attributes.items()
        if not name.startswith("__")
    }


def _is_component_object(value: object) -> bool:
    if isinstance(value, (str, bytes, int, float, bool, Enum)):
        return False
    if isinstance(value, (ModuleType, type)) or callable(value):
        return False
    return bool(_object_attributes(value))


def _component_label(name: str) -> str:
    """Format an internal component identifier for the selector tree."""
    words = name.lstrip("_").split("_")
    return " ".join(
        word.upper() if word.casefold() in _COMPONENT_ACRONYMS else word.capitalize()
        for word in words
        if word
    )


def _is_simple_value(value: object) -> bool:
    return isinstance(
        value,
        (str, bytes, int, float, bool, Enum, type(None)),
    )


def _is_sensitive_name(name: str) -> bool:
    normalized = name.casefold()
    return any(
        part in normalized
        for part in (
            "password",
            "secret",
            "token",
            "credential",
            "api_key",
            "authorization",
        )
    )


def _format_value(name: str, value: object) -> str:
    if _is_sensitive_name(name):
        return string("debug.redacted")
    if isinstance(value, Enum):
        return value.name
    if value is None:
        return string("debug.none")
    if isinstance(value, bool):
        return string("debug.true") if value else string("debug.false")
    if isinstance(value, str):
        safe_value = value.replace("\r", "\\r").replace("\n", "\\n")
        return safe_value[:160] + ("…" if len(safe_value) > 160 else "")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return string("debug.byte_count", count=len(value))
    if isinstance(value, (list, tuple, set, frozenset, dict, deque)):
        return string(
            "debug.collection_count",
            kind=type(value).__name__,
            count=len(value),
        )
    return f"<{type(value).__name__}>"


def _is_state_name(name: str) -> bool:
    normalized = name.casefold().lstrip("_")
    return "state" in normalized or normalized in {
        "status",
        "active",
        "ready",
        "loaded",
        "connected",
        "enabled",
        "available",
    }


def _short_value(value: object) -> str:
    if value is None:
        return string("debug.unknown")
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)):
        return _format_value("", value)
    return type(value).__name__


def _component_signature(
    components: list[_InspectorComponent],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (component.path, id(component.value))
        for component in _iter_components(components)
    )


def _iter_components(
    components: list[_InspectorComponent],
) -> Iterator[_InspectorComponent]:
    for component in components:
        yield component
        yield from _iter_components(component.children)


def _format_duration(entry: DebugCallEntry) -> str:
    if entry.duration_ms is not None:
        return string("debug.duration_ms", duration=f"{entry.duration_ms:.1f}")
    if entry.result == "running":
        elapsed_ms = (time.perf_counter_ns() - entry.started_ns) / 1_000_000
        return string("debug.running_duration", duration=f"{elapsed_ms:.1f}")
    return string("debug.unknown")


def _is_debug_enabled(ui: object) -> bool:
    engine = getattr(ui, "celune", None)
    log_level = getattr(engine, "log_level", None)
    if log_level is None:
        log_level = getattr(ui, "_startup_log_level", "info")
    return log_level == "debug"
