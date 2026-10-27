# SPDX-License-Identifier: Apache-2.0
"""Tests for debug inspector call tracing and no-core startup behavior."""

import time
import asyncio
from pathlib import Path
from unittest import mock

from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, DataTable, Input, Static

from celune.ui.debug import (
    DebugCallTrace,
    DebugCallEntry,
    DebugInspectorScreen,
    _has_python_call_method,
    _is_python_callable,
    _classify_source,
    _module_matches_filters,
)


class _InspectorHost(App[None]):
    """Mount the inspector with a minimal UI object for lifecycle tests."""

    def __init__(self, ui: object, trace: DebugCallTrace) -> None:
        super().__init__()
        self.inspector = DebugInspectorScreen(ui, trace)

    def compose(self) -> ComposeResult:
        yield Static()

    def on_mount(self) -> None:
        """Open the inspector as soon as the host app is ready."""
        self.push_screen(self.inspector)


def test_source_classification_covers_known_and_unknown_sources(
    tmp_path: Path,
) -> None:
    """Classify Celune, standard, third-party, built-in, and unknown sources."""
    package_root = tmp_path / "site-packages"
    package_file = package_root / "sample_package" / "module.py"
    with mock.patch("celune.ui.debug._SITE_PACKAGE_ROOTS", (package_root,)):
        assert _classify_source("celune.ui.debug", "celune/ui/debug.py") == "celune"
        assert _classify_source("json.encoder", None) == "standard_library"
        assert _classify_source("builtins", None) == "built_in"
        assert _classify_source("sample_package.module", str(package_file)) == (
            "third_party"
        )
        assert _classify_source("application.module", None) == "unknown"
        assert _classify_source(None, None) == "unknown"


def test_source_filters_append_package_names_and_reset() -> None:
    """Append module filters and return to matching calls from every source."""
    trace = DebugCallTrace(object())

    assert trace.add_source_filter("numpy")
    assert trace.add_source_filter("pathlib")
    assert trace.source_filters == ("numpy", "pathlib")
    assert _module_matches_filters("numpy.linalg", trace.source_filters)
    assert _module_matches_filters("pathlib", trace.source_filters)
    assert not _module_matches_filters("json", trace.source_filters)

    trace.reset_source_filters()

    assert trace.source_filters == ()
    assert _module_matches_filters("json", trace.source_filters)


def test_monitor_scope_skips_its_rendering_framework_by_default() -> None:
    """Avoid tracing the inspector's high-volume Textual and Rich rendering."""
    trace = DebugCallTrace(object())

    assert not trace._should_monitor_module("celune.ui.debug")
    assert not trace._should_monitor_module("textual.widget")
    assert not trace._should_monitor_module("rich.segment")
    assert trace._should_monitor_module("json.encoder")
    assert trace._should_monitor_module("sample_package.module")

    assert not trace.add_source_filter("textual")
    assert not trace.add_source_filter("rich.segment")
    assert trace.source_filters == ()

    assert trace.add_source_filter("celune.ui.debug")
    assert trace._should_monitor_module("celune.ui.debug")
    assert not trace._should_monitor_module("json.encoder")

    trace.reset_source_filters()

    assert not trace._should_monitor_module("textual.widget")
    assert trace._should_monitor_module("json.encoder")


def test_callable_detection_caches_types_and_keeps_python_callables() -> None:
    """Cache repeated C-call type checks and identify Python callable objects."""
    _has_python_call_method.cache_clear()

    assert not _is_python_callable(abs)
    initial_hits = _has_python_call_method.cache_info().hits
    assert not _is_python_callable(len)
    assert _has_python_call_method.cache_info().hits == initial_hits + 1

    class PythonCallable:
        """Provide a Python implementation of ``__call__``."""

        def __call__(self) -> None:
            return None

    assert _is_python_callable(PythonCallable())
    assert _is_python_callable(PythonCallable())


def test_local_monitor_records_python_and_builtin_calls() -> None:
    """Trace loaded function code and the built-in call it makes."""
    trace = DebugCallTrace(object())

    def monitored_sample() -> int:
        return len((1, 2, 3))

    class MonitoredObject:
        """Provide a special method for the inspector toggle test."""

        def __init__(self) -> None:
            self.ready = True

    assert trace.start()
    tool_id = trace._tool_id
    monitored_code = monitored_sample.__code__
    try:
        trace.clear()
        assert monitored_sample() == 3
        entries = trace.entries()
        assert any(entry.name.endswith("monitored_sample") for entry in entries)
        assert any(
            entry.name == "builtins.len"
            and entry.source_category == "built_in"
            and entry.duration_ms is not None
            for entry in entries
        )

        trace.clear()
        MonitoredObject()
        assert not any(entry.name.endswith(".__init__") for entry in trace.entries())

        trace.set_show_special_methods(True)
        trace.clear()
        MonitoredObject()
        assert any(entry.name.endswith(".__init__") for entry in trace.entries())
    finally:
        trace.stop()
    assert tool_id is not None
    assert trace._monitoring is not None
    assert trace._monitoring.get_tool(tool_id) is None
    assert monitored_code not in trace._active_codes


def test_inspector_remains_usable_without_an_attached_core() -> None:
    """Mount the debug inspector while startup has not attached the engine."""
    ui = type("DebugUI", (), {"celune": None, "_startup_log_level": "debug"})()
    trace = DebugCallTrace(ui)
    trace.set_enabled(True)
    host = _InspectorHost(ui, trace)

    async def inspect_screen() -> None:
        async with host.run_test(size=(110, 32)) as pilot:
            await pilot.pause()
            assert trace.available
            status = host.screen.query_one("#debug-status", Static)
            sources = host.screen.query_one("#debug-source-status", Static)
            title = host.screen.query_one("#debug-live-title", Static)
            live_pane = host.screen.query_one("#debug-live-pane")
            calls = host.screen.query_one("#debug-call-table", DataTable)
            source_input = host.screen.query_one("#debug-source-filter", Input)
            special_methods = host.screen.query_one(
                "#debug-show-special-methods", Checkbox
            )
            host.screen.query_one("#debug-select-source", Button)
            host.screen.query_one("#debug-reset-sources", Button)
            assert "not attached" in str(status.content)
            assert str(sources.content) == "Tracing all function calls"
            assert sources.region.y == title.region.y + 1
            assert calls.region.y >= (
                special_methods.region.y + special_methods.region.height + 1
            )
            assert live_pane.region.height <= 10
            assert len(calls.ordered_columns) == 4
            assert calls.ordered_columns[3].width == 18
            assert calls.max_scroll_x == 0
            assert trace.source_filters == ()
            assert 16 <= source_input.region.width <= 34
            inspector = host.screen
            assert isinstance(inspector, DebugInspectorScreen)
            trace.set_paused(True)
            with trace._lock:
                trace._entries.clear()
                trace._entries.extend(
                    DebugCallEntry(
                        entry_id=entry_id,
                        timestamp_ns=entry_id,
                        name=f"sample_package.module.call_{entry_id}",
                        display_name=(
                            "sample_package.module." + "very_long_function_name_" * 8
                        ),
                        source_category="third_party",
                        module_name="sample_package.module",
                        special_method=False,
                        call_path=("sample_package.module.call",),
                        arguments=(),
                        thread_id=1,
                        source_path="sample_package/module.py",
                        source_line=1,
                        started_ns=time.perf_counter_ns(),
                    )
                    for entry_id in range(1, DebugInspectorScreen._CALL_TABLE_LIMIT + 1)
                )
                trace._next_entry_id = DebugInspectorScreen._CALL_TABLE_LIMIT
            inspector._refresh_live_calls(force=True)
            await pilot.pause()
            inspector._refresh_live_calls()
            assert calls.row_count <= DebugInspectorScreen._CALL_TABLE_LIMIT
            assert calls.cursor_row == calls.row_count - 1
            assert len(trace.entries()) <= DebugCallTrace._TRACE_LIMIT
            assert calls.max_scroll_y > 0
            assert calls.max_scroll_x == 0
            calls.scroll_to(y=0, animate=False, force=True)
            assert calls.scroll_y == 0, (
                f"scroll did not move: scroll_y={calls.scroll_y}, "
                f"max_scroll_y={calls.max_scroll_y}"
            )
            await pilot.pause()
            assert not calls.is_vertical_scroll_end, (
                f"scroll_y={calls.scroll_y}, max_scroll_y={calls.max_scroll_y}"
            )
            assert " ms running" in calls.render_line(1).text
            rows_before_scroll_refresh = tuple(calls.rows)
            with trace._lock:
                trace._next_entry_id += 1
                trace._entries.append(
                    DebugCallEntry(
                        entry_id=trace._next_entry_id,
                        timestamp_ns=trace._next_entry_id,
                        name="sample_package.module.new_call",
                        display_name="sample_package.module.new_call",
                        source_category="third_party",
                        module_name="sample_package.module",
                        special_method=False,
                        call_path=("sample_package.module.new_call",),
                        arguments=(),
                        thread_id=1,
                        source_path="sample_package/module.py",
                        source_line=1,
                        started_ns=trace._next_entry_id,
                    )
                )
            inspector._refresh_live_calls()
            assert tuple(calls.rows) == rows_before_scroll_refresh
            assert calls.scroll_y == 0
            calls.scroll_to(y=calls.max_scroll_y, animate=False, force=True)
            await pilot.pause()
            inspector._refresh_live_calls()
            assert inspector._shown_trace_ids[-1] == trace._next_entry_id
            assert calls.cursor_row == calls.row_count - 1
            await pilot.pause()
            assert calls.is_vertical_scroll_end

            await pilot.click("#debug-show-special-methods")
            assert trace._show_special_methods

            trace.set_paused(True)
            entries_before_close = {entry.entry_id for entry in trace.entries()}
            await pilot.press("escape")
            await pilot.pause()
            assert not trace.available
            retained_entry_ids = {entry.entry_id for entry in trace.entries()}
            assert entries_before_close <= retained_entry_ids

            host.push_screen(DebugInspectorScreen(ui, trace))
            await pilot.pause()
            assert trace.available
            assert trace.paused
            trace.set_paused(False)

    asyncio.run(inspect_screen())
    assert not trace.available
