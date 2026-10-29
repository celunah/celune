# SPDX-License-Identifier: Apache-2.0
"""Read-only runtime inspector for the Textual debug view."""

from __future__ import annotations

import sys
import site
import time
import types
import inspect
import functools
import sysconfig
import threading
import contextlib
from enum import Enum
from types import CodeType, FrameType, ModuleType
from typing import Literal, ClassVar, Optional, cast
from pathlib import Path
from datetime import datetime
from collections import abc, deque
from dataclasses import field, dataclass

from textual import app as textual_app
from textual import events, screen, binding, widgets, containers
from rich.text import Text

from ..i18n import string
from ..paths import running_compiled
from ..constants import APP_NAME

ComposeResult = textual_app.ComposeResult


def _site_package_roots() -> tuple[Path, ...]:
    """Return installed third-party package roots for this interpreter."""
    roots: set[Path] = set()
    for path in sysconfig.get_paths().values():
        if path and ("site-packages" in path or "dist-packages" in path):
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                roots.add(Path(path).resolve())
    for path in site.getsitepackages():
        if "site-packages" in path or "dist-packages" in path:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                roots.add(Path(path).resolve())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str) and (
        "site-packages" in user_site or "dist-packages" in user_site
    ):
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            roots.add(Path(user_site).resolve())
    return tuple(sorted(roots, key=str))


_SITE_PACKAGE_ROOTS = _site_package_roots()
_STANDARD_LIBRARY_MODULES = getattr(sys, "stdlib_module_names", frozenset())


def _inspect_code_flag(name: str) -> int:
    """Return an inspect code flag by its standard-library name."""
    return cast(int, getattr(inspect, name))


_CODE_FLAG_VARARGS = _inspect_code_flag("CO_VARARGS")
_CODE_FLAG_VARKEYWORDS = _inspect_code_flag("CO_VARKEYWORDS")


@dataclass(frozen=True)
class _MonitorEvents:
    """Event constants used by the optional execution monitor."""

    PY_START: int
    PY_RESUME: int
    PY_RETURN: int
    PY_YIELD: int
    PY_UNWIND: int
    CALL: int
    C_RETURN: int
    C_RAISE: int
    NO_EVENTS: int


@dataclass(frozen=True)
class _MonitoringAPI:
    """Small typed surface of Python's execution monitoring API."""

    events: _MonitorEvents
    get_tool: abc.Callable[[int], Optional[str]]
    use_tool_id: abc.Callable[[int, str], None]
    free_tool_id: abc.Callable[[int], None]
    register_callback: abc.Callable[
        [int, int, Optional[abc.Callable[..., object]]],
        Optional[abc.Callable[..., object]],
    ]
    set_events: abc.Callable[[int, int], None]
    set_local_events: abc.Callable[[int, CodeType, int], None]


def _monitoring_attribute(api: object, name: str) -> object:
    """Return an optional monitoring API attribute by name."""
    return getattr(api, name)


def _load_monitoring_api() -> Optional[_MonitoringAPI]:
    """Return a typed adapter for Python's optional execution monitor."""
    api = getattr(sys, "monitoring", None)
    if api is None:
        return None
    try:
        event_api = _monitoring_attribute(api, "events")
        monitor_events = _MonitorEvents(
            PY_START=cast(int, _monitoring_attribute(event_api, "PY_START")),
            PY_RESUME=cast(int, _monitoring_attribute(event_api, "PY_RESUME")),
            PY_RETURN=cast(int, _monitoring_attribute(event_api, "PY_RETURN")),
            PY_YIELD=cast(int, _monitoring_attribute(event_api, "PY_YIELD")),
            PY_UNWIND=cast(int, _monitoring_attribute(event_api, "PY_UNWIND")),
            CALL=cast(int, _monitoring_attribute(event_api, "CALL")),
            C_RETURN=cast(int, _monitoring_attribute(event_api, "C_RETURN")),
            C_RAISE=cast(int, _monitoring_attribute(event_api, "C_RAISE")),
            NO_EVENTS=cast(int, _monitoring_attribute(event_api, "NO_EVENTS")),
        )
        return _MonitoringAPI(
            events=monitor_events,
            get_tool=cast(
                abc.Callable[[int], Optional[str]],
                _monitoring_attribute(api, "get_tool"),
            ),
            use_tool_id=cast(
                abc.Callable[[int, str], None],
                _monitoring_attribute(api, "use_tool_id"),
            ),
            free_tool_id=cast(
                abc.Callable[[int], None],
                _monitoring_attribute(api, "free_tool_id"),
            ),
            register_callback=cast(
                abc.Callable[
                    [int, int, Optional[abc.Callable[..., object]]],
                    Optional[abc.Callable[..., object]],
                ],
                _monitoring_attribute(api, "register_callback"),
            ),
            set_events=cast(
                abc.Callable[[int, int], None],
                _monitoring_attribute(api, "set_events"),
            ),
            set_local_events=cast(
                abc.Callable[[int, CodeType, int], None],
                _monitoring_attribute(api, "set_local_events"),
            ),
        )
    except AttributeError:
        return None


@dataclass
class DebugCallEntry:
    """One observed Python or native callable activation."""

    entry_id: int
    timestamp_ns: int
    name: str
    display_name: str
    source_category: str
    module_name: Optional[str]
    special_method: bool
    call_path: tuple[str, ...]
    arguments: tuple[tuple[str, str], ...]
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
    kind: str = "python"
    instruction_offset: Optional[int] = None
    callable_id: Optional[int] = None


class DebugCallTrace:
    """Capture bounded Python and native calls visible to Python monitoring."""

    _MAX_CALL_DEPTH = 24
    _TRACE_LIMIT = 100
    _EVENTS: ClassVar[tuple[str, ...]] = (
        "PY_START",
        "PY_RESUME",
        "PY_RETURN",
        "PY_YIELD",
        "PY_UNWIND",
        "CALL",
        "C_RETURN",
        "C_RAISE",
    )
    _LOCAL_EVENTS: ClassVar[tuple[str, ...]] = (
        "PY_START",
        "PY_RESUME",
        "PY_RETURN",
        "PY_YIELD",
        "CALL",
    )
    _INSPECTOR_MODULES: ClassVar[tuple[str, ...]] = ("textual", "rich")

    def __init__(self, ui: object) -> None:
        self._ui = ui
        self._monitoring: Optional[_MonitoringAPI] = _load_monitoring_api()
        self._tool_id: Optional[int] = None
        self._entries: deque[DebugCallEntry] = deque(maxlen=self._TRACE_LIMIT)
        self._stacks: dict[int, list[_ActiveCall]] = {}
        self._lock = threading.RLock()
        self._source_cache: dict[tuple[Optional[str], Optional[str]], str] = {}
        self._source_filters: tuple[str, ...] = ()
        self._module_codes: dict[str, set[CodeType]] = {}
        self._module_identities: dict[str, int] = {}
        self._code_modules: dict[CodeType, str] = {}
        self._active_codes: set[CodeType] = set()
        self._local_event_mask = 0
        self._next_entry_id = 0
        self._recording = True
        self._suspended = 0
        self._show_special_methods = False
        self._enabled = False
        self._inspector_open = False
        self._active_event_mask = 0
        self._events_active = False
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
            was_enabled = self._enabled
            self._enabled = False
            if was_enabled or self.available:
                self.stop()
            if was_enabled:
                self.clear()
                self._recording = True
            return

        if not self._enabled:
            self._enabled = True
            self._failed = False
            self._recording = True
            self.clear()
        if self._inspector_open:
            self._ensure_started()

    def set_inspector_open(self, opened: bool) -> None:
        """Start tracing while the debug inspector is mounted."""
        if self._inspector_open == opened:
            return
        self._inspector_open = opened
        if not opened:
            self.stop()
            return
        if self._enabled:
            self._ensure_started()

    def _ensure_started(self) -> None:
        if self.available:
            return
        if self._tool_id is not None:
            self.stop()
        self._failed = False
        self.start()

    def start(self) -> bool:
        """Install local Python and exposed C-call monitoring."""
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
                "CALL": self._on_c_call,
                "C_RETURN": self._on_c_return,
                "C_RAISE": self._on_c_raise,
            }
            for event_name, callback in callbacks.items():
                event = getattr(self._monitoring.events, event_name)
                self._monitoring.register_callback(tool_id, event, callback)
            self._active_event_mask = self._monitoring.events.PY_UNWIND
            self._local_event_mask = 0
            for event_name in self._LOCAL_EVENTS:
                self._local_event_mask |= getattr(self._monitoring.events, event_name)
            self._monitoring.set_events(tool_id, self._active_event_mask)
            self._events_active = True
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
        """Remove this monitor's callbacks and global events."""
        tool_id = self._tool_id
        monitoring = self._monitoring
        self._tool_id = None
        self._events_active = False
        self._active_event_mask = 0
        if tool_id is None or monitoring is None:
            return

        self._finalize_active_calls()
        for code in tuple(self._active_codes):
            with contextlib.suppress(Exception):
                monitoring.set_local_events(
                    tool_id,
                    code,
                    monitoring.events.NO_EVENTS,
                )
        self._active_codes.clear()
        self._code_modules.clear()
        self._module_codes.clear()
        self._module_identities.clear()
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

    @contextlib.contextmanager
    def suspend_events(self) -> abc.Generator[None, None, None]:
        """Suspend monitoring around inspector work to avoid tracing its redraws."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    def set_paused(self, paused: bool) -> None:
        """Pause or resume trace collection and its local monitor events."""
        recording = not paused
        if self._recording == recording:
            return
        self._recording = recording
        self._finalize_active_calls()
        if self.available:
            monitoring = self._monitoring
            if monitoring is not None and self._tool_id is not None:
                event_mask = (
                    self._active_event_mask
                    if recording
                    else monitoring.events.NO_EVENTS
                )
                with contextlib.suppress(Exception):
                    monitoring.set_events(self._tool_id, event_mask)
            self._apply_local_event_scope()

    @property
    def source_filters(self) -> tuple[str, ...]:
        """Return the active package and module filters."""
        with self._lock:
            return self._source_filters

    def add_source_filter(self, module_name: str) -> bool:
        """Append a package or module name to the active source filters."""
        normalized = module_name.strip()
        if not normalized or _is_blacklisted_module(normalized):
            return False
        with self._lock:
            if normalized in self._source_filters:
                return True
            self._source_filters += (normalized,)
        self._refresh_module_scope()
        return True

    def reset_source_filters(self) -> None:
        """Clear source filters so calls from all modules are recorded."""
        with self._lock:
            self._source_filters = ()
        self._refresh_module_scope()

    def refresh_code_objects(self, *, apply_scope: bool = False) -> None:
        """Monitor newly loaded Python modules without tracing the inspector UI."""
        tool_id = self._tool_id
        monitoring = self._monitoring
        if tool_id is None or monitoring is None or self._failed:
            return
        for module_name, module in tuple(sys.modules.items()):
            if not isinstance(module, ModuleType):
                continue
            module_identity = id(module)
            if self._module_identities.get(module_name) == module_identity:
                continue
            old_codes = self._module_codes.get(module_name, set())
            for code in old_codes:
                self._set_code_monitoring(code, False)
                self._code_modules.pop(code, None)
            codes: set[CodeType] = set()
            visited: set[int] = set()
            for value in vars(module).values():
                codes.update(_codes_from_value(value, visited, module_name))
            self._module_codes[module_name] = codes
            self._module_identities[module_name] = module_identity
            self._code_modules.update((code, module_name) for code in codes)
            for code in codes:
                self._set_code_monitoring(
                    code,
                    self._recording and self._should_monitor_module(module_name),
                )
        if apply_scope:
            self._apply_local_event_scope()

    def set_show_special_methods(self, show: bool) -> None:
        """Choose whether Python special methods are included in the trace."""
        self._show_special_methods = show

    def clear(self) -> None:
        """Remove all visible entries from the rolling trace."""
        with self._lock:
            self._entries.clear()

    def entries(
        self,
        limit: Optional[int] = None,
    ) -> tuple[DebugCallEntry, ...]:
        """Return a stable snapshot of the rolling call buffer."""
        if limit is not None and limit <= 0:
            return ()
        with self._lock:
            snapshot = tuple(self._entries)
            source_filters = self._source_filters
            show_special_methods = self._show_special_methods
        selected: list[DebugCallEntry] = []
        for entry in reversed(snapshot):
            if not _module_matches_filters(entry.module_name, source_filters):
                continue
            if not show_special_methods and entry.special_method:
                continue
            selected.append(entry)
            if limit is not None and len(selected) >= limit:
                break
        return tuple(reversed(selected))

    def get_entry(self, entry_id: int) -> Optional[DebugCallEntry]:
        """Return a buffered call by its entry identifier."""
        with self._lock:
            snapshot = tuple(self._entries)
        return next(
            (entry for entry in reversed(snapshot) if entry.entry_id == entry_id),
            None,
        )

    def _on_start(self, code: CodeType, instruction_offset: int) -> None:
        del instruction_offset
        if self._failed or self._suspended or not self._recording:
            return
        try:
            frame = sys._getframe().f_back
            module_name = _frame_module_name(frame, code)
            source_path = code.co_filename
            source_category = self._classify_source(module_name, source_path)
            name = _qualified_python_name(module_name, code)
            special_method = _is_special_method(code.co_name)
            started_ns = time.perf_counter_ns()
            thread_id = threading.get_ident()
            with self._lock:
                stack = self._stacks.setdefault(thread_id, [])
                entry: Optional[DebugCallEntry] = None
                if (
                    self._recording
                    and _module_matches_filters(module_name, self._source_filters)
                    and (self._show_special_methods or not special_method)
                ):
                    path = tuple(call.name for call in stack) + (name,)
                    arguments = _call_arguments(code, frame)
                    self._next_entry_id += 1
                    entry = DebugCallEntry(
                        entry_id=self._next_entry_id,
                        timestamp_ns=time.time_ns(),
                        name=name,
                        display_name=f"{name}()",
                        source_category=source_category,
                        module_name=module_name,
                        special_method=special_method,
                        call_path=path[-self._MAX_CALL_DEPTH :],
                        arguments=arguments,
                        thread_id=thread_id,
                        source_path=code.co_filename,
                        source_line=code.co_firstlineno,
                        started_ns=started_ns,
                    )
                    self._entries.append(entry)
                stack.append(_ActiveCall(code, name, started_ns, entry))
        except Exception:
            self._fail()

    def _on_c_call(
        self,
        code: CodeType,
        instruction_offset: int,
        callable_value: object,
        arg0: object,
    ) -> None:
        del arg0
        if (
            self._failed
            or self._suspended
            or not self._recording
            or _is_python_callable(callable_value)
        ):
            return
        try:
            if isinstance(callable_value, types.BuiltinFunctionType):
                module_name, name, special_method = _builtin_callable_metadata(
                    callable_value
                )
            else:
                module_name = _callable_module_name(callable_value)
                name = _qualified_callable_name(module_name, callable_value)
                special_method = _is_special_method(name.rpartition(".")[2])
            source_category = self._classify_source(module_name, None)
            started_ns = time.perf_counter_ns()
            thread_id = threading.get_ident()
            frame = sys._getframe().f_back
            source_path = code.co_filename
            source_line = frame.f_lineno if frame is not None else code.co_firstlineno
            with self._lock:
                stack = self._stacks.setdefault(thread_id, [])
                entry: Optional[DebugCallEntry] = None
                if (
                    self._recording
                    and _module_matches_filters(module_name, self._source_filters)
                    and (self._show_special_methods or not special_method)
                ):
                    path = tuple(call.name for call in stack) + (name,)
                    self._next_entry_id += 1
                    entry = DebugCallEntry(
                        entry_id=self._next_entry_id,
                        timestamp_ns=time.time_ns(),
                        name=name,
                        display_name=f"{name}()",
                        source_category=source_category,
                        module_name=module_name,
                        special_method=special_method,
                        call_path=path[-self._MAX_CALL_DEPTH :],
                        arguments=(),
                        thread_id=thread_id,
                        source_path=source_path,
                        source_line=source_line,
                        started_ns=started_ns,
                    )
                    self._entries.append(entry)
                stack.append(
                    _ActiveCall(
                        code,
                        name,
                        started_ns,
                        entry,
                        kind="native",
                        instruction_offset=instruction_offset,
                        callable_id=id(callable_value),
                    )
                )
        except Exception:
            self._fail()

    def _on_c_return(
        self,
        code: CodeType,
        instruction_offset: int,
        callable_value: object,
        arg0: object,
    ) -> None:
        del arg0
        self._finish_call(
            code,
            "returned",
            None,
            kind="native",
            instruction_offset=instruction_offset,
            callable_id=id(callable_value),
        )

    def _on_c_raise(
        self,
        code: CodeType,
        instruction_offset: int,
        callable_value: object,
        arg0: object,
    ) -> None:
        del arg0
        self._finish_call(
            code,
            "raised",
            None,
            kind="native",
            instruction_offset=instruction_offset,
            callable_id=id(callable_value),
        )

    def _on_return(
        self,
        code: CodeType,
        instruction_offset: int,
        return_value: object,
    ) -> None:
        del instruction_offset, return_value
        if self._suspended or not self._recording:
            return
        self._finish_call(code, "returned", None)

    def _on_unwind(
        self,
        code: CodeType,
        instruction_offset: int,
        exception: BaseException,
    ) -> None:
        del instruction_offset
        if self._suspended or not self._recording:
            return
        self._finish_call(code, "raised", type(exception).__name__)

    def _finish_call(
        self,
        code: CodeType,
        result: str,
        exception_type: Optional[str],
        *,
        kind: str = "python",
        instruction_offset: Optional[int] = None,
        callable_id: Optional[int] = None,
    ) -> None:
        if self._failed or self._suspended or not self._recording:
            return
        thread_id = threading.get_ident()
        finished_ns = time.perf_counter_ns()
        try:
            with self._lock:
                stack = self._stacks.get(thread_id)
                if not stack:
                    return
                active = stack[-1]
                if (
                    active.code is code
                    and active.kind == kind
                    and (
                        kind == "python"
                        or (
                            active.instruction_offset == instruction_offset
                            and active.callable_id == callable_id
                        )
                    )
                ):
                    stack.pop()
                    if not stack:
                        self._stacks.pop(thread_id, None)
                    entry = active.entry
                    if entry is not None:
                        entry.duration_ms = (
                            finished_ns - active.started_ns
                        ) / 1_000_000
                        entry.result = result
                        entry.exception_type = exception_type
                    return
                matching_index = next(
                    (
                        index
                        for index in range(len(stack) - 2, -1, -1)
                        if stack[index].code is code
                        and stack[index].kind == kind
                        and (
                            kind == "python"
                            or (
                                stack[index].instruction_offset == instruction_offset
                                and stack[index].callable_id == callable_id
                            )
                        )
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
            self._fail()

    def _classify_source(
        self,
        module_name: Optional[str],
        source_path: Optional[str],
    ) -> str:
        key = (module_name, source_path)
        source_category = self._source_cache.get(key)
        if source_category is None:
            source_category = _classify_source(module_name, source_path)
            self._source_cache[key] = source_category
        return source_category

    def _fail(self) -> None:
        self._failed = True
        self.stop()

    def _should_monitor_module(self, module_name: str) -> bool:
        if _is_blacklisted_module(module_name):
            return False
        source_filters = self.source_filters
        if source_filters:
            return _module_matches_filters(module_name, source_filters)
        if module_name == "celune.ui.debug":
            return False
        return not any(
            module_name == package or module_name.startswith(f"{package}.")
            for package in self._INSPECTOR_MODULES
        )

    def _refresh_module_scope(self) -> None:
        if not self.available:
            return
        self._finalize_active_calls()
        self.refresh_code_objects(apply_scope=True)

    def _apply_local_event_scope(self) -> None:
        tool_id = self._tool_id
        monitoring = self._monitoring
        if tool_id is None or monitoring is None:
            return
        for code, module_name in self._code_modules.items():
            self._set_code_monitoring(
                code,
                self._recording and self._should_monitor_module(module_name),
            )

    def _set_code_monitoring(self, code: CodeType, enabled: bool) -> None:
        tool_id = self._tool_id
        monitoring = self._monitoring
        if tool_id is None or monitoring is None:
            return
        if enabled == (code in self._active_codes):
            return
        event_mask = self._local_event_mask if enabled else monitoring.events.NO_EVENTS
        try:
            monitoring.set_local_events(tool_id, code, event_mask)
        except Exception:
            return
        if enabled:
            self._active_codes.add(code)
        else:
            self._active_codes.discard(code)

    def _finalize_active_calls(self) -> None:
        finished_ns = time.perf_counter_ns()
        with self._lock:
            for stack in self._stacks.values():
                for active in stack:
                    if active.entry is not None:
                        active.entry.duration_ms = (
                            finished_ns - active.started_ns
                        ) / 1_000_000
                        active.entry.result = "interrupted"
            self._stacks.clear()


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


class _DebugCallTable(widgets.DataTable):
    """Call table that can reposition its hidden cursor without auto-scrolling."""

    _preserve_cursor_scroll = False

    def move_cursor_preserving_scroll(self, row: int, column: int) -> None:
        """Move the cursor while suppressing Textual's deferred auto-scroll."""
        self._preserve_cursor_scroll = True
        self.move_cursor(row=row, column=column, animate=False, scroll=False)
        self.call_after_refresh(self._restore_cursor_scroll)

    def _restore_cursor_scroll(self) -> None:
        """Restore normal keyboard cursor scrolling after the pending refresh."""
        self._preserve_cursor_scroll = False

    def _scroll_cursor_into_view(self, animate: bool = False) -> None:
        """Keep deferred cursor updates from overriding manual scrollbar use."""
        if not self._preserve_cursor_scroll:
            super()._scroll_cursor_into_view(animate)


class _DebugResizeHandle(widgets.Static):
    """Resize one inspector split when its handle is dragged with the mouse."""

    def __init__(
        self,
        axis: str,
        on_drag: abc.Callable[[int], None],
        *,
        widget_id: str,
    ) -> None:
        super().__init__("", id=widget_id)
        self._axis = axis
        self._on_drag = on_drag
        self._dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Capture the mouse when the primary button grabs this handle."""
        if event.button != 1:
            return
        self._dragging = True
        self.app.capture_mouse(self)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Resize the adjacent pane while the mouse remains captured."""
        if not self._dragging:
            return
        delta = event.delta_x if self._axis == "x" else event.delta_y
        if delta:
            self._on_drag(delta)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Release the mouse after a resize drag ends."""
        if not self._dragging:
            return
        self._dragging = False
        self.app.capture_mouse(None)
        event.stop()

    def on_mouse_release(self, event: events.MouseRelease) -> None:
        """Clear drag state if another widget takes mouse capture."""
        del event
        self._dragging = False


class DebugInspectorScreen(screen.ModalScreen[None]):
    """Show live Celune and UI state in a debug-only modal."""

    BINDINGS: ClassVar[list[binding.BindingType]] = [
        ("escape", "close_inspector", "Close inspector"),
        ("f", "focus_filter", "Filter components"),
    ]
    _CALL_TABLE_LIMIT: ClassVar[int] = 25
    _LIVE_CALL_REFRESH_SECONDS: ClassVar[float] = 1.0
    _SNAPSHOT_REFRESH_SECONDS: ClassVar[float] = 1.0
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
    #debug-main-resizer,
    #debug-call-resizer {
        width: 1;
        min-width: 1;
        height: 1fr;
        color: $background;
        background: $background;
    }
    #debug-live-resizer {
        height: 1;
        min-height: 1;
        width: 1fr;
        color: $background;
        background: $background;
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
    #debug-modal-preview-title,
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
    #debug-modal-preview {
        display: none;
        width: 1fr;
        height: 1fr;
    }
    #debug-modal-preview-title {
        height: 1;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }
    #debug-modal-preview-description,
    #debug-modal-preview-result {
        height: auto;
        color: $secondary;
        margin-bottom: 1;
    }
    #debug-modal-preview-actions {
        height: auto;
        width: 1fr;
    }
    #debug-modal-preview-actions Button {
        width: 1fr;
        min-width: 20;
        height: 3;
        padding: 0 1;
        border: round $primary;
        background: $surface;
        color: $primary;
        margin-bottom: 1;
    }
    #debug-live-pane {
        height: 10;
        min-height: 9;
        border: round $secondary;
        padding: 0 1;
    }
    #debug-live-heading {
        height: 5;
        align: left top;
    }
    #debug-live-labels {
        width: 26;
        min-width: 26;
        height: 2;
    }
    #debug-live-controls {
        width: 1fr;
        height: 5;
    }
    #debug-live-actions {
        height: 3;
        width: 1fr;
        align: right middle;
    }
    #debug-live-title {
        width: auto;
        min-width: 12;
    }
    #debug-pause,
    #debug-clear,
    #debug-select-source,
    #debug-reset-sources {
        min-width: 10;
        height: 3;
        padding: 0 1;
        border: round $primary;
        background: $surface;
        color: $primary;
        margin-left: 1;
    }
    #debug-source-status-row {
        height: 1;
        align: right middle;
        margin-bottom: 1;
    }
    #debug-source-filter {
        width: 1fr;
        min-width: 16;
        max-width: 34;
        height: 3;
        border: round $secondary;
        margin-left: 1;
        margin-right: 1;
    }
    #debug-select-source,
    #debug-reset-sources {
        min-width: 16;
    }
    #debug-source-status {
        width: 1fr;
        height: 1;
        color: $secondary;
    }
    #debug-show-special-methods {
        width: auto;
        height: 1;
        padding: 0 1;
        border: none;
        background: $background;
    }
    #debug-live-content {
        height: 1fr;
        min-height: 3;
    }
    #debug-call-table {
        width: 75%;
        height: 1fr;
        overflow-y: auto;
        scrollbar-size-vertical: 2;
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
        self._selected_call_entry: Optional[DebugCallEntry] = None
        self._displayed_call_entries: dict[int, DebugCallEntry] = {}
        self._component_width_percent = 30.0
        self._live_pane_height = 10
        self._call_table_width_percent = 75.0
        self._shown_trace_signature: tuple[tuple[int, Optional[float], str], ...] = ()
        self._shown_trace_ids: tuple[int, ...] = ()
        self._shown_call_detail_id: Optional[int] = None
        self._shown_call_detail: Optional[str] = None
        self._compiled_notice_filters: tuple[str, ...] = ()
        self._compiled_notice_target: Optional[str] = None
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
                yield _DebugResizeHandle(
                    "x",
                    self._resize_component_pane,
                    widget_id="debug-main-resizer",
                )
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
                    with containers.Vertical(id="debug-modal-preview"):
                        yield widgets.Static(
                            string("debug.component_ui_modals"),
                            id="debug-modal-preview-title",
                        )
                        yield widgets.Static(
                            string("debug.modal_preview_description"),
                            id="debug-modal-preview-description",
                        )
                        yield widgets.Static(
                            string("debug.modal_preview_result_ignored"),
                            id="debug-modal-preview-result",
                        )
                        with containers.Vertical(id="debug-modal-preview-actions"):
                            yield widgets.Button(
                                string("debug.modal_preview_voice"),
                                id="debug-modal-preview-voice",
                            )
                            yield widgets.Button(
                                string("debug.modal_preview_settings"),
                                id="debug-modal-preview-settings",
                            )
                            yield widgets.Button(
                                string("debug.modal_preview_license"),
                                id="debug-modal-preview-license",
                            )
            yield _DebugResizeHandle(
                "y",
                self._resize_live_pane,
                widget_id="debug-live-resizer",
            )
            with containers.Vertical(id="debug-live-pane"):
                with containers.Horizontal(id="debug-live-heading"):
                    with containers.Vertical(id="debug-live-labels"):
                        yield widgets.Static(
                            string("debug.live_calls"), id="debug-live-title"
                        )
                        yield widgets.Static("", id="debug-source-status")
                    with containers.Vertical(id="debug-live-controls"):
                        with containers.Horizontal(id="debug-live-actions"):
                            yield widgets.Input(
                                placeholder=string("debug.package_module_placeholder"),
                                id="debug-source-filter",
                            )
                            yield widgets.Button(
                                string("debug.select_source"),
                                id="debug-select-source",
                                compact=True,
                            )
                            yield widgets.Button(
                                string("debug.reset_sources"),
                                id="debug-reset-sources",
                                compact=True,
                            )
                            yield widgets.Button(
                                string("debug.pause"), id="debug-pause", compact=True
                            )
                            yield widgets.Button(
                                string("debug.clear"), id="debug-clear", compact=True
                            )
                        with containers.Horizontal(id="debug-source-status-row"):
                            yield widgets.Checkbox(
                                string("debug.show_special_methods"),
                                id="debug-show-special-methods",
                            )
                with containers.Horizontal(id="debug-live-content"):
                    yield _DebugCallTable(
                        cursor_type="row",
                        show_header=True,
                        show_row_labels=False,
                        id="debug-call-table",
                    )
                    yield _DebugResizeHandle(
                        "x",
                        self._resize_call_table,
                        widget_id="debug-call-resizer",
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
        table = self.query_one("#debug-call-table", _DebugCallTable)
        table.add_column(string("debug.time"), width=12)
        table.add_column(string("debug.call"), width=14)
        table.add_column(string("debug.source"), key="debug-call-source", width=16)
        table.add_column(string("debug.duration"), key="debug-duration", width=18)
        self._refresh_snapshot()
        self._refresh_source_status()
        self._refresh_live_calls()
        self.set_interval(self._SNAPSHOT_REFRESH_SECONDS, self._refresh_snapshot)
        self.set_interval(self._LIVE_CALL_REFRESH_SECONDS, self._refresh_live_calls)
        self.trace.set_inspector_open(True)

    def on_unmount(self) -> None:
        """Stop tracing when the debug inspector closes."""
        self.trace.set_inspector_open(False)

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

    def on_input_submitted(self, event: widgets.Input.Submitted) -> None:
        """Apply a package or module filter submitted from its input."""
        if event.input.id == "debug-source-filter":
            self._select_source_filter()

    def on_checkbox_changed(self, event: widgets.Checkbox.Changed) -> None:
        """Update whether special methods are shown in the call trace."""
        if event.checkbox.id == "debug-show-special-methods":
            self.trace.set_show_special_methods(event.value)
            self._refresh_live_calls(force=True)

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Apply the live-trace controls."""
        if event.button.id == "debug-pause":
            self.trace.set_paused(not self.trace.paused)
            self._update_pause_button()
        elif event.button.id == "debug-clear":
            self.trace.clear()
            self._selected_call_id = None
            self._selected_call_entry = None
            self._refresh_live_calls(force=True)
        elif event.button.id == "debug-select-source":
            self._select_source_filter()
        elif event.button.id == "debug-reset-sources":
            self.trace.reset_source_filters()
            self._refresh_source_status()
            self._refresh_live_calls(force=True)
        elif event.button.id == "debug-modal-preview-voice":
            self._open_menu_preview("voice")
        elif event.button.id == "debug-modal-preview-settings":
            self._open_menu_preview("settings")
        elif event.button.id == "debug-modal-preview-license":
            self._open_license_preview()

    def _ignore_modal_preview_result(self, result: object) -> None:
        """Consume a preview modal result without changing application state."""
        del result

    def _open_menu_preview(self, menu_kind: Literal["voice", "settings"]) -> None:
        """Open a no-op preview of the active voice or settings menu."""
        from . import app as _app

        ui = cast(_app.CeluneUI, self.ui)
        if ui.celune is not None:
            if menu_kind == "voice":
                ui.open_voice_menu(preview=True)
            else:
                ui.open_settings_menu(preview=True)
            if self.app.screen is not self:
                return

        if menu_kind == "voice":
            options = (
                _app.SelectMenuOption(
                    label=string("debug.modal_preview_voice_pack"),
                    value="example_voice_1",
                    autocomplete_values=("example_voice_1", "example_voice_2"),
                    autocomplete_labels=(
                        string("debug.modal_preview_voice_one"),
                        string("debug.modal_preview_voice_two"),
                    ),
                ),
            )
            title = string("ui.voice_menu_title")
            confirm_hint = string("ui.voice_confirm_hint")
            return_value = True
        else:
            options = (
                _app.SelectMenuOption(
                    label=string("debug.modal_preview_theme_setting"),
                    value="dark",
                    autocomplete_values=("dark", "light"),
                    display_value=string("debug.modal_preview_theme_dark"),
                    explanation=string("ui.settings_explanation.theme"),
                    autocomplete_labels=(
                        string("debug.modal_preview_theme_dark"),
                        string("debug.modal_preview_theme_light"),
                    ),
                ),
                _app.SelectMenuOption(
                    label=string("debug.modal_preview_vram_setting"),
                    value="medium",
                    autocomplete_values=("low", "medium", "high", "xhigh"),
                    display_value=string("debug.modal_preview_vram_medium"),
                    explanation=string("ui.settings_explanation.vram"),
                    autocomplete_labels=(
                        string("debug.modal_preview_vram_low"),
                        string("debug.modal_preview_vram_medium"),
                        string("debug.modal_preview_vram_high"),
                        string("debug.modal_preview_vram_xhigh"),
                    ),
                ),
            )
            title = string("ui.settings_title")
            confirm_hint = string("ui.settings_confirm_hint")
            return_value = False
        footer = string("ui.menu_hint_separator").join(
            (
                string("ui.menu_hint_select"),
                string("ui.menu_hint_choose"),
                confirm_hint,
                string("ui.menu_hint_cancel"),
            )
        )
        menu = _app.SelectMenuWidget(
            title,
            options,
            value_display="all",
            return_value=return_value,
            footer=footer,
        )

        self.app.push_screen(
            _app.SelectMenuOverlay(menu, discard_result=True),
            self._ignore_modal_preview_result,
        )

    def _open_license_preview(self) -> None:
        """Open the license dialog with preview-only text and no model load."""
        from . import app as _app

        self.app.push_screen(
            _app.ModelConsentScreen(
                prompt=string("debug.modal_preview_license_prompt"),
                instruction=string("registry.consent_instruction"),
                details=(string("debug.modal_preview_license_detail"),),
                entry_hint=string("registry.consent_entry_hint"),
                consent_phrase=string("registry.consent_phrase"),
                accept_label=string("registry.consent_accept"),
                decline_label=string("registry.consent_decline"),
            ),
            self._ignore_modal_preview_result,
        )

    def on_data_table_row_highlighted(
        self,
        event: widgets.DataTable.RowHighlighted,
    ) -> None:
        """Show the call path for the highlighted trace entry."""
        if event.data_table.id != "debug-call-table":
            return
        self._select_call(event.row_key.value)

    def on_data_table_row_selected(
        self,
        event: widgets.DataTable.RowSelected,
    ) -> None:
        """Show details when a mouse click selects a trace entry."""
        if event.data_table.id != "debug-call-table":
            return
        self._select_call(event.row_key.value)

    def _select_call(self, row_key: Optional[str]) -> None:
        """Keep a stable detail snapshot for the selected trace entry."""
        if row_key is None:
            return
        try:
            entry_id = int(row_key)
        except (TypeError, ValueError):
            return
        entry = self._displayed_call_entries.get(entry_id)
        if entry is None:
            entry = self.trace.get_entry(entry_id)
        if entry is None:
            return
        self._selected_call_id = entry_id
        self._selected_call_entry = entry
        self._render_call_detail()

    def action_close_inspector(self) -> None:
        """Close the modal and stop its call trace."""
        self.app.pop_screen()

    def action_focus_filter(self) -> None:
        """Focus the component filter input."""
        self.query_one("#debug-filter", widgets.Input).focus()

    def _refresh_snapshot(self) -> None:
        with self.trace.suspend_events():
            self._update_snapshot()

    def _resize_component_pane(self, delta_x: int) -> None:
        """Resize the component selector relative to its available width."""
        main = self.query_one("#debug-main")
        if main.size.width <= 0:
            return
        self._component_width_percent = min(
            70.0,
            max(
                18.0,
                self._component_width_percent + delta_x * 100 / main.size.width,
            ),
        )
        self.query_one(
            "#debug-components-pane"
        ).styles.width = f"{self._component_width_percent:.1f}%"

    def _resize_live_pane(self, delta_y: int) -> None:
        """Resize Live Calls while keeping enough height for the upper panes."""
        panel = self.query_one("#debug-panel")
        max_height = max(9, panel.content_region.height - 12)
        self._live_pane_height = min(
            max_height,
            max(9, self._live_pane_height - delta_y),
        )
        self.query_one("#debug-live-pane").styles.height = self._live_pane_height

    def _resize_call_table(self, delta_x: int) -> None:
        """Resize the call table relative to its available width."""
        content = self.query_one("#debug-live-content")
        if content.size.width <= 0:
            return
        self._call_table_width_percent = min(
            90.0,
            max(
                25.0,
                self._call_table_width_percent + delta_x * 100 / content.size.width,
            ),
        )
        self.query_one(
            "#debug-call-table"
        ).styles.width = f"{self._call_table_width_percent:.1f}%"

    def _update_snapshot(self) -> None:
        if not _is_debug_enabled(self.ui):
            self.app.pop_screen()
            return
        self.trace.refresh_code_objects()
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

    def _refresh_live_calls(self, *, force: bool = False) -> None:
        with self.trace.suspend_events():
            self._update_live_calls(force=force)

    def _update_live_calls(self, *, force: bool = False) -> None:
        self._refresh_source_status()
        table = self.query_one("#debug-call-table", _DebugCallTable)
        if not force and table.scroll_x > 0:
            return
        notice_target = self._compiled_notice_target_for_filters()
        notice: Optional[DebugCallEntry] = None
        if notice_target is not None:
            call_label = string("debug.compiled_code")
            notice = DebugCallEntry(
                entry_id=-1,
                timestamp_ns=0,
                name=call_label,
                display_name=call_label,
                source_category=_classify_source(notice_target, None),
                module_name=notice_target,
                special_method=False,
                call_path=(),
                arguments=(),
                thread_id=0,
                source_path="",
                source_line=0,
                started_ns=0,
                result="notice",
            )
        entries = self.trace.entries(
            self._CALL_TABLE_LIMIT - (1 if notice is not None else 0)
        )
        if notice is not None:
            entries = (notice,) + entries
        ids = tuple(entry.entry_id for entry in entries)
        self._displayed_call_entries = {entry.entry_id: entry for entry in entries}
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
        previous_scroll_y = table.scroll_y
        scroll_anchor_id, scroll_anchor_offset = self._call_table_scroll_anchor(table)
        scroll_anchor_index = (
            previous_ids.index(scroll_anchor_id)
            if scroll_anchor_id in previous_ids
            else None
        )
        rebuild_for_notice = (
            notice is not None and notice.entry_id not in previous_id_set
        )
        if rebuild_for_notice:
            removed_ids = previous_ids
            added_ids = ids
            previous_id_set.clear()
        else:
            removed_ids = tuple(
                entry_id for entry_id in previous_ids if entry_id not in ids
            )
            added_ids = tuple(
                entry_id for entry_id in ids if entry_id not in previous_id_set
            )

        if previous_selection == -1 and -1 not in ids:
            self._selected_call_id = None
            self._selected_call_entry = None

        for entry_id in removed_ids:
            table.remove_row(str(entry_id))

        for entry in entries:
            row_key = str(entry.entry_id)
            if entry.entry_id not in previous_id_set:
                table.add_row(
                    (
                        _format_timestamp(time.time_ns())
                        if entry.result == "notice"
                        else _format_timestamp(entry.timestamp_ns)
                    ),
                    Text(
                        entry.display_name,
                        no_wrap=entry.result != "notice",
                        overflow=("fold" if entry.result == "notice" else "ellipsis"),
                    ),
                    (
                        string("debug.source_unreachable")
                        if entry.result == "notice"
                        else _source_category_label(entry.source_category)
                    ),
                    (
                        string("debug.not_applicable")
                        if entry.result == "notice"
                        else _format_duration(entry)
                    ),
                    key=row_key,
                )
            elif previous_signature.get(entry.entry_id) != (
                entry.duration_ms,
                entry.result,
            ):
                table.update_cell(
                    row_key,
                    "debug-duration",
                    "" if entry.result == "notice" else _format_duration(entry),
                )

        self._shown_trace_signature = signature
        self._shown_trace_ids = ids
        if not ids:
            if self._selected_call_entry is None:
                self._selected_call_id = None
        elif previous_selection not in ids:
            if previous_selection is None or not previous_ids:
                self._selected_call_id = ids[-1]
            else:
                previous_row = (
                    previous_ids.index(previous_selection)
                    if previous_selection in previous_ids
                    else 0
                )
                self._selected_call_id = ids[min(previous_row, len(ids) - 1)]
        selected_call_id = self._selected_call_id
        if (
            selected_call_id is not None
            and selected_call_id in self._displayed_call_entries
        ):
            self._selected_call_entry = self._displayed_call_entries[selected_call_id]

        if (
            self._selected_call_id is not None
            and self._selected_call_id in ids
            and (self._selected_call_id != previous_selection or removed_ids)
        ):
            table.move_cursor_preserving_scroll(
                ids.index(self._selected_call_id),
                0,
            )
        if removed_ids or added_ids:
            if scroll_anchor_id not in ids:
                scroll_anchor_id = self._next_scroll_anchor(
                    previous_ids,
                    scroll_anchor_index,
                    ids,
                )
            table.call_after_refresh(
                self._restore_call_table_scroll,
                scroll_anchor_id,
                scroll_anchor_offset,
                previous_scroll_y,
            )
        self._render_call_detail()

    def _compiled_notice_target_for_filters(self) -> Optional[str]:
        """Return the selected source when Nuitka makes it untraceable."""
        source_filters = self.trace.source_filters
        if source_filters != self._compiled_notice_filters:
            self._compiled_notice_filters = source_filters
            self._compiled_notice_target = _compiled_source_filter(source_filters)
        return self._compiled_notice_target

    def _call_table_scroll_anchor(
        self,
        table: widgets.DataTable,
    ) -> tuple[Optional[int], float]:
        """Return the call at the top of the viewport and its visible offset."""
        scroll_y = table.scroll_y
        row_top = 0.0
        for row in table.ordered_rows:
            if row.height > 0 and row_top + row.height > scroll_y:
                row_id = row.key.value
                if row_id is None:
                    return None, scroll_y
                try:
                    return int(row_id), scroll_y - row_top
                except (TypeError, ValueError):
                    return None, scroll_y
            row_top += row.height
        return None, scroll_y

    @staticmethod
    def _next_scroll_anchor(
        previous_ids: tuple[int, ...],
        anchor_index: Optional[int],
        current_ids: tuple[int, ...],
    ) -> Optional[int]:
        """Choose the next retained call when the visible call was evicted."""
        if not previous_ids or not current_ids:
            return None
        start = anchor_index if anchor_index is not None else 0
        retained_ids = set(current_ids)
        for entry_id in previous_ids[start:]:
            if entry_id in retained_ids:
                return entry_id
        for entry_id in reversed(previous_ids[:start]):
            if entry_id in retained_ids:
                return entry_id
        return None

    def _restore_call_table_scroll(
        self,
        anchor_id: Optional[int],
        anchor_offset: float,
        fallback_scroll_y: float,
    ) -> None:
        """Restore the viewport to the same call after row updates settle."""
        table = self.query_one("#debug-call-table", _DebugCallTable)
        if anchor_id is None or str(anchor_id) not in table.rows:
            target_y = fallback_scroll_y
        else:
            row_index = table.get_row_index(str(anchor_id))
            target_y = (
                sum(row.height for row in table.ordered_rows[:row_index])
                + anchor_offset
            )
        table.scroll_to(
            y=min(max(0.0, target_y), table.max_scroll_y),
            animate=False,
            force=True,
        )

    def _select_source_filter(self) -> None:
        source_input = self.query_one("#debug-source-filter", widgets.Input)
        if not self.trace.add_source_filter(source_input.value):
            return
        self._refresh_source_status()
        self._refresh_live_calls(force=True)

    def _refresh_source_status(self) -> None:
        filters = self.trace.source_filters
        if filters:
            status = string("debug.trace_sources", sources=", ".join(filters))
        else:
            status = string("debug.trace_all_calls")
        self.query_one("#debug-source-status", widgets.Static).update(status)

    def _render_call_detail(self) -> None:
        body = self.query_one("#debug-call-detail-body", widgets.Static)
        entry = self._selected_call_entry
        if entry is None and self._selected_call_id is not None:
            entry = self._displayed_call_entries.get(self._selected_call_id)
            if entry is None:
                entry = self.trace.get_entry(self._selected_call_id)
            self._selected_call_entry = entry
        if (
            entry is not None
            and entry.duration_ms is None
            and self._shown_call_detail_id == entry.entry_id
        ):
            return
        if entry is None:
            content = string("debug.select_call")
        elif entry.result == "notice":
            content = string("debug.compiled_code_cannot_be_traced")
        else:
            lines = [
                f"{string('debug.function')}: {entry.name}()",
                (
                    f"{string('debug.source_category')}: "
                    f"{_source_category_label(entry.source_category)}"
                ),
                (
                    f"{string('debug.package_module')}: "
                    f"{entry.module_name or string('debug.unknown')}"
                ),
                f"{string('debug.call_path')}: {' → '.join(entry.call_path)}",
                f"{string('debug.arguments')}:",
            ]
            if entry.arguments:
                lines.extend(f"  {name}={value}" for name, value in entry.arguments)
            else:
                lines.append(f"  {string('debug.none')}")
            lines.extend(
                [
                    f"{string('debug.duration')}: {_format_duration(entry)}",
                    f"{string('debug.thread')}: {entry.thread_id}",
                    f"{string('debug.source')}: {entry.source_path}:{entry.source_line}",
                ]
            )
            if entry.exception_type is not None:
                lines.append(f"{string('debug.exception')}: {entry.exception_type}")
            content = "\n".join(lines)
        if content == self._shown_call_detail:
            self._shown_call_detail_id = entry.entry_id if entry is not None else None
            return

        body.update(Text(content, no_wrap=False) if entry is not None else content)
        self._shown_call_detail = content
        self._shown_call_detail_id = entry.entry_id if entry is not None else None

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
        ui_component.children.append(
            _InspectorComponent(
                path="ui.modals",
                label=string("debug.component_ui_modals"),
                value=(),
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
        is_modal_catalog = component is not None and component.path == "ui.modals"
        self.query_one("#debug-modal-preview").display = is_modal_catalog
        self.query_one("#debug-selected-component").display = not is_modal_catalog
        self.query_one("#debug-state-title").display = not is_modal_catalog
        self.query_one("#debug-state-table").display = not is_modal_catalog
        self.query_one("#debug-attributes-title").display = not is_modal_catalog
        self.query_one("#debug-attributes-table").display = not is_modal_catalog
        if component is None:
            return
        if is_modal_catalog:
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
            worker = string("debug.worker_not_applicable")
        else:
            operation = _short_value(getattr(engine, "operation_state", None)).upper()
            activity = _short_value(getattr(engine, "cur_state", None))
            backend_value = getattr(engine, "backend", None)
            backend_name = getattr(engine, "tts_backend", None)
            if not backend_name:
                backend_name = getattr(backend_value, "name", backend_value)
            backend = _short_value(backend_name)
            worker = _worker_status(backend_value)
        return string(
            "debug.status_format",
            operation=operation,
            activity=activity,
            backend=backend,
            worker=worker,
        )

    def _update_pause_button(self) -> None:
        button = self.query_one("#debug-pause", widgets.Button)
        if not self.trace.available:
            button.disabled = True
            button.label = string("debug.trace_unavailable")
            return
        button.disabled = False
        button.label = string("debug.resume" if self.trace.paused else "debug.pause")


def _classify_source(module_name: Optional[str], source_path: Optional[str]) -> str:
    """Classify a callable source when its module or file identifies it."""
    if not module_name:
        return "unknown"
    root_module = module_name.partition(".")[0]
    if root_module == "celune":
        return "celune"
    if root_module == "builtins":
        return "built_in"
    if root_module in _STANDARD_LIBRARY_MODULES:
        return "standard_library"

    if source_path is None:
        module = sys.modules.get(module_name)
        source_path = getattr(module, "__file__", None)
        if not isinstance(source_path, str):
            specification = getattr(module, "__spec__", None)
            origin = getattr(specification, "origin", None)
            source_path = origin if isinstance(origin, str) else None
    if source_path is None or source_path in {"built-in", "frozen"}:
        return "unknown"
    try:
        path = Path(source_path).resolve()
        if any(path == root or root in path.parents for root in _SITE_PACKAGE_ROOTS):
            return "third_party"
    except (OSError, RuntimeError, ValueError):
        return "unknown"
    return "unknown"


def _codes_from_value(
    value: object,
    visited: set[int],
    module_name: str,
) -> abc.Iterator[CodeType]:
    """Yield code objects owned by a module's functions and classes."""
    value_id = id(value)
    if value_id in visited:
        return
    value_type = type(value)
    if value_type is types.FunctionType:
        function = cast(types.FunctionType, value)
        visited.add(value_id)
        if function.__module__ != module_name:
            return
        yield from _walk_code(function.__code__, set())
        wrapped = getattr(function, "__wrapped__", None)
        wrapped_type = type(wrapped)
        if wrapped_type is types.FunctionType:
            yield from _codes_from_value(wrapped, visited, module_name)
        return
    if value_type is staticmethod:
        visited.add(value_id)
        descriptor = cast(staticmethod, value)
        yield from _codes_from_value(descriptor.__func__, visited, module_name)
        return
    if value_type is classmethod:
        visited.add(value_id)
        descriptor = cast(classmethod, value)
        yield from _codes_from_value(descriptor.__func__, visited, module_name)
        return
    if value_type is property:
        visited.add(value_id)
        descriptor = cast(property, value)
        for accessor in (descriptor.fget, descriptor.fset, descriptor.fdel):
            if accessor is not None:
                yield from _codes_from_value(accessor, visited, module_name)
        return
    if issubclass(value_type, type):
        class_value = cast(type, value)
        class_module = type.__getattribute__(class_value, "__module__")
        if class_module == module_name:
            visited.add(value_id)
            members = type.__getattribute__(class_value, "__dict__").values()
            for member in members:
                yield from _codes_from_value(member, visited, module_name)


def _walk_code(code: CodeType, seen: set[CodeType]) -> abc.Iterator[CodeType]:
    """Yield a code object and nested code constants once each."""
    if code in seen:
        return
    seen.add(code)
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _walk_code(value, seen)


def _frame_module_name(frame: Optional[FrameType], code: CodeType) -> Optional[str]:
    if frame is None or frame.f_code is not code:
        return None
    module_name = frame.f_globals.get("__name__")
    return module_name if isinstance(module_name, str) else None


def _callable_module_name(callable_value: object) -> Optional[str]:
    module_name = getattr(callable_value, "__module__", None)
    if isinstance(module_name, str) and module_name:
        return module_name
    owner = getattr(callable_value, "__objclass__", None)
    module_name = getattr(owner, "__module__", None)
    if isinstance(module_name, str) and module_name:
        return module_name
    module_name = getattr(type(callable_value), "__module__", None)
    return module_name if isinstance(module_name, str) and module_name else None


def _qualified_python_name(
    module_name: Optional[str],
    code: CodeType,
) -> str:
    qualified_name = code.co_qualname
    if not module_name:
        return qualified_name
    return f"{module_name}.{qualified_name}"


def _qualified_callable_name(module_name: Optional[str], callable_value: object) -> str:
    name = getattr(callable_value, "__qualname__", None)
    if not isinstance(name, str):
        name = getattr(callable_value, "__name__", None)
    if not isinstance(name, str):
        name = type(callable_value).__qualname__
    if not module_name:
        return name
    return f"{module_name}.{name}"


@functools.lru_cache(maxsize=1024)
def _builtin_callable_metadata(
    callable_value: types.BuiltinFunctionType,
) -> tuple[Optional[str], str, bool]:
    """Return cached source and display metadata for a built-in callable."""
    module_name = getattr(callable_value, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        module_name = None
    name = _qualified_callable_name(module_name, callable_value)
    return module_name, name, _is_special_method(name.rpartition(".")[2])


@functools.lru_cache(maxsize=256)
def _has_python_call_method(callable_type: type[object]) -> bool:
    """Return whether instances of this type use a Python ``__call__``."""
    call_method = inspect.getattr_static(callable_type, "__call__", None)
    return isinstance(call_method, (types.FunctionType, types.MethodType))


def _is_python_callable(callable_value: object) -> bool:
    if isinstance(callable_value, (types.FunctionType, types.MethodType)):
        return True
    return _has_python_call_method(type(callable_value))


def _module_matches_filters(
    module_name: Optional[str],
    source_filters: tuple[str, ...],
) -> bool:
    if not source_filters:
        return True
    if not module_name:
        return False
    return any(
        module_name == selected or module_name.startswith(f"{selected}.")
        for selected in source_filters
    )


def _compiled_source_filter(source_filters: tuple[str, ...]) -> Optional[str]:
    """Find a selected source covered by compiled code in this Nuitka process."""
    if not source_filters or not running_compiled():
        return None

    compiled_modules = tuple(
        module_name
        for module_name, module in tuple(sys.modules.items())
        if isinstance(module, ModuleType) and bool(vars(module).get("__compiled__"))
    )
    app_package = (__package__ or "").partition(".")[0]
    for source_filter in source_filters:
        if app_package and _module_names_overlap(source_filter, app_package):
            return source_filter
        if any(
            _module_names_overlap(source_filter, module_name)
            for module_name in compiled_modules
        ):
            return source_filter
    return None


def _module_names_overlap(first: str, second: str) -> bool:
    """Return whether two dotted module names share a package boundary."""
    return (
        first == second
        or first.startswith(f"{second}.")
        or second.startswith(f"{first}.")
    )


def _is_special_method(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _is_blacklisted_module(module_name: str) -> bool:
    """Return whether a module belongs to the inspector's excluded UI stack."""
    return any(
        module_name == package or module_name.startswith(f"{package}.")
        for package in DebugCallTrace._INSPECTOR_MODULES
    )


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


def _call_arguments(
    code: CodeType,
    frame: Optional[FrameType],
) -> tuple[tuple[str, str], ...]:
    """Return bounded, readable values for the parameters at function entry."""
    if frame is None or frame.f_code is not code:
        return ()

    parameter_count = code.co_argcount + code.co_kwonlyargcount
    parameter_names = list(code.co_varnames[:parameter_count])
    next_name = parameter_count
    if code.co_flags & _CODE_FLAG_VARARGS:
        parameter_names.append(code.co_varnames[next_name])
        next_name += 1
    if code.co_flags & _CODE_FLAG_VARKEYWORDS:
        parameter_names.append(code.co_varnames[next_name])

    local_values = frame.f_locals
    return tuple(
        (name, _format_value(name, local_values[name]))
        for name in parameter_names
        if name in local_values
    )


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


def _worker_status(backend: object) -> str:
    worker = getattr(backend, "_worker", backend)
    if not hasattr(worker, "_process"):
        return string("debug.worker_not_applicable")

    process = getattr(worker, "_process", None)
    reader = getattr(worker, "_reader_thread", None)
    poll = getattr(process, "poll", None)
    is_alive = getattr(reader, "is_alive", None)
    if (
        process is None
        or reader is None
        or not callable(poll)
        or not callable(is_alive)
    ):
        return string("debug.worker_inactive")
    try:
        active = poll() is None and is_alive()
    except Exception:
        active = False
    return string("debug.worker_active" if active else "debug.worker_inactive")


def _component_signature(
    components: list[_InspectorComponent],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (component.path, id(component.value))
        for component in _iter_components(components)
    )


def _iter_components(
    components: list[_InspectorComponent],
) -> abc.Iterator[_InspectorComponent]:
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


def _format_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%H:%M:%S.%f")[
        :-3
    ]


def _source_category_label(source_category: str) -> str:
    labels = {
        "celune": "debug.source_celune",
        "third_party": "debug.source_third_party",
        "standard_library": "debug.source_standard_library",
        "built_in": "debug.source_built_in",
        "unknown": "debug.source_unknown",
    }
    key = labels.get(source_category, "debug.source_unknown")
    if source_category == "celune":
        return string(key, app_name=APP_NAME)
    return string(key)


def _is_debug_enabled(ui: object) -> bool:
    engine = getattr(ui, "celune", None)
    log_level = getattr(engine, "log_level", None)
    if log_level is None:
        log_level = getattr(ui, "_startup_log_level", "info")
    return log_level == "debug"
