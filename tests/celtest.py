# SPDX-License-Identifier: Apache-2.0
"""Portable friendly pytest terminal reporter plugin.

Integrate this module into any pytest project by loading it from that
project's ``conftest.py`` and decorating tests with ``celtest``::

    # conftest.py
    pytest_plugins = ("path.to.celtest",)

    # test_example.py
    from path.to.celtest import celtest

    @celtest("loads the default voice")
    def test_default_voice() -> None:
        ...

The project header is read from the active project's ``pyproject.toml``::

    [project]
    name = "example-project"
    version = "1.2.3"

    [tool.celtest]
    display_name = "Example Project"

``tool.celtest.display_name`` is optional; ``project.name`` is used when it
is absent. The plugin has no dependency on the host project's package.
"""

from __future__ import annotations

import ast
import re
import sys
import time
import tomllib
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Callable, Generator, Mapping
from typing import Optional, ParamSpec, Protocol, TypeVar, cast

import pytest

_P = ParamSpec("_P")
_R = TypeVar("_R")
_DESCRIPTION_PROPERTY = "celtest_description"
_HINT_PROPERTY = "celtest_hint"
_SKIP_PROPERTY = "celtest_known_skip"
_PARALLEL_SETUP_LABEL = "setting up parallel test harness"
_PARALLEL_SETUP_SOURCE = "bringing up nodes..."
_ERROR_SYMBOL = "E"
_FAILURE_SYMBOL = "F"
_NOT_RUN_SYMBOL = "N"
_PASS_SYMBOL = "."
_PROCESSING_SYMBOL = "?"
_SKIP_SYMBOL = "S"
_WARNING_SYMBOL = "W"
_ACRONYMS = {
    "ansi": "ANSI",
    "api": "API",
    "asr": "ASR",
    "cedts": "CEDTS",
    "cechar": "CECHAR",
    "cevoice": "CEVOICE",
    "ci": "CI",
    "cpu": "CPU",
    "cuda": "CUDA",
    "dsp": "DSP",
    "eof": "EOF",
    "flac": "FLAC",
    "gpu": "GPU",
    "hf": "HF",
    "http": "HTTP",
    "json": "JSON",
    "pcm": "PCM",
    "ram": "RAM",
    "rms": "RMS",
    "tts": "TTS",
    "tui": "TUI",
    "ui": "UI",
    "url": "URL",
    "utf8": "UTF-8",
    "vc": "VC",
    "vram": "VRAM",
    "wav": "WAV",
    "vlm": "VLM",
    "webui": "WebUI",
    "xdist": "xdist",
    "yaml": "YAML",
    "zip": "ZIP",
}


class _TerminalWriter(Protocol):
    """Small terminal-writer surface used by the presentation plugin."""

    def write(
        self,
        message: str,
        *,
        flush: bool = False,
        **markup: bool,
    ) -> None:
        """Write terminal text."""

    def line(self) -> None:
        """Write a terminal newline."""


class _TerminalReporter(Protocol):
    """Small terminal-reporter surface used by the presentation plugin."""

    _tw: _TerminalWriter

    def write_line(self, line: str) -> None:
        """Write one complete terminal line."""

    def rewrite(self, line: str, **markup: bool) -> None:
        """Rewrite the current terminal line."""


@dataclass(frozen=True)
class CeltestMetadata:
    """Friendly presentation metadata attached to one test callable."""

    description: str
    hint: Optional[str] = None


@dataclass(frozen=True)
class _ProjectMetadata:
    """Application metadata loaded from one project's TOML document."""

    display_name: str
    version: str


def _toml_section(value: object) -> Optional[Mapping[str, object]]:
    """Return one TOML table when the parsed value has table shape."""
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return None


def _toml_text(section: Optional[Mapping[str, object]], key: str) -> Optional[str]:
    """Return one non-empty TOML string value from a table."""
    if section is None:
        return None
    value = section.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _project_metadata(rootpath: Optional[Path]) -> _ProjectMetadata:
    """Load the display name and version from the project's pyproject file."""
    candidates = [rootpath / "pyproject.toml"] if rootpath is not None else []
    working_file = Path.cwd() / "pyproject.toml"
    if working_file not in candidates:
        candidates.append(working_file)
    for candidate in candidates:
        try:
            with candidate.open("rb") as file:
                document = cast(Mapping[str, object], tomllib.load(file))
        except (OSError, tomllib.TOMLDecodeError):
            continue

        project = _toml_section(document.get("project"))
        tools = _toml_section(document.get("tool"))
        celtest_tool = _toml_section(tools.get("celtest")) if tools else None
        project_name = _toml_text(project, "name")
        display_name = _toml_text(celtest_tool, "display_name") or project_name
        version = _toml_text(project, "version")
        return _ProjectMetadata(display_name or "project", version or "unknown")

    return _ProjectMetadata("project", "unknown")


def celtest(
    description: str,
    *,
    hint: Optional[str] = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a test with its friendly terminal description and failure hint."""

    metadata = CeltestMetadata(description, hint)

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        """Attach Celtest metadata without wrapping the pytest callable."""
        function.__dict__["__celtest_metadata__"] = metadata
        return function

    return decorate


@dataclass
class _TestRecord:
    """Presentation state for one collected test item."""

    nodeid: str
    description: str
    hint: Optional[str]
    known_skipped: bool = False
    reports: dict[str, pytest.TestReport] = field(default_factory=dict)
    warning_lines: list[str] = field(default_factory=list)
    finished: bool = False
    status: Optional[str] = None
    failure_report: Optional[pytest.TestReport] = None
    processing_written: bool = False


@dataclass
class _PluginState:
    """Process-local state collected by the reporter hooks."""

    started_at: float = field(default_factory=time.monotonic)
    records: dict[str, _TestRecord] = field(default_factory=dict)
    warning_lines: list[str] = field(default_factory=list)
    collection_errors: list[tuple[str, str]] = field(default_factory=list)
    internal_errors: list[str] = field(default_factory=list)
    pending_records: set[str] = field(default_factory=set)
    source_trees: dict[str, Optional[ast.Module]] = field(default_factory=dict)
    rootpath: Optional[Path] = None
    project_metadata: _ProjectMetadata = field(
        default_factory=lambda: _ProjectMetadata("project", "unknown")
    )
    terminalreporter: Optional[_TerminalReporter] = None
    is_controller: bool = True
    parallel: bool = False
    verbose: bool = False
    replace_lines: bool = False
    live_line_width: int = 0


_state = _PluginState()


def _friendly_name(value: str) -> str:
    """Turn a pytest function name or node ID into a readable fallback."""
    function_name = value.rsplit("::", 1)[-1]
    function_name = re.sub(r"\[[^]]*\]$", "", function_name)
    function_name = re.sub(r"^test_", "", function_name)
    words = re.sub(r"[_-]+", " ", function_name).strip().split()
    if not words:
        return "Test"
    formatted_words = []
    for index, word in enumerate(words):
        formatted_words.append(
            _ACRONYMS.get(
                word.lower(), word[:1].upper() + word[1:] if index == 0 else word
            )
        )
    return " ".join(formatted_words)


def _docstring_description(value: str) -> str:
    """Normalize a legacy docstring description without changing its terms."""
    description = value.strip().removesuffix(".")
    if not description:
        return description

    first_word, separator, remainder = description.partition(" ")
    normalized_first_word = _ACRONYMS.get(
        first_word.lower(),
        first_word[:1].lower() + first_word[1:],
    )
    return normalized_first_word + (separator + remainder if separator else "")


def _metadata_for(item: pytest.Item) -> CeltestMetadata:
    """Resolve decorator metadata, then use the existing test documentation."""
    function = getattr(item, "obj", None)
    metadata = getattr(function, "__celtest_metadata__", None)
    if isinstance(metadata, CeltestMetadata):
        return metadata

    class_metadata = getattr(getattr(item, "parent", None), "obj", None)
    metadata = getattr(class_metadata, "__celtest_metadata__", None)
    if isinstance(metadata, CeltestMetadata):
        return metadata

    docstring = getattr(function, "__doc__", None)
    if isinstance(docstring, str):
        first_line = next(
            (line.strip() for line in docstring.splitlines() if line.strip()),
            "",
        )
        if first_line:
            return CeltestMetadata(_docstring_description(first_line))
    return CeltestMetadata(_friendly_name(item.nodeid))


def _record_for_report(report: pytest.TestReport) -> _TestRecord:
    """Create a controller record from a report received from xdist."""
    properties = dict(report.user_properties)
    description_value = properties.get(_DESCRIPTION_PROPERTY)
    description = (
        description_value
        if isinstance(description_value, str)
        else _friendly_name(report.nodeid)
    )
    hint_value = properties.get(_HINT_PROPERTY)
    hint = hint_value if isinstance(hint_value, str) and hint_value else None
    known_skipped = properties.get(_SKIP_PROPERTY) == "true"
    record = _state.records.get(report.nodeid)
    if record is None:
        record = _TestRecord(
            nodeid=report.nodeid,
            description=description,
            hint=hint,
            known_skipped=known_skipped,
        )
        _state.records[report.nodeid] = record
    elif properties:
        record.description = description
        record.hint = hint
        record.known_skipped = known_skipped
    return record


def _decorator_metadata(node: ast.AST) -> Optional[CeltestMetadata]:
    """Read a literal ``celtest`` decorator from a syntax-tree node."""
    for decorator in getattr(node, "decorator_list", []):
        if not isinstance(decorator, ast.Call):
            continue
        function = decorator.func
        if not (
            isinstance(function, ast.Name)
            and function.id == "celtest"
            or isinstance(function, ast.Attribute)
            and function.attr == "celtest"
        ):
            continue
        if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
            continue
        description = decorator.args[0].value
        if not isinstance(description, str):
            continue
        hint_node = next(
            (keyword.value for keyword in decorator.keywords if keyword.arg == "hint"),
            None,
        )
        hint = (
            hint_node.value
            if isinstance(hint_node, ast.Constant) and isinstance(hint_node.value, str)
            else None
        )
        return CeltestMetadata(description, hint)
    return None


def _source_metadata(
    nodeid: str,
    location: tuple[str, int, str],
) -> Optional[CeltestMetadata]:
    """Resolve decorator or docstring metadata for a remote xdist item."""
    parts = nodeid.split("::")[1:]
    source_path = Path(location[0])
    if not source_path.is_absolute() and _state.rootpath is not None:
        source_path = _state.rootpath / source_path
    source_key = str(source_path)
    if source_key not in _state.source_trees:
        try:
            _state.source_trees[source_key] = ast.parse(
                source_path.read_text(encoding="utf-8")
            )
        except (OSError, SyntaxError):
            _state.source_trees[source_key] = None
    tree = _state.source_trees[source_key]
    if tree is None:
        return None

    body: list[ast.stmt] = tree.body
    current: Optional[ast.AST] = None
    for raw_part in parts:
        part = re.sub(r"\[[^]]*\]$", "", raw_part)
        current = next(
            (
                candidate
                for candidate in body
                if isinstance(
                    candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and candidate.name == part
            ),
            None,
        )
        if current is None:
            return None
        body = current.body
    if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        metadata = _decorator_metadata(current)
        if metadata is not None:
            return metadata
        docstring = ast.get_docstring(current)
        if docstring:
            return CeltestMetadata(_docstring_description(docstring.splitlines()[0]))
    return None


def _is_known_skip(item: pytest.Item) -> bool:
    """Recognize skip markers whose condition is already known at collection."""
    if item.get_closest_marker("skip") is not None:
        return True
    return any(
        marker.args and marker.args[0] is True for marker in item.iter_markers("skipif")
    )


def _terminal_writer() -> Optional[_TerminalWriter]:
    """Return pytest's terminal writer when the terminal plugin is available."""
    terminalreporter = _state.terminalreporter
    return getattr(terminalreporter, "_tw", None)


def _terminal_markup(color: Optional[str]) -> dict[str, bool]:
    """Return color markup only when the terminal supports ANSI styling."""
    writer = _terminal_writer()
    if color is None or writer is None or not getattr(writer, "hasmarkup", False):
        return {}
    return {color: True}


def _terminal_supports_ansi() -> bool:
    """Return whether the terminal writer supports ANSI styling and controls."""
    writer = _terminal_writer()
    return writer is not None and bool(getattr(writer, "hasmarkup", False))


def _write_line(value: str, *, color: Optional[str] = None) -> None:
    """Write one custom line through pytest's terminal reporter."""
    if not _state.is_controller:
        return
    terminalreporter = _state.terminalreporter
    if terminalreporter is not None:
        markup = _terminal_markup(color)
        if markup:
            writer = _terminal_writer()
            if writer is not None:
                writer.write(value, **markup)
                writer.line()
                return
        terminalreporter.write_line(value)


def _write_live_line(
    value: str,
    *,
    replace: bool,
    complete: bool = True,
    color: Optional[str] = None,
) -> None:
    """Write a processing or final result line with carriage-return replacement."""
    writer = _terminal_writer()
    if not replace or writer is None or not _terminal_supports_ansi():
        _write_line(value, color=color)
        if complete:
            _state.live_line_width = 0
        return
    line_width = max(_state.live_line_width, len(value))
    padded = f"{value}{' ' * (line_width - len(value))}"
    suffix = "\n" if complete else ""
    markup = _terminal_markup(color)
    writer.write(f"\r{padded}{suffix}", flush=True, **markup)
    _state.live_line_width = 0 if complete else line_width


def _compact_symbol(record: _TestRecord) -> str:
    """Return the inline icon for one compact test status."""
    symbols = {
        "error": _ERROR_SYMBOL,
        "failed": _FAILURE_SYMBOL,
        "not_run": _NOT_RUN_SYMBOL,
        "passed": _PASS_SYMBOL,
        "skipped": _SKIP_SYMBOL,
        "warning": _WARNING_SYMBOL,
    }
    return symbols.get(record.status or "processing", _PROCESSING_SYMBOL)


def _compact_color(record: _TestRecord) -> Optional[str]:
    """Return the terminal color for one compact test status."""
    if record.status is None:
        return None
    colors = {
        "error": "red",
        "failed": "red",
        "not_run": "blue",
        "passed": "green",
        "skipped": "blue",
        "warning": "yellow",
    }
    return colors.get(record.status)


def _write_compact_output(
    value: str,
    *,
    color: Optional[str] = None,
    flush: bool = True,
) -> None:
    """Write compact output inline through pytest or the captured stdout."""
    writer = _terminal_writer()
    if writer is not None:
        markup = _terminal_markup(color)
        writer.write(value, flush=flush, **markup)
        return
    sys.stdout.write(value)
    if flush:
        sys.stdout.flush()


def _write_compact_result_icon(record: _TestRecord) -> None:
    """Append one final compact status icon without redrawing prior output."""
    _write_compact_output(_compact_symbol(record), color=_compact_color(record))


def _finish_compact_status_line() -> None:
    """End the inline compact status line before printing the totals."""
    _write_compact_output("\n")


def _write_compact_summary_details(
    runnable: list[_TestRecord],
) -> None:
    """Write concise failures and warnings below the compact totals."""
    failures = [record for record in runnable if record.status == "failed"]
    errors = [record for record in runnable if record.status == "error"]
    if failures:
        _write_line("")
        _write_line(f"{_FAILURE_SYMBOL} failures", color="red")
        for record in failures:
            detail = record.hint or _failure_message(record.failure_report)
            _write_line(f"{record.description}: {detail}")

    if errors:
        _write_line("")
        _write_line(f"{_ERROR_SYMBOL} errors", color="red")
        for record in errors:
            detail = record.hint or _failure_message(record.failure_report)
            _write_line(f"{record.description}: {detail}")

    if _state.warning_lines:
        _write_line("")
        _write_line(f"{_WARNING_SYMBOL} warnings", color="yellow")
        for warning_line in _state.warning_lines:
            _write_line(warning_line)


def _install_parallel_setup_label(terminalreporter: _TerminalReporter) -> None:
    """Translate xdist's worker-startup label while retaining its behavior."""
    write_line = terminalreporter.write_line
    rewrite = terminalreporter.rewrite

    def translated_line(line: str) -> None:
        write_line(line.replace(_PARALLEL_SETUP_SOURCE, _PARALLEL_SETUP_LABEL))

    def translated_rewrite(line: str, **markup: bool) -> None:
        rewrite(line.replace(_PARALLEL_SETUP_SOURCE, _PARALLEL_SETUP_LABEL), **markup)

    terminalreporter.write_line = translated_line
    terminalreporter.rewrite = translated_rewrite


def _format_duration(seconds: float) -> str:
    """Format elapsed test time as minutes and zero-padded seconds."""
    elapsed = max(0, int(seconds))
    return f"{elapsed // 60}:{elapsed % 60:02d}"


def _failure_message(report: Optional[object]) -> str:
    """Extract an assertion or exception message without stack frames."""
    if report is None:
        return "The test stopped before pytest reported its failure."
    longrepr = getattr(report, "longrepr", report)
    crash = getattr(longrepr, "reprcrash", None)
    message = getattr(crash, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    for line in str(longrepr).splitlines():
        candidate = line.strip()
        candidate = re.sub(r"^[E>]\s+", "", candidate)
        if not candidate or candidate.startswith(
            ("Traceback", "File ", "^", "During handling")
        ):
            continue
        if re.search(r"(?:Error|Exception|Warning):", candidate):
            return candidate
        if candidate.startswith("assert "):
            return candidate
    return "Pytest reported a failure without an assertion message."


def _is_assertion_failure(report: pytest.TestReport) -> bool:
    """Return whether a failed call report represents a test assertion."""
    if report.when != "call":
        return False
    message = _failure_message(report)
    return message.startswith(("assert ", "AssertionError", "Failed"))


def _collection_name(nodeid: str) -> str:
    """Return the concise module name for a collection failure."""
    return re.split(r"[\\/]", nodeid.rsplit("::", 1)[0])[-1]


def _warning_text(
    warning_message: warnings.WarningMessage,
    nodeid: Optional[str],
) -> str:
    """Render one warning reported by pytest without changing its content."""
    formatted = warnings.formatwarning(
        str(warning_message.message),
        warning_message.category,
        warning_message.filename,
        warning_message.lineno,
        warning_message.line,
    ).rstrip()
    if nodeid and formatted:
        return f"{nodeid}: {formatted}"
    return formatted or f"{nodeid or 'test setup'}: warning"


def _write_failure_traceback(report: pytest.TestReport) -> None:
    """Write pytest's original failure representation after its result line."""
    if not _state.verbose:
        return
    writer = _terminal_writer()
    if writer is None:
        return
    longrepr = report.longrepr
    toterminal = getattr(longrepr, "toterminal", None)
    if callable(toterminal):
        toterminal(writer)
        writer.line()
    else:
        writer.write(f"{longrepr}\n", flush=True)


def _finish_record(record: _TestRecord) -> None:
    """Emit one final result after all setup, call, and teardown reports arrive."""
    if record.finished:
        return
    record.finished = True

    failures = [report for report in record.reports.values() if report.failed]
    if failures:
        record.failure_report = failures[0]
        record.status = (
            "failed" if _is_assertion_failure(record.failure_report) else "error"
        )
    elif any(report.skipped for report in record.reports.values()):
        record.status = "skipped"
    elif not {"setup", "call", "teardown"}.issubset(record.reports):
        record.status = "not_run"
        return
    else:
        record.status = "warning" if record.warning_lines else "passed"

    if not _state.verbose:
        if _state.is_controller:
            _write_compact_result_icon(record)
        return
    if record.status == "skipped":
        return

    symbol = _compact_symbol(record)
    _write_live_line(
        f"{symbol} {record.description}",
        replace=_state.replace_lines and record.processing_written,
        color=_compact_color(record),
    )
    if record.failure_report is not None:
        _write_failure_traceback(record.failure_report)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest to keep its internals while replacing terminal output."""
    global _state
    _state = _PluginState()
    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    _state.terminalreporter = cast(Optional[_TerminalReporter], terminalreporter)
    _state.is_controller = not hasattr(config, "workerinput")
    _state.rootpath = config.rootpath
    _state.project_metadata = _project_metadata(config.rootpath)
    _state.verbose = getattr(config.option, "verbose", 0) > 0
    worker_count = getattr(config.option, "numprocesses", 0)
    parallel = worker_count not in {0, 1, None, "0", "1"}
    _state.parallel = parallel
    _state.replace_lines = _state.is_controller


def pytest_sessionstart(session: pytest.Session) -> None:
    """Print the authoritative Celune application header before tests run."""
    config = session.config
    config.option.no_header = True
    config.option.no_summary = True
    config.option.verbose = -2
    _state.terminalreporter = cast(
        Optional[_TerminalReporter],
        config.pluginmanager.get_plugin("terminalreporter"),
    )
    if (
        _state.verbose
        and _state.is_controller
        and _state.parallel
        and _state.terminalreporter is not None
    ):
        _install_parallel_setup_label(_state.terminalreporter)
    if not _state.is_controller:
        return
    _write_line(
        f"testing {_state.project_metadata.display_name} "
        f"{_state.project_metadata.version}"
    )
    _write_line("")


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Register friendly metadata without removing or rewriting collected items."""
    del session, config
    for item in items:
        metadata = _metadata_for(item)
        _state.records[item.nodeid] = _TestRecord(
            nodeid=item.nodeid,
            description=metadata.description,
            hint=metadata.hint,
            known_skipped=_is_known_skip(item),
        )


def pytest_deselected(items: list[pytest.Item]) -> None:
    """Remove items filtered out by pytest before the run starts."""
    for item in items:
        _state.records.pop(item.nodeid, None)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int, str]) -> None:
    """Show a processing line when a non-skipped test starts in verbose mode."""
    for pending_nodeid in tuple(_state.pending_records):
        pending = _state.records.get(pending_nodeid)
        if pending is not None:
            _finish_record(pending)
        _state.pending_records.discard(pending_nodeid)
    record = _state.records.get(nodeid)
    if record is None and _state.is_controller:
        metadata = _source_metadata(nodeid, location)
        record = _TestRecord(
            nodeid,
            metadata.description if metadata is not None else _friendly_name(nodeid),
            metadata.hint if metadata is not None else None,
        )
        _state.records[nodeid] = record
    del location
    if record is None:
        return
    if not _state.verbose or not _state.is_controller:
        return
    if record.known_skipped:
        return
    if not _terminal_supports_ansi():
        return
    record.processing_written = True
    _write_live_line(
        f"{_PROCESSING_SYMBOL} {record.description}",
        replace=_state.replace_lines,
        complete=False,
    )


def pytest_warning_recorded(
    warning_message: warnings.WarningMessage,
    when: str,
    nodeid: Optional[str],
    location: Optional[tuple[str, int, str]],
) -> None:
    """Collect warnings pytest reports during setup, call, or teardown."""
    if when != "runtest":
        return
    warning_line = _warning_text(
        warning_message, nodeid or (location[0] if location else None)
    )
    _state.warning_lines.append(warning_line)
    if nodeid is not None and nodeid in _state.records:
        _state.records[nodeid].warning_lines.append(warning_line)
    if _state.is_controller and nodeid is not None and nodeid in _state.pending_records:
        _finish_record(_state.records[nodeid])
        _state.pending_records.discard(nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Store phase reports until pytest has completed the test protocol."""
    record = _state.records.get(report.nodeid)
    if record is None and _state.is_controller:
        record = _record_for_report(report)
    if record is None:
        return
    if not _state.is_controller and report.when == "setup":
        report.user_properties.extend(
            [
                (_DESCRIPTION_PROPERTY, record.description),
                (_HINT_PROPERTY, record.hint or ""),
                (_SKIP_PROPERTY, "true" if record.known_skipped else "false"),
            ]
        )
    record.reports[report.when] = report


@pytest.hookimpl(tryfirst=True)
def pytest_report_to_serializable(
    config: pytest.Config,
    report: pytest.TestReport,
) -> Optional[dict[str, object]]:
    """Include friendly metadata when xdist serializes a worker report."""
    del config
    if _state.is_controller:
        return None
    record = _state.records.get(report.nodeid)
    if record is None:
        return None
    result = report._to_json()
    result["$report_type"] = "TestReport"
    result["user_properties"].extend(
        [
            (_DESCRIPTION_PROPERTY, record.description),
            (_HINT_PROPERTY, record.hint or ""),
            (_SKIP_PROPERTY, "true" if record.known_skipped else "false"),
        ]
    )
    return result


def pytest_runtest_logfinish(
    nodeid: str,
    location: tuple[str, int, str],
) -> None:
    """Finish controller records after a remote test protocol completes."""
    del location
    if _state.is_controller and _state.parallel:
        _state.pending_records.add(nodeid)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item,
    nextitem: Optional[pytest.Item],
) -> Generator[None, object, object]:
    """Finish a test after pytest's warning-capture wrapper has reported warnings."""
    del nextitem
    result = yield
    record = _state.records.get(item.nodeid)
    if record is not None:
        _finish_record(record)
    return result


def pytest_report_teststatus(
    report: pytest.TestReport,
    config: pytest.Config,
) -> tuple[str, str, str]:
    """Suppress pytest's progress characters while retaining its report hooks."""
    del config
    if report.when in {"setup", "teardown"}:
        if report.failed:
            return "error", "", ""
        if report.skipped:
            return "skipped", "", ""
        return "", "", ""
    if report.failed:
        return "failed", "", ""
    if report.skipped:
        return "skipped", "", ""
    return "passed", "", ""


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Record collection failures for the concise fatal-error presentation."""
    if report.failed:
        _state.collection_errors.append(
            (
                report.nodeid or report.fspath,
                _failure_message(report),
            )
        )


@pytest.hookimpl(tryfirst=True)
def pytest_internalerror(excrepr: object, excinfo: object) -> bool:
    """Record fatal pytest errors without printing their trace during the run."""
    del excinfo
    _state.internal_errors.append(_failure_message(excrepr))
    return True


def _render_summary(
    exitstatus: pytest.ExitCode,
    fatal_hint: Optional[str],
) -> None:
    """Print the compact summary after all pytest reports are available."""
    if not _state.is_controller:
        return
    for record in _state.records.values():
        _finish_record(record)

    if _state.collection_errors:
        _write_line("")
        _write_line("test collection failed")
        module_names = [
            _collection_name(nodeid) for nodeid, _hint in _state.collection_errors
        ]
        if not module_names:
            module_names = [
                _collection_name(record.nodeid) for record in _state.records.values()
            ] or ["pytest internal error"]
        for nodeid in dict.fromkeys(module_names):
            _write_line(f"{_ERROR_SYMBOL} {nodeid}", color="red")
        _write_line("")
        _write_line("test collection failure hint")
        hints = [hint for _nodeid, hint in _state.collection_errors]
        hints.extend(_state.internal_errors)
        if fatal_hint:
            hints.append(fatal_hint)
        _write_line(hints[0] if hints else "Pytest stopped before tests could run.")
        return

    fatal_exit = exitstatus not in {
        pytest.ExitCode.OK,
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
    }
    if _state.internal_errors or fatal_exit:
        _write_line("")
        _write_line("pytest run failed")
        module_names = [
            _collection_name(record.nodeid) for record in _state.records.values()
        ] or ["pytest internal error"]
        for nodeid in dict.fromkeys(module_names):
            _write_line(f"{_ERROR_SYMBOL} {nodeid}", color="red")
        _write_line("")
        _write_line("pytest failure hint")
        hints = list(_state.internal_errors)
        if fatal_hint:
            hints.append(fatal_hint)
        _write_line(hints[0] if hints else "Pytest stopped before the run completed.")
        return

    interrupted = exitstatus == pytest.ExitCode.INTERRUPTED
    if interrupted:
        _write_line("")
        _write_line("interrupted")
        return

    runnable = [
        record for record in _state.records.values() if record.status != "skipped"
    ]
    passed = sum(record.status in {"passed", "warning"} for record in runnable)
    not_run = [record for record in runnable if record.status == "not_run"]
    elapsed = time.monotonic() - _state.started_at

    if not _state.verbose:
        _finish_compact_status_line()
        _write_line("")
        _write_line(
            f"passed {passed}/{len(runnable)} time {_format_duration(elapsed)} "
            f"warnings {len(_state.warning_lines)}"
        )
        _write_compact_summary_details(runnable)
        if not_run:
            _write_line("")
            _write_line(f"{_NOT_RUN_SYMBOL} not run {len(not_run)}", color="blue")
        return

    _write_line("")
    _write_line(
        f"passed {passed}/{len(runnable)} time {_format_duration(elapsed)} "
        f"warnings {len(_state.warning_lines)}"
    )

    if not_run:
        _write_line("")
        _write_line(f"not run {len(not_run)}")
        for record in not_run:
            _write_line(f"{_NOT_RUN_SYMBOL} {record.description}", color="blue")

    if _state.warning_lines:
        _write_line("")
        _write_line(f"{_WARNING_SYMBOL} warnings", color="yellow")
        for warning_line in _state.warning_lines:
            _write_line(warning_line)

    failures = [record for record in runnable if record.status in {"failed", "error"}]
    if failures:
        _write_line("")
        _write_line("test failure hint")
        for record in failures:
            hint = record.hint or _failure_message(record.failure_report)
            _write_line(f"{record.description}: {hint}")


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(
    session: pytest.Session,
    exitstatus: pytest.ExitCode,
) -> Generator[None, object, object]:
    """Suppress pytest's terminal summary and print Celtest's summary."""
    result = yield
    stop_reason = session.shouldstop or session.shouldfail
    fatal_hint = str(stop_reason) if stop_reason else None
    session.shouldfail = ""
    session.shouldstop = ""
    session.config.option.no_summary = True
    session.config.option.verbose = -2
    terminalreporter = _state.terminalreporter
    if terminalreporter is not None:
        terminalreporter.__dict__["_report_keyboardinterrupt"] = lambda: None
    _render_summary(exitstatus, fatal_hint)
    return result
