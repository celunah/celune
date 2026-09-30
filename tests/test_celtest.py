# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's friendly pytest plugin."""

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import cast
from pathlib import Path
from types import SimpleNamespace

import tests.celtest as celtest_plugin
from tests.celtest import CeltestMetadata, celtest
from tests.support import CeluneTestCase


class TestCeltest(CeluneTestCase):
    """Tests for the Celtest decorator and pytest presentation."""

    def test_celtest_attaches_metadata_without_wrapping_the_test(self) -> None:
        """Verify the decorator preserves the test callable and stores its metadata."""

        description = "keeps lowercase text without adding punctuation"

        @celtest(description, hint="Explain the expected state")
        def sample() -> None:
            """A decorated sample test."""

        metadata = sample.__dict__["__celtest_metadata__"]

        assert isinstance(metadata, CeltestMetadata)
        assert metadata.description == description
        assert metadata.hint == "Explain the expected state"
        assert sample.__name__ == "sample"

    def test_celtest_preserves_acronyms_in_fallback_names(self) -> None:
        """Verify generated names keep common Celune acronyms capitalized."""
        assert (
            celtest_plugin._friendly_name("test_ui_uses_tts_and_vram")
            == "UI uses TTS and VRAM"
        )

    def test_celtest_normalizes_legacy_docstring_descriptions(self) -> None:
        """Verify docstring fallback descriptions use lower case without a period."""
        assert (
            celtest_plugin._docstring_description(
                "Verify custom packs named Celune use TTS."
            )
            == "verify custom packs named Celune use TTS"
        )

    def test_celtest_reads_project_metadata_and_falls_back_to_project_name(
        self, tmp_path: Path
    ) -> None:
        """Verify header metadata comes only from each project's pyproject."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "portable-project"
version = "1.2.3"

[tool.celtest]
display_name = "Portable Project"
""",
            encoding="utf-8",
        )

        assert celtest_plugin._project_metadata(tmp_path) == (
            celtest_plugin._ProjectMetadata("Portable Project", "1.2.3")
        )

        pyproject.write_text(
            """
[project]
name = "fallback-project"
version = "4.5.6"
""",
            encoding="utf-8",
        )

        assert celtest_plugin._project_metadata(tmp_path) == (
            celtest_plugin._ProjectMetadata("fallback-project", "4.5.6")
        )

    def test_celtest_replaces_processing_line_with_carriage_return(self) -> None:
        """Verify replacement uses carriage return and fixed-width padding."""
        messages: list[str] = []

        class Writer:
            """Capture terminal writes for the replacement assertion."""

            hasmarkup = True

            def write(
                self,
                message: str,
                *,
                flush: bool = False,
                **markup: bool,
            ) -> None:
                """Capture a terminal write."""
                del flush, markup
                messages.append(message)

            def line(self) -> None:
                """Capture a terminal newline."""
                messages.append("\n")

        writer = Writer()
        reporter = cast(
            celtest_plugin._TerminalReporter,
            SimpleNamespace(_tw=writer),
        )
        previous_state = celtest_plugin._state
        celtest_plugin._state = celtest_plugin._PluginState(
            terminalreporter=reporter,
            replace_lines=True,
        )
        try:
            celtest_plugin._write_live_line("? working", replace=True, complete=False)
            celtest_plugin._write_live_line(". working", replace=True)
        finally:
            celtest_plugin._state = previous_state

        assert messages == ["\r? working", "\r. working\n"]

    def test_celtest_appends_compact_results_without_redrawing(self) -> None:
        """Verify compact results do not emit carriage-return redraws."""
        messages: list[str] = []

        class Writer:
            """Capture compact output writes."""

            def write(
                self,
                message: str,
                *,
                flush: bool = False,
                **markup: bool,
            ) -> None:
                """Capture one compact output write."""
                del flush, markup
                messages.append(message)

            def line(self) -> None:
                """Capture a terminal newline."""
                messages.append("\n")

        writer = Writer()
        reporter = cast(
            celtest_plugin._TerminalReporter,
            SimpleNamespace(_tw=writer),
        )
        previous_state = celtest_plugin._state
        celtest_plugin._state = celtest_plugin._PluginState(
            terminalreporter=reporter,
            is_controller=True,
        )
        record = celtest_plugin._TestRecord("test_pass", "pass", None)
        record.status = "passed"
        try:
            celtest_plugin._write_compact_result_icon(record)
        finally:
            celtest_plugin._state = previous_state

        assert messages == ["."]

    def test_celtest_non_ansi_compact_fallback_emits_plain_final_result(self) -> None:
        """Verify non-ANSI compact output emits no color or processing marker."""
        writes: list[tuple[str, dict[str, bool]]] = []

        class Writer:
            """Capture output from a writer without ANSI markup support."""

            hasmarkup = False

            def write(
                self,
                message: str,
                *,
                flush: bool = False,
                **markup: bool,
            ) -> None:
                """Capture one output write and its markup."""
                del flush
                writes.append((message, markup))

            def line(self) -> None:
                """Capture a terminal newline."""

        reporter = cast(
            celtest_plugin._TerminalReporter,
            SimpleNamespace(_tw=Writer()),
        )
        previous_state = celtest_plugin._state
        celtest_plugin._state = celtest_plugin._PluginState(
            terminalreporter=reporter,
            is_controller=True,
        )
        record = celtest_plugin._TestRecord("test_error", "error", None)
        record.status = "error"
        try:
            celtest_plugin._write_compact_result_icon(record)
        finally:
            celtest_plugin._state = previous_state

        assert writes == [("E", {})]

    def test_celtest_non_ansi_verbose_fallback_emits_one_plain_line(self) -> None:
        """Verify non-ANSI verbose output uses a final newline without redraws."""
        writes: list[str] = []
        lines: list[str] = []

        class Writer:
            """Capture output from a writer without ANSI markup support."""

            hasmarkup = False

            def write(
                self,
                message: str,
                *,
                flush: bool = False,
                **markup: bool,
            ) -> None:
                """Capture unexpected direct writer output."""
                del flush, markup
                writes.append(message)

            def line(self) -> None:
                """Capture an unexpected direct writer newline."""
                writes.append("\n")

        class Reporter:
            """Capture plain reporter lines."""

            _tw = Writer()

            def write_line(self, value: str) -> None:
                """Capture one newline-terminated reporter line."""
                lines.append(f"{value}\n")

        previous_state = celtest_plugin._state
        celtest_plugin._state = celtest_plugin._PluginState(
            terminalreporter=cast(celtest_plugin._TerminalReporter, Reporter()),
            is_controller=True,
            verbose=True,
        )
        try:
            celtest_plugin._write_live_line(
                "E failed test",
                replace=True,
                color="red",
            )
        finally:
            celtest_plugin._state = previous_state

        assert not writes
        assert lines == ["E failed test\n"]

    def test_celtest_formats_verbose_pass_warning_failure_and_skip(
        self, tmp_path: Path
    ) -> None:
        """Verify verbose mode retains the detailed result contract."""
        test_file = tmp_path / "test_sample.py"
        test_file.write_text(
            """
import pytest
import warnings

from tests.celtest import celtest

@celtest("friendly pass")
def test_pass():
    pass

@celtest("friendly warning")
def test_warning():
    warnings.warn("sample warning", UserWarning)

@pytest.mark.skip(reason="not part of this run")
@celtest("hidden skip")
def test_skip():
    pass

@celtest("friendly failure", hint="The sample assertion was false.")
def test_failure():
    warnings.warn("failure warning", UserWarning)
    assert "left" == "right"

@celtest("friendly error")
def test_error():
    raise RuntimeError("sample runtime error")
""",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = project_root
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.celtest",
                "-v",
                str(test_file),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=environment,
        )
        output = result.stdout + result.stderr

        assert result.returncode == 1
        metadata = celtest_plugin._project_metadata(Path(project_root))
        assert f"testing {metadata.display_name} {metadata.version}" in output
        assert "? friendly pass" not in output
        assert ". friendly pass" in output
        assert "? friendly warning" not in output
        assert "W friendly warning" in output
        assert "F friendly failure" in output
        assert "hidden skip" not in output
        assert "passed 2/4" in output
        assert "E friendly error" in output
        assert "warnings 2" in output
        assert "UserWarning: sample warning" in output
        assert "UserWarning: failure warning" in output
        assert "The sample assertion was false." in output
        assert "Traceback" not in output
        assert "assert 'left' == 'right'" in output
        assert "\r" not in output

    def test_celtest_formats_non_verbose_results_without_details(
        self, tmp_path: Path
    ) -> None:
        """Verify non-verbose output uses compact progress and concise details."""
        test_file = tmp_path / "test_sample.py"
        test_file.write_text(
            """
import pytest
import warnings

from tests.celtest import celtest

@celtest("friendly pass")
def test_pass():
    pass

@celtest("friendly warning")
def test_warning():
    warnings.warn("sample warning", UserWarning)

@pytest.mark.skip(reason="not part of this run")
@celtest("hidden skip")
def test_skip():
    pass

@celtest("friendly failure", hint="The sample assertion was false.")
def test_failure():
    warnings.warn("failure warning", UserWarning)
    assert "left" == "right"

@celtest("friendly error")
def test_error():
    raise RuntimeError("sample runtime error")
""",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = project_root
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.celtest",
                "-q",
                str(test_file),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=environment,
        )
        output = result.stdout + result.stderr

        assert result.returncode == 1
        metadata = celtest_plugin._project_metadata(Path(project_root))
        assert f"testing {metadata.display_name} {metadata.version}" in output
        assert "?" not in output
        assert ".WSFE" in output
        assert "\x1b[" not in output
        assert "\b" not in output
        assert "passed 2/4" in output
        assert "warnings 2" in output
        assert "friendly pass" not in output
        assert "F failures" in output
        assert "friendly failure: The sample assertion was false." in output
        assert "E errors" in output
        assert "friendly error: RuntimeError: sample runtime error" in output
        assert "W warnings" in output
        assert "UserWarning: sample warning" in output
        assert "UserWarning: failure warning" in output
        assert "Traceback" not in output
        assert "assert 'left' == 'right'" not in output

    def test_celtest_suppresses_foreign_unknown_status_letters(
        self, tmp_path: Path
    ) -> None:
        """Verify third-party unknown status hooks cannot leak invalid markers."""
        plugin_file = tmp_path / "foreign_status.py"
        plugin_file.write_text(
            """
def pytest_report_teststatus(report, config):
    del config
    if report.when == "call" and report.passed:
        return "unknown", "u", "UNKNOWN"
    return None
""",
            encoding="utf-8",
        )
        test_file = tmp_path / "test_sample.py"
        test_file.write_text(
            """
def test_first():
    pass


def test_second():
    pass
""",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join((project_root, str(tmp_path)))
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "tests.celtest",
                "-p",
                "foreign_status",
                test_file.name,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=environment,
        )
        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "\nu" not in output
        assert "[" not in output
        assert "passed 2/2" in output

    def test_celtest_reports_unstarted_tests_in_non_verbose_summary(
        self, tmp_path: Path
    ) -> None:
        """Verify compact output accounts for tests stopped before execution."""
        test_file = tmp_path / "test_fail_fast.py"
        test_file.write_text(
            """
from tests.celtest import celtest

@celtest("friendly failure")
def test_failure():
    assert False

@celtest("unstarted test")
def test_unstarted():
    pass
""",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = project_root
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.celtest",
                "-q",
                "-x",
                str(test_file),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=environment,
        )
        output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "passed 0/2" in output
        assert "N not run 1" in output

    def test_celtest_formats_collection_failure_without_traceback(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify collection failures use the fatal collection layout and hint."""
        test_file = tmp_path / "test_broken.py"
        test_file.write_text(
            "def test_broken(:\n    pass\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = project_root
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.celtest",
                "-q",
                str(test_file),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=environment,
        )
        output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "test collection failed" in output
        assert "E test_broken.py" in output
        assert "test collection failure hint" in output
        assert "SyntaxError" in output
        assert "Traceback" not in output
        assert "Interrupted" not in output

    def test_celtest_handles_interrupted_runs_without_incomplete_tests(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify KeyboardInterrupt does not turn incomplete tests into passes."""
        marker = tmp_path / "started"
        interrupt_request = tmp_path / "interrupt"
        test_file = tmp_path / "test_interrupt.py"
        test_file.write_text(
            """
import os
import time
from pathlib import Path

from tests.celtest import celtest

@celtest("completed test")
def test_completed():
    pass

@celtest("interrupted test")
def test_interrupted():
    Path(os.environ["CELTEST_MARKER"]).write_text("started", encoding="utf-8")
    interrupt_request = Path(os.environ["CELTEST_INTERRUPT"])
    while not interrupt_request.exists():
        time.sleep(0.05)
    raise KeyboardInterrupt

@celtest("unstarted test")
def test_unstarted():
    pass
""",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = project_root
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["CELTEST_MARKER"] = str(marker)
        environment["CELTEST_INTERRUPT"] = str(interrupt_request)
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.celtest",
                "-q",
                str(test_file),
            ],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=environment,
        ) as process:
            try:
                deadline = time.monotonic() + 15
                while not marker.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                if not marker.exists():
                    if process.poll() is None:
                        process.kill()
                    startup_output, _ = process.communicate(timeout=5)
                    raise AssertionError(
                        "the interrupted test did not start:\n" + startup_output
                    )
                interrupt_request.write_text("interrupt", encoding="utf-8")
                output, _ = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

        assert process.returncode != 0
        assert output.rstrip().endswith("interrupted")
        assert "test collection failed" not in output
        assert "passed " not in output
