# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's friendly pytest plugin."""

from __future__ import annotations

import os
import sys
import time
import signal
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

            def write(self, message: str, *, flush: bool = False) -> None:
                """Capture a terminal write."""
                del flush
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
            celtest_plugin._write_live_line("⚙️ working", replace=True, complete=False)
            celtest_plugin._write_live_line("✅ working", replace=True)
        finally:
            celtest_plugin._state = previous_state

        assert messages == ["\r⚙️ working", "\r✅ working \n"]

    def test_celtest_formats_pass_warning_failure_and_skip(
        self, tmp_path: Path
    ) -> None:
        """Verify the plugin emits the requested compact result contract."""
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
        assert "⚙️ friendly pass" in output
        assert "✅ friendly pass" in output
        assert "⚙️ friendly warning" in output
        assert "⚠️ friendly warning" in output
        assert "❌ friendly failure" in output
        assert "hidden skip" not in output
        assert "passed 2/3" in output
        assert "❌ friendly failure" in output
        assert "warnings 2" in output
        assert "UserWarning: sample warning" in output
        assert "UserWarning: failure warning" in output
        assert "The sample assertion was false." in output
        assert "Traceback" not in output
        assert "assert 'left' == 'right'" in output

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
        assert "❌ test_broken.py" in output
        assert "ℹ️ test collection failure hint" in output
        assert "SyntaxError" in output
        assert "Traceback" not in output
        assert "Interrupted" not in output

    def test_celtest_ends_interrupted_runs_without_incomplete_tests(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify CTRL+C does not turn incomplete tests into passes or collection errors."""
        marker = tmp_path / "started"
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
    time.sleep(30)

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
                assert marker.exists(), "the interrupted test did not start"
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

        assert process.returncode != 0
        assert output.rstrip().endswith("interrupted")
        assert "test collection failed" not in output
        assert "✅ interrupted test" not in output
        assert "✅ unstarted test" not in output
        assert "ℹ️ interruption hint" not in output
        assert "False" not in output
        assert "passed " not in output
        assert "not run " not in output
