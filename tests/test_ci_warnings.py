# SPDX-License-Identifier: Apache-2.0
"""Tests for GitHub Actions warning annotation helpers."""

import io
import sys
import contextlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "ci_warnings.py"
SPEC = importlib.util.spec_from_file_location("ci_warnings_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CI_WARNINGS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CI_WARNINGS
SPEC.loader.exec_module(CI_WARNINGS)


class TestCIWarnings:
    """Verify warning output is recognized and safely annotated."""

    def test_warning_lines_include_common_tool_formats(self) -> None:
        """Recognize prose, deprecation, and Pylint warning formats."""
        assert CI_WARNINGS._is_warning_line("WARN type inference degraded")
        assert CI_WARNINGS._is_warning_line("DeprecationWarning: old API")
        assert CI_WARNINGS._is_warning_line("celune/file.py:12: W0611")
        assert not CI_WARNINGS._is_warning_line("All checks passed")

    def test_annotation_messages_escape_github_command_delimiters(self) -> None:
        """Escape percent and line delimiters before emitting annotations."""
        assert CI_WARNINGS._annotation_message("50%\r\nwarning") == "50%25%0D%0Awarning"

    def test_main_preserves_failure_status_while_annotating_output(self) -> None:
        """Preserve wrapped command failures instead of masking them as warnings."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = CI_WARNINGS.main(
                [
                    "--",
                    sys.executable,
                    "-c",
                    "print('warning'); raise SystemExit(7)",
                ]
            )

        assert status == 7
        assert "::warning::warning" in output.getvalue()
