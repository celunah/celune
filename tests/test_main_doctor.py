# SPDX-License-Identifier: MIT
"""Tests for the lightweight `celune doctor` CLI path."""

import io
import contextlib
from pathlib import Path
from unittest import TestCase, mock

import main

entrypoint = main.load_entrypoint_module()


class DoctorCommandTests(TestCase):
    """Verify `celune doctor` works without booting the full app."""

    def test_main_routes_doctor_without_starting_app(self) -> None:
        """Verify the doctor branch exits through `run_doctor` instead of `start()`."""
        with (
            mock.patch.object(
                entrypoint, "main", side_effect=SystemExit(7)
            ) as entry_main,
            self.assertRaises(SystemExit) as exit_info,
        ):
            main.main(["celune", "doctor"])

        self.assertEqual(exit_info.exception.code, 7)
        entry_main.assert_called_once_with(["celune", "doctor"])

    def test_run_doctor_fix_invokes_repo_setup(self) -> None:
        """Verify `--fix` executes the repository-local setup.py with the current interpreter."""
        checks = [entrypoint.DoctorCheck("Python", True, "3.12.0")]

        with (
            mock.patch.object(entrypoint, "_doctor_checks", return_value=checks),
            mock.patch.object(entrypoint.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.return_value.returncode = 0
            exit_code = entrypoint.run_doctor(["celune", "doctor", "--fix"])

        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(
            [entrypoint.sys.executable, str(entrypoint.SETUP_PATH)],
            cwd=entrypoint.PROJECT_ROOT,
            check=False,
        )

    def test_run_doctor_rejects_unknown_args(self) -> None:
        """Verify unsupported doctor flags produce usage output and a CLI error code."""
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exit_code = entrypoint.run_doctor(["celune", "doctor", "--mystery"])

        self.assertEqual(exit_code, entrypoint.EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
        self.assertIn("Usage: celune doctor [--fix]", stdout.getvalue())

    def test_run_doctor_returns_failure_when_required_checks_fail(self) -> None:
        """Verify failed doctor checks propagate a failing exit status."""
        checks = [
            entrypoint.DoctorCheck("Python", True, "3.12.0"),
            entrypoint.DoctorCheck("uv", False, "not found", hint="Install uv."),
        ]

        with (
            mock.patch.object(entrypoint, "_doctor_checks", return_value=checks),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            exit_code = entrypoint.run_doctor(["celune", "doctor"])

        self.assertEqual(exit_code, entrypoint.EXIT_CODES.EXIT_FAILURE.value)
        output = stdout.getvalue()
        self.assertIn("[FAIL] uv: not found", output)
        self.assertIn("Summary:", output)

    def test_doctor_checks_warn_when_running_outside_project_venv(self) -> None:
        """Verify doctor prefers the project virtual environment over system Python."""
        with (
            mock.patch.object(entrypoint.platform, "system", return_value="Windows"),
            mock.patch.object(entrypoint.platform, "machine", return_value="AMD64"),
            mock.patch.object(
                entrypoint.platform, "python_version", return_value="3.13.11"
            ),
            mock.patch.object(
                entrypoint, "_display_version", return_value=("4.0.1", "")
            ),
            mock.patch.object(entrypoint, "_doctor_import", return_value=True),
            mock.patch.object(
                entrypoint, "_doctor_binary_path", return_value=Path("C:/bin/sox.exe")
            ),
            mock.patch.object(
                entrypoint,
                "_doctor_config_path",
                return_value=Path("C:/runtime/config.yaml"),
            ),
            mock.patch.object(entrypoint, "_doctor_torch_details", return_value=[]),
            mock.patch.object(entrypoint.shutil, "which", return_value="C:/bin/uv.exe"),
            mock.patch.object(
                entrypoint,
                "_doctor_running_python",
                return_value=Path("C:/Python313/python.exe"),
            ),
            mock.patch.object(
                entrypoint,
                "_doctor_venv_python",
                return_value=Path("C:/repo/.venv/Scripts/python.exe"),
            ),
            mock.patch.object(entrypoint.Path, "exists", return_value=True),
        ):
            checks = entrypoint._doctor_checks()

        python_env = next(
            check for check in checks if check.label == "Python environment"
        )
        self.assertFalse(python_env.ok)
        self.assertEqual(python_env.severity, "warning")
        self.assertIn("system interpreter", python_env.detail)
