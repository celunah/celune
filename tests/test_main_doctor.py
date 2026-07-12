# SPDX-License-Identifier: MIT
"""Tests for the lightweight `celune doctor` CLI path."""

import io
import contextlib
from types import SimpleNamespace
from unittest import TestCase, mock
from pathlib import Path, PureWindowsPath

import main

entrypoint = main.load_entrypoint_module()


class DoctorCommandTests(TestCase):
    """Verify `celune doctor` works without booting the full app."""

    def test_ui_test_backend_is_loaded_lazily(self) -> None:
        """Verify normal entrypoint imports do not require the test suite package."""
        fake_support = mock.Mock(FakeBackend=SimpleNamespace())

        with mock.patch.object(
            entrypoint.importlib,
            "import_module",
            return_value=fake_support,
        ) as import_module:
            backend = entrypoint._load_ui_test_backend()

        self.assertIs(backend, fake_support.FakeBackend)
        import_module.assert_called_once_with("tests.support")

    def test_main_reports_unsupported_python_before_loading_entrypoint(self) -> None:
        """Verify doctor on Python 3.11 exits cleanly before importing 3.12-only modules."""
        with (
            mock.patch.object(main.sys, "version_info", (3, 11, 9)),
            mock.patch.object(main, "_load_entrypoint_module") as load_entrypoint,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            self.assertRaises(SystemExit) as exit_info,
        ):
            main.main(["celune", "doctor"])

        self.assertEqual(exit_info.exception.code, 6)
        load_entrypoint.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("will not run on Python 3.11.9", output)
        self.assertIn("use at least Python 3.12", output)
        self.assertIn("doctor", output)

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
            [str(entrypoint.doctor_running_python()), str(entrypoint.SETUP_PATH)],
            cwd=entrypoint.PROJECT_ROOT,
            check=False,
        )

    def test_run_doctor_fix_uses_repo_venv_python_when_compiled(self) -> None:
        """Verify compiled doctor fixups use the repo virtualenv Python."""
        checks = [entrypoint.DoctorCheck("Python", True, "3.12.0")]

        with (
            mock.patch.object(entrypoint, "_doctor_checks", return_value=checks),
            mock.patch.object(entrypoint, "running_compiled", return_value=True),
            mock.patch.object(
                entrypoint,
                "_doctor_venv_python",
                return_value=Path("C:/repo/.venv/Scripts/python.exe"),
            ),
            mock.patch.object(entrypoint.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.return_value.returncode = 0
            exit_code = entrypoint.run_doctor(["celune", "doctor", "--fix"])

        self.assertEqual(exit_code, 0)

        run.assert_called_once()
        args, kwargs = run.call_args
        command = args[0]

        self.assertEqual(
            PureWindowsPath(command[0]),
            PureWindowsPath(r"C:\repo\.venv\Scripts\python.exe"),
        )
        self.assertEqual(command[1], str(entrypoint.SETUP_PATH))
        self.assertEqual(kwargs["cwd"], entrypoint.PROJECT_ROOT)
        self.assertFalse(kwargs["check"])

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

    def test_run_doctor_reports_warning_only_state(self) -> None:
        """Verify warning-only doctor results do not masquerade as a clean pass."""
        checks = [
            entrypoint.DoctorCheck(
                "Accelerator backend",
                False,
                "Detected non-NVIDIA CUDA compatibility mode.",
                severity="warning",
            )
        ]

        with (
            mock.patch.object(entrypoint, "_doctor_checks", return_value=checks),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            exit_code = entrypoint.run_doctor(["celune", "doctor"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("[WARN] Accelerator backend", output)
        self.assertIn("performance may be impacted", output)

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
            checks = entrypoint.doctor_checks()

        python_env = next(
            check for check in checks if check.label == "Python environment"
        )
        self.assertFalse(python_env.ok)
        self.assertEqual(python_env.severity, "warning")
        self.assertIn("system interpreter", python_env.detail)

    def test_doctor_torch_details_detects_zluda_and_runs_compute_test(self) -> None:
        """Verify doctor mirrors the app's ZLUDA warning and CUDA compute smoke test."""
        fake_torch = mock.Mock()
        fake_torch.__version__ = "2.7.0+cu128"
        fake_torch.version = mock.Mock(cuda="12.8", hip=None)
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 1
        fake_torch.cuda.get_device_name.return_value = "AMD Radeon RX 7800 XT"
        fake_torch.cuda.get_device_capability.return_value = (8, 9)
        fake_torch.backends = mock.Mock()
        fake_torch.backends.mps.is_available.return_value = False

        with (
            mock.patch.object(entrypoint, "_doctor_import", return_value=True),
            mock.patch.object(
                entrypoint.importlib, "import_module", return_value=fake_torch
            ),
            mock.patch.object(
                entrypoint, "_doctor_run_compute_test", return_value="cuda:0"
            ),
        ):
            checks = entrypoint.doctor_torch_details()

        by_label = {check.label: check for check in checks}
        self.assertEqual(by_label["Accelerator backend"].severity, "warning")
        self.assertIn("ZLUDA", by_label["Accelerator backend"].detail)
        self.assertTrue(by_label["CUDA compute test"].ok)
        self.assertEqual(by_label["CUDA compute test"].detail, "Succeeded on cuda:0")
