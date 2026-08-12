# SPDX-License-Identifier: MIT
"""Tests for the lightweight `celune doctor` CLI path."""

import contextlib
import io
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest import mock

import pytest

import main

from .support import CeluneTestCase

entrypoint = main.load_entrypoint_module()


class TestDoctorCommand(CeluneTestCase):
    """Verify `celune doctor` works without booting the full app."""

    @staticmethod
    def test_close_existing_processes_never_kills_current_process() -> None:
        """Verify launcher cleanup excludes itself and waits for the old process."""
        current = SimpleNamespace(pid=101, name=lambda: "python.exe")
        launcher = SimpleNamespace(pid=303, name=lambda: "celune.exe")
        existing = SimpleNamespace(
            pid=202,
            name=lambda: "CELUNE-BIN.EXE",
            kill=mock.Mock(),
            wait=mock.Mock(),
        )
        psutil = SimpleNamespace(
            process_iter=lambda: [existing, launcher, current],
            AccessDenied=PermissionError,
            NoSuchProcess=ProcessLookupError,
            ZombieProcess=RuntimeError,
            TimeoutExpired=TimeoutError,
        )
        runtime = SimpleNamespace(
            psutil=psutil,
            SelectMenu=mock.Mock(
                return_value=SimpleNamespace(start=mock.Mock(return_value=True))
            ),
        )

        with (
            mock.patch.object(entrypoint.os, "getpid", return_value=101),
            mock.patch.object(entrypoint.os, "getppid", return_value=303),
            mock.patch.dict(entrypoint.os.environ, {"CELUNE_LAUNCHER_PID": ""}),
        ):
            entrypoint._close_existing_celune_processes(runtime)

        existing.kill.assert_called_once_with()
        existing.wait.assert_called_once_with(timeout=5)
        runtime.SelectMenu.assert_called_once()

    def test_ui_test_backend_is_loaded_lazily(self) -> None:
        """Verify normal entrypoint imports do not require the test suite package."""
        fake_support = mock.Mock(FakeBackend=SimpleNamespace())

        with mock.patch.object(
            entrypoint.importlib,
            "import_module",
            return_value=fake_support,
        ) as import_module:
            backend = entrypoint._load_ui_test_backend()

        assert backend is fake_support.FakeBackend
        import_module.assert_called_once_with("tests.support")

    def test_main_reports_unsupported_python_before_loading_entrypoint(self) -> None:
        """Verify doctor on Python 3.11 exits cleanly before importing 3.12-only modules."""
        with (
            mock.patch.object(main.sys, "version_info", (3, 11, 9)),
            mock.patch.object(main, "_load_entrypoint_module") as load_entrypoint,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            pytest.raises(SystemExit) as exit_info,
        ):
            main.main(["celune", "doctor"])

        assert exit_info.value.code == 6
        load_entrypoint.assert_not_called()
        output = stdout.getvalue()
        assert "will not run on Python 3.11.9" in output
        assert "use at least Python 3.12" in output
        assert "doctor" in output

    def test_main_routes_doctor_without_starting_app(self) -> None:
        """Verify the doctor branch exits through `run_doctor` instead of `start()`."""
        with (
            mock.patch.object(
                entrypoint, "main", side_effect=SystemExit(7)
            ) as entry_main,
            pytest.raises(SystemExit) as exit_info,
        ):
            main.main(["celune", "doctor"])

        assert exit_info.value.code == 7
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

        assert exit_code == 0
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

        assert exit_code == 0

        run.assert_called_once()
        args, kwargs = run.call_args
        command = args[0]

        assert PureWindowsPath(command[0]) == PureWindowsPath(
            r"C:\repo\.venv\Scripts\python.exe"
        )
        assert command[1] == str(entrypoint.SETUP_PATH)
        assert kwargs["cwd"] == entrypoint.PROJECT_ROOT
        assert not kwargs["check"]

    def test_run_doctor_rejects_unknown_args(self) -> None:
        """Verify unsupported doctor flags produce usage output and a CLI error code."""
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exit_code = entrypoint.run_doctor(["celune", "doctor", "--mystery"])

        assert exit_code == entrypoint.EXIT_CODES.EXIT_UNKNOWN_ARGS.value
        assert "Usage: celune doctor [--fix]" in stdout.getvalue()

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

        assert exit_code == entrypoint.EXIT_CODES.EXIT_FAILURE.value
        output = stdout.getvalue()
        assert "[FAIL] uv: not found" in output
        assert "Summary:" in output

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

        assert exit_code == 0
        output = stdout.getvalue()
        assert "[WARN] Accelerator backend" in output
        assert "performance may be impacted" in output

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
        assert not python_env.ok
        assert python_env.severity == "warning"
        assert "system interpreter" in python_env.detail

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
        assert by_label["Accelerator backend"].severity == "warning"
        assert "ZLUDA" in by_label["Accelerator backend"].detail
        assert by_label["CUDA compute test"].ok
        assert by_label["CUDA compute test"].detail == "Succeeded on cuda:0"
