# SPDX-License-Identifier: MIT
"""Tests for Celune runtime path handling."""

import os
import sys
import tempfile
from typing import cast
from pathlib import Path
from unittest import TestCase, mock

import yaml
from textual.widgets import RichLog

from celune.ui.app import CeluneUI
from celune.utils import format_error
from celune.constants import APP_SLUG
from celune.persona.memory import default_memory_dir
from celune.cevoice import bundled_voices_dir, default_bundle_path
from celune.paths import ensure_config_path, project_root, running_compiled


class RuntimePathTests(TestCase):
    """Verify runtime files are written into the user data directory."""

    def tearDown(self) -> None:
        """Reset singleton UI guards after each test."""
        CeluneUI._instance = None

    def test_default_memory_dir_uses_runtime_memory_directory(self) -> None:
        """Verify Persona memory now defaults to the shared runtime memory path."""
        expected = Path("C:/runtime-data/memory")

        with mock.patch("celune.persona.memory.memory_data_dir", return_value=expected):
            self.assertEqual(default_memory_dir(), expected)

    def test_format_error_writes_traceback_to_runtime_directory(self) -> None:
        """Verify developer tracebacks are saved via the runtime path helper.

        Raises:
            RuntimeError: An exception was raised for testing purposes and caught afterward.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / f"{APP_SLUG}_traceback.txt"

            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                with mock.patch("celune.utils.traceback_path", return_value=trace_path):
                    output = format_error(exc, dev=True)

                self.assertIn("RuntimeError: boom", output)
                self.assertTrue(trace_path.exists())
                self.assertIn(
                    "RuntimeError: boom", trace_path.read_text(encoding="utf-8")
                )

    def test_safe_log_persists_main_window_copy(self) -> None:
        """Verify UI log messages are mirrored into the runtime log file."""
        ui = CeluneUI()
        ui.logs = cast(RichLog, None)
        ui._log_file_initialized = False

        with tempfile.TemporaryDirectory() as temp_dir:
            ui._log_file_path = Path(temp_dir) / f"{APP_SLUG}.log"

            ui.safe_log("Hello from Celune")
            ui.safe_log("Something odd happened", "warning")

            persisted = ui._log_file_path.read_text(encoding="utf-8")

        self.assertIn("[INFO] Hello from Celune", persisted)
        self.assertIn("[WARNING] Something odd happened", persisted)

    def test_ensure_config_path_prefers_legacy_repo_config(self) -> None:
        """Verify first-run config creation prefers the historical repo-root config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            runtime_config = temp_root / "user-data" / "config.yaml"
            legacy_config = temp_root / "config.yaml"
            bundled_default = temp_root / "default_config.yaml"

            legacy_payload = {"headless": True, "theme": "light", "dev": False}
            bundled_payload = {"headless": False, "theme": "dark", "dev": False}
            legacy_config.write_text(yaml.safe_dump(legacy_payload), encoding="utf-8")
            bundled_default.write_text(
                yaml.safe_dump(bundled_payload),
                encoding="utf-8",
            )

            created_path, was_created = ensure_config_path(
                active_path=runtime_config,
                default_path=bundled_default,
                legacy_path=legacy_config,
            )

            saved = yaml.safe_load(created_path.read_text(encoding="utf-8"))
            self.assertTrue(was_created)
            self.assertEqual(saved["theme"], "light")
            self.assertEqual(saved["headless"], True)

    def test_running_compiled_detects_compiled_main_module(self) -> None:
        """Verify compiled-mode detection checks the active main module."""
        main_module = sys.modules["__main__"]
        original = getattr(main_module, "__compiled__", None)
        had_attr = hasattr(main_module, "__compiled__")

        # the type errors are suppressed because they are Nuitka specific
        try:
            main_module.__compiled__ = True  # type: ignore[missing-attribute]
            self.assertTrue(running_compiled())
        finally:
            if had_attr:
                main_module.__compiled__ = original  # type: ignore[missing-attribute]
            else:
                delattr(main_module, "__compiled__")

    def test_compiled_project_root_and_bundled_paths_follow_executable(self) -> None:
        """Verify bundled files resolve beside the compiled executable."""
        fake_main = type("CompiledMain", (), {"__compiled__": True})()

        if os.name == "nt":
            exe_path = "C:/Apps/Celune/celune.exe"
            expected_root = Path("C:/Apps/Celune")
        else:
            exe_path = "/opt/celune/celune"
            expected_root = Path("/opt/celune")

        with (
            mock.patch.dict(sys.modules, {"__main__": fake_main}),
            mock.patch.object(sys, "argv", [exe_path]),
        ):
            self.assertEqual(project_root(), Path(expected_root))
            self.assertEqual(
                default_bundle_path(),
                expected_root / "voices" / "default.cevoice",
            )
            self.assertEqual(bundled_voices_dir(), expected_root / "voices")
