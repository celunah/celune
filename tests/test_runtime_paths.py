# SPDX-License-Identifier: MIT
"""Tests for Celune runtime path handling."""

import tempfile
import sys
import os
from pathlib import Path
from typing import Optional, cast
from unittest import TestCase, mock

import yaml
from textual.widgets import RichLog

from celune.constants import APP_SLUG
from celune.paths import (
    configure_huggingface_cache_environment,
    configure_huggingface_runtime,
    ensure_config_path,
    huggingface_home_dir,
    huggingface_hub_cache_dir,
    persona_data_dir,
    project_root,
    running_compiled,
)
from celune.persona.memory import default_memory_dir
from celune.cevoice import bundled_voices_dir, default_bundle_path
from celune.ui.app import CeluneUI
from celune.utils import format_error


class RuntimePathTests(TestCase):
    """Verify runtime files are written into the user data directory."""

    @staticmethod
    def _compiled_root_layout(root_parts: tuple[str, ...]) -> tuple[Path, Path]:
        """Return a platform-native fake app root and compiled executable path."""
        if os.name == "nt":
            root = Path("C:/", *root_parts)
            executable = root / "celune.exe"
            return root, executable

        root = Path("/", *root_parts)
        executable = root / "celune"
        return root, executable

    @staticmethod
    def _compiled_bin_layout(root_parts: tuple[str, ...]) -> tuple[Path, Path]:
        """Return a platform-native fake repo root and compiled bin executable path."""
        if os.name == "nt":
            root = Path("C:/", *root_parts)
            executable = root / "bin" / "celune.exe"
            return root, executable

        root = Path("/", *root_parts)
        executable = root / "bin" / "celune"
        return root, executable

    def tearDown(self) -> None:
        """Reset singleton UI guards after each test."""
        CeluneUI._instance = None

    def test_default_memory_dir_uses_persona_runtime_directory(self) -> None:
        """Verify Persona memory defaults to the shared character app-data path."""
        expected = Path("C:/runtime-data/persona")

        with mock.patch(
            "celune.persona.memory.persona_data_dir", return_value=expected
        ):
            self.assertEqual(default_memory_dir(), expected)

    def test_persona_data_dir_uses_runtime_persona_directory(self) -> None:
        """Verify Persona character data lives below the shared app-data directory."""
        expected = Path("C:/runtime-data/persona")

        with mock.patch("celune.paths.user_data_dir", return_value="C:/runtime-data"):
            self.assertEqual(persona_data_dir(), expected)

    def test_huggingface_cache_dirs_live_in_runtime_data(self) -> None:
        """Verify Celune's default Hugging Face caches live under user data."""
        expected_root = Path("C:/runtime-data")

        with mock.patch("celune.paths.user_data_dir", return_value=str(expected_root)):
            self.assertEqual(
                huggingface_home_dir(),
                expected_root / "huggingface",
            )
            self.assertEqual(
                huggingface_hub_cache_dir(),
                expected_root / "huggingface" / "hub",
            )

    def test_huggingface_cache_environment_defaults_to_runtime_data(self) -> None:
        """Verify Celune points Hugging Face caches at the runtime data directory."""
        expected_root = Path("C:/runtime-data")

        with (
            mock.patch("celune.paths.user_data_dir", return_value=str(expected_root)),
            mock.patch("celune.paths.running_compiled", return_value=True),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_cache_environment()

            self.assertEqual(
                os.environ["HF_HOME"],
                str(expected_root / "huggingface"),
            )
            self.assertEqual(
                os.environ["HF_HUB_CACHE"],
                str(expected_root / "huggingface" / "hub"),
            )

    def test_huggingface_cache_environment_respects_existing_overrides(self) -> None:
        """Verify explicit cache env vars are preserved."""
        existing = {
            "HF_HOME": "X:/hf-home",
            "HF_HUB_CACHE": "X:/hf-hub",
        }

        with (
            mock.patch("celune.paths.running_compiled", return_value=True),
            mock.patch.dict(os.environ, existing.copy(), clear=True),
        ):
            configure_huggingface_cache_environment()
            self.assertEqual(os.environ["HF_HOME"], existing["HF_HOME"])
            self.assertEqual(os.environ["HF_HUB_CACHE"], existing["HF_HUB_CACHE"])

    def test_huggingface_cache_environment_skips_source_tree_imports(self) -> None:
        """Verify source-tree runs keep the host Hugging Face cache defaults."""
        with (
            mock.patch("celune.paths.running_compiled", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_cache_environment()
            self.assertNotIn("HF_HOME", os.environ)
            self.assertNotIn("HF_HUB_CACHE", os.environ)

    def test_huggingface_cache_environment_clears_celune_portable_defaults(
        self,
    ) -> None:
        """Verify source-tree runs clear Celune-owned portable cache defaults."""
        expected_root = Path("C:/runtime-data")

        with (
            mock.patch("celune.paths.user_data_dir", return_value=str(expected_root)),
            mock.patch("celune.paths.running_compiled", return_value=False),
            mock.patch.dict(
                os.environ,
                {
                    "HF_HOME": str(expected_root / "huggingface"),
                    "HF_HUB_CACHE": str(expected_root / "huggingface" / "hub"),
                },
                clear=True,
            ),
        ):
            configure_huggingface_cache_environment()
            self.assertNotIn("HF_HOME", os.environ)
            self.assertNotIn("HF_HUB_CACHE", os.environ)

    def test_huggingface_cache_environment_keeps_non_celune_overrides(
        self,
    ) -> None:
        """Verify source-tree runs preserve unrelated explicit Hugging Face overrides."""
        existing = {
            "HF_HOME": "X:/hf-home",
            "HF_HUB_CACHE": "X:/hf-hub",
        }

        with (
            mock.patch("celune.paths.running_compiled", return_value=False),
            mock.patch.dict(os.environ, existing.copy(), clear=True),
        ):
            configure_huggingface_cache_environment()
            self.assertEqual(os.environ["HF_HOME"], existing["HF_HOME"])
            self.assertEqual(os.environ["HF_HUB_CACHE"], existing["HF_HUB_CACHE"])

    def test_huggingface_runtime_disables_global_progress_bars(self) -> None:
        """Verify Celune suppresses Hugging Face progress bars without muting logs."""
        with (
            mock.patch("celune.paths.disable_progress_bar") as disable_transformers,
            mock.patch("celune.paths.disable_progress_bars") as disable_hub,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_runtime()
            self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")

        disable_transformers.assert_called_once_with()
        disable_hub.assert_called_once_with()

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

    def test_format_error_keeps_traceback_after_exception_handler_returns(self) -> None:
        """Verify deferred UI error formatting does not report a blank traceback.

        Raises:
            RuntimeError: If `RuntimeError` needs to be raised.
        """
        captured: Optional[RuntimeError] = None
        try:
            raise RuntimeError("deferred boom")
        except RuntimeError as exc:
            captured = exc

        if captured is None:
            self.fail("The test exception was not captured")

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / f"{APP_SLUG}_traceback.txt"
            with mock.patch("celune.utils.traceback_path", return_value=trace_path):
                output = format_error(captured, dev=True)

        self.assertIn("RuntimeError: deferred boom", output)
        self.assertNotIn("NoneType: None", output)

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
        expected_root, executable = self._compiled_root_layout(("Apps", "Celune"))

        with (
            mock.patch.dict(sys.modules, {"__main__": fake_main}),
            mock.patch.object(sys, "argv", [str(executable)]),
        ):
            self.assertEqual(project_root(), expected_root)
            self.assertEqual(
                default_bundle_path(),
                expected_root / "voices" / "default.cevoice",
            )
            self.assertEqual(bundled_voices_dir(), expected_root / "voices")

    def test_compiled_project_root_uses_repo_parent_when_running_from_bin(self) -> None:
        """Verify compiled launches from bin/ still resolve the repository root."""
        fake_main = type("CompiledMain", (), {"__compiled__": True})()
        expected_root, executable = self._compiled_bin_layout(("repo",))

        def fake_exists(path: Path) -> bool:
            normalized = str(path).replace("\\", "/")
            return normalized in {
                str(expected_root / "celune").replace("\\", "/"),
                str(expected_root / "default_config.yaml").replace("\\", "/"),
                str(expected_root / "pyproject.toml").replace("\\", "/"),
            }

        with (
            mock.patch.dict(sys.modules, {"__main__": fake_main}),
            mock.patch.object(sys, "argv", [str(executable)]),
            mock.patch.object(Path, "exists", fake_exists),
        ):
            self.assertEqual(project_root(), expected_root)
            self.assertEqual(
                default_bundle_path(),
                expected_root / "voices" / "default.cevoice",
            )
