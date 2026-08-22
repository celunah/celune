# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune runtime path handling."""

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, cast
from unittest import mock

import pytest
import yaml
from textual.widgets import RichLog
from celune.constants import APP_SLUG
from celune.utils import discard, format_error
from celune.ui.app import CeluneUI, UILogMessage
from celune.persona.memory import default_memory_dir
from celune.cevoice import bundled_voices_dir, default_bundle_path
from celune.paths import (
    project_root,
    voices_data_dir,
    persona_data_dir,
    running_compiled,
    runtime_data_dir,
    ensure_config_path,
    huggingface_home_dir,
    migrate_legacy_app_data,
    backend_environments_dir,
    huggingface_hub_cache_dir,
    huggingface_progress,
    configure_huggingface_runtime,
    configure_huggingface_cache_environment,
)

from .support import CeluneTestCase


class TestRuntimePath(CeluneTestCase):
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
            assert default_memory_dir() == expected

    def test_persona_data_dir_uses_runtime_persona_directory(self) -> None:
        """Verify Persona character data lives below the shared app-data directory."""
        expected = Path("C:/runtime-data/persona")

        with mock.patch("celune.paths.user_data_dir", return_value="C:/runtime-data"):
            assert persona_data_dir() == expected

    def test_voices_data_dir_uses_runtime_voice_pack_directory(self) -> None:
        """Verify voice packs use a sibling directory in Celune's app data."""
        expected = Path("C:/runtime-data/voices")

        with mock.patch("celune.paths.user_data_dir", return_value="C:/runtime-data"):
            assert voices_data_dir() == expected

    def test_runtime_and_environment_dirs_use_organized_app_data_paths(self) -> None:
        """Verify loose runtime data and backend environments use dedicated folders."""
        expected_root = Path("C:/runtime-data")

        with mock.patch("celune.paths.user_data_dir", return_value="C:/runtime-data"):
            assert runtime_data_dir() == expected_root / "runtime"
            assert backend_environments_dir() == expected_root / "environments"

    def test_migrate_legacy_app_data_moves_existing_directories(self) -> None:
        """Verify legacy loose data is moved without overwriting new data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_directories = {
                "backends": ("mini", "environment-marker.txt"),
                "fast_langdetect": ("model-marker.txt",),
                "gpt_sovits": ("source-marker.txt",),
                "nltk_data": ("resource-marker.txt",),
            }
            for name, parts in legacy_directories.items():
                path = root / name / Path(*parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")

            with mock.patch("celune.paths.user_data_dir", return_value=temp_dir):
                migrate_legacy_app_data()
                migrate_legacy_app_data()

            assert (
                root / "environments" / "mini" / "environment-marker.txt"
            ).read_text(encoding="utf-8") == "backends"
            for name in ("fast_langdetect", "gpt_sovits", "nltk_data"):
                assert (root / "runtime" / name).is_dir()
                assert not (root / name).exists()

    def test_huggingface_cache_dirs_live_in_runtime_data(self) -> None:
        """Verify Celune's default Hugging Face caches live under user data."""
        expected_root = Path("C:/runtime-data")

        with mock.patch("celune.paths.user_data_dir", return_value=str(expected_root)):
            assert huggingface_home_dir() == expected_root / "huggingface"
            assert huggingface_hub_cache_dir() == expected_root / "huggingface" / "hub"

    def test_huggingface_cache_environment_defaults_to_runtime_data(self) -> None:
        """Verify Celune points Hugging Face caches at the runtime data directory."""
        expected_root = Path("C:/runtime-data")

        with (
            mock.patch("celune.paths.user_data_dir", return_value=str(expected_root)),
            mock.patch("celune.paths.running_compiled", return_value=True),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_cache_environment()

            assert os.environ["HF_HOME"] == str(expected_root / "huggingface")
            assert os.environ["HF_HUB_CACHE"] == str(
                expected_root / "huggingface" / "hub"
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
            assert os.environ["HF_HOME"] == existing["HF_HOME"]
            assert os.environ["HF_HUB_CACHE"] == existing["HF_HUB_CACHE"]

    def test_huggingface_cache_environment_skips_source_tree_imports(self) -> None:
        """Verify source-tree runs keep the host Hugging Face cache defaults."""
        with (
            mock.patch("celune.paths.running_compiled", return_value=False),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_cache_environment()
            assert "HF_HOME" not in os.environ
            assert "HF_HUB_CACHE" not in os.environ

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
            assert "HF_HOME" not in os.environ
            assert "HF_HUB_CACHE" not in os.environ

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
            assert os.environ["HF_HOME"] == existing["HF_HOME"]
            assert os.environ["HF_HUB_CACHE"] == existing["HF_HUB_CACHE"]

    def test_huggingface_runtime_disables_global_progress_bars(self) -> None:
        """Verify Celune suppresses Hugging Face progress bars without muting logs."""
        with (
            mock.patch(
                "transformers.utils.logging.disable_progress_bar"
            ) as disable_transformers,
            mock.patch("huggingface_hub.utils.disable_progress_bars") as disable_hub,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            configure_huggingface_runtime()
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

        disable_transformers.assert_called_once_with()
        disable_hub.assert_called_once_with()

    def test_huggingface_progress_forwards_disabled_transfer_updates(self) -> None:
        """Verify quiet Hugging Face bars still update Celune's progress callback."""
        from huggingface_hub import _snapshot_download, file_download

        callback = mock.Mock()
        previous_tqdm = file_download.tqdm
        previous_hf_tqdm = _snapshot_download.hf_tqdm
        with huggingface_progress(callback):
            progress_bar = file_download.tqdm(
                total=100,
                initial=0,
                desc="model.safetensors",
                disable=True,
            )
            progress_bar.update(25)
            progress_bar.close()
            aggregate_bar = _snapshot_download.hf_tqdm(
                total=200,
                initial=0,
                desc="aggregate",
                disable=True,
            )
            aggregate_bar.update(50)
            aggregate_bar.close()

        assert file_download.tqdm is previous_tqdm
        assert _snapshot_download.hf_tqdm is previous_hf_tqdm
        callback.assert_any_call(25.0, 100.0)
        callback.assert_any_call(50.0, 200.0)

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
                    output = format_error(exc, log_level="debug")

                assert "RuntimeError: boom" in output
                assert trace_path.exists()
                assert "RuntimeError: boom" in trace_path.read_text(encoding="utf-8")

    def test_format_error_keeps_traceback_after_exception_handler_returns(self) -> None:
        """Verify deferred UI error formatting does not report a blank traceback.

        Raises:
            RuntimeError: Raised by the test to verify deferred traceback formatting.
        """
        captured: Optional[RuntimeError] = None
        discard(captured)
        try:
            raise RuntimeError("deferred boom")
        except RuntimeError as exc:
            captured = exc

        if captured is None:
            pytest.fail("The test exception was not captured")

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / f"{APP_SLUG}_traceback.txt"
            with mock.patch("celune.utils.traceback_path", return_value=trace_path):
                output = format_error(captured, log_level="debug")

        assert "RuntimeError: deferred boom" in output
        assert "NoneType: None" not in output

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

        assert "[INFO] Hello from Celune" in persisted
        assert "[WARNING] Something odd happened" in persisted

    def test_safe_log_from_worker_posts_without_waiting_for_ui_thread(self) -> None:
        """Verify background logging queues a message without a synchronous UI callback."""
        ui = CeluneUI()
        ui.logs = cast(RichLog, mock.Mock())
        ui._persist_log_entry = mock.Mock()
        ui.post_message = mock.Mock()
        ui.call_from_thread = mock.Mock()

        worker = threading.Thread(target=ui.safe_log, args=("worker log",))
        worker.start()
        worker.join(timeout=1.0)

        assert not worker.is_alive()
        ui.post_message.assert_called_once()
        ui.call_from_thread.assert_not_called()
        message = ui.post_message.call_args.args[0]
        assert isinstance(message, UILogMessage)
        assert message.message == "worker log"
        assert UILogMessage.handler_name == "on_uilog_message"

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
            assert was_created
            assert saved["theme"] == "light"
            assert saved["headless"]

    def test_running_compiled_detects_compiled_main_module(self) -> None:
        """Verify compiled-mode detection checks the active main module."""
        main_module = sys.modules["__main__"]
        original = getattr(main_module, "__compiled__", None)
        had_attr = hasattr(main_module, "__compiled__")

        # the type errors are suppressed because they are Nuitka specific
        try:
            main_module.__compiled__ = True  # type: ignore[missing-attribute]
            assert running_compiled()
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
            mock.patch(
                "celune.cevoice.voices_data_dir",
                return_value=Path("C:/runtime-data/voices"),
            ),
        ):
            assert project_root() == expected_root
            assert default_bundle_path() == Path(
                "C:/runtime-data/voices/default.cevoice"
            )
            assert bundled_voices_dir() == Path("C:/runtime-data/voices")

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
            mock.patch(
                "celune.cevoice.voices_data_dir",
                return_value=Path("C:/runtime-data/voices"),
            ),
        ):
            assert project_root() == expected_root
            assert default_bundle_path() == Path(
                "C:/runtime-data/voices/default.cevoice"
            )
