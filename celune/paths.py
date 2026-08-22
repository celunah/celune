# SPDX-License-Identifier: Apache-2.0
"""Runtime filesystem paths and global Hugging Face runtime setup for Celune."""

import contextlib
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import threading
import time
from collections.abc import Callable, Generator
from typing import Optional

from platformdirs import user_data_dir

from .constants import APP_NAME, APP_SLUG

_REPO_MARKERS = ("celune", "default_config.yaml", "pyproject.toml")
_HF_HOME_ENV = "HF_HOME"
_HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
_HF_HUB_DISABLE_PROGRESS_BARS_ENV = "HF_HUB_DISABLE_PROGRESS_BARS"
_HF_PROGRESS_PATCH_LOCK = threading.RLock()
_LEGACY_APP_DATA_MIGRATIONS = (
    ("backends", ("environments",)),
    ("fast_langdetect", ("runtime", "fast_langdetect")),
    ("gpt_sovits", ("runtime", "gpt_sovits")),
    ("nltk_data", ("runtime", "nltk_data")),
)


def running_compiled() -> bool:
    """Return whether Celune is running from a compiled entrypoint.

    Returns:
        bool: ``True`` when the active ``__main__`` module was marked as compiled.
    """
    main_module = sys.modules.get("__main__")
    return bool(getattr(main_module, "__compiled__", False))


def _looks_like_repo_root(path: Path) -> bool:
    """Return whether a path looks like the Celune repository root."""
    return all((path / marker).exists() for marker in _REPO_MARKERS)


def _compiled_project_root() -> Path:
    """Resolve the repository root for compiled launches started from bin/."""
    executable_dir = Path(sys.argv[0]).resolve().parent

    for candidate in (executable_dir, executable_dir.parent):
        if _looks_like_repo_root(candidate):
            return candidate

    return executable_dir


def app_data_dir(create: bool = False) -> Path:
    """Return Celune's user data directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's user data directory.
    """
    path = Path(user_data_dir(APP_NAME, appauthor=False))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def persona_data_dir(create: bool = False) -> Path:
    """Return Celune's Persona character data directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: The Persona directory inside Celune's user data directory.
    """
    path = app_data_dir(create=create) / "persona"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_data_dir(create: bool = False) -> Path:
    """Return Celune's directory for runtime-owned loose data.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's runtime data directory.
    """
    path = app_data_dir(create=create) / "runtime"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def huggingface_home_dir(create: bool = False) -> Path:
    """Return Celune's default Hugging Face home directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's Hugging Face home directory inside the user data folder.
    """
    path = app_data_dir(create=create) / "huggingface"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def huggingface_hub_cache_dir(create: bool = False) -> Path:
    """Return Celune's default Hugging Face Hub snapshot cache directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's Hugging Face Hub cache directory.
    """
    path = huggingface_home_dir(create=create) / "hub"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_huggingface_cache_environment(force: bool = False) -> None:
    """Point Hugging Face caches at Celune's user data directory.

    Args:
        force: Retained for compatibility; defaults apply in every launch mode.
    """
    default_hf_home = str(huggingface_home_dir())
    default_hf_hub_cache = str(huggingface_hub_cache_dir())

    # Keep explicit deployment or host overrides, but make Celune's cache
    # deterministic for source-tree, compiled, and isolated worker processes.
    if _HF_HOME_ENV not in os.environ:
        os.environ[_HF_HOME_ENV] = default_hf_home
    if _HF_HUB_CACHE_ENV not in os.environ:
        os.environ[_HF_HUB_CACHE_ENV] = default_hf_hub_cache


def configure_huggingface_runtime() -> None:
    """Apply Celune's process-wide Hugging Face progress suppression."""
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils.logging import disable_progress_bar

    os.environ.setdefault(_HF_HUB_DISABLE_PROGRESS_BARS_ENV, "1")
    disable_progress_bar()
    disable_progress_bars()


def configure_numba_cache() -> Path:
    """Configure a writable process cache for Numba-backed dependencies.

    Returns:
        Path: The writable Numba cache directory.
    """
    cache_dir = temp_data_dir(create=True) / "numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    return cache_dir


@contextlib.contextmanager
def huggingface_progress(
    callback: Optional[Callable[[Optional[float], Optional[float]], None]],
) -> Generator[None, None, None]:
    """Forward Hugging Face transfer progress to one Celune progress callback.

    Args:
        callback: Receiver for downloaded bytes and their total, or ``None`` to leave the Hub unchanged.
    """
    if callback is None:
        yield
        return

    import importlib

    from huggingface_hub.utils.tqdm import tqdm as HuggingFaceTqdm

    file_download = importlib.import_module("huggingface_hub.file_download")
    snapshot_download = importlib.import_module("huggingface_hub._snapshot_download")

    class CeluneHuggingFaceTqdm(HuggingFaceTqdm):
        """Keep Hugging Face bars quiet while forwarding their transfer state."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            initial = kwargs.get("initial", 0)
            self._celune_downloaded = (
                float(initial) if isinstance(initial, (int, float)) else 0.0
            )
            self._celune_progress_callback = callback
            self._celune_progress_lock = threading.Lock()
            self._celune_last_report = 0.0
            super().__init__(*args, **kwargs)
            self._report_celune_progress(force=True)

        def update(self, n: Optional[float] = 1) -> Optional[bool]:
            if isinstance(n, (int, float)) and not isinstance(n, bool):
                self._celune_downloaded = max(
                    0.0,
                    self._celune_downloaded + float(n),
                )
            result = super().update(n)
            self._report_celune_progress()
            return result

        def close(self) -> None:
            self._report_celune_progress(force=True)
            super().close()

        def _report_celune_progress(self, force: bool = False) -> None:
            total = getattr(self, "total", None)
            normalized_total = (
                float(total)
                if isinstance(total, (int, float))
                and not isinstance(total, bool)
                and total > 0
                else None
            )
            progress = self._celune_downloaded if normalized_total is not None else None
            now = time.monotonic()
            with self._celune_progress_lock:
                if not force and now - self._celune_last_report < 0.1:
                    return
                self._celune_last_report = now
            with contextlib.suppress(Exception):
                self._celune_progress_callback(progress, normalized_total)

    with _HF_PROGRESS_PATCH_LOCK:
        previous_file_tqdm = file_download.__dict__["tqdm"]
        previous_snapshot_tqdm = snapshot_download.__dict__["hf_tqdm"]
        file_download.__dict__["tqdm"] = CeluneHuggingFaceTqdm
        snapshot_download.__dict__["hf_tqdm"] = CeluneHuggingFaceTqdm
        try:
            yield
        finally:
            file_download.__dict__["tqdm"] = previous_file_tqdm
            snapshot_download.__dict__["hf_tqdm"] = previous_snapshot_tqdm


def memory_data_dir(create: bool = False) -> Path:
    """Return the persistent memory directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's persistent memory directory.
    """
    path = app_data_dir(create=create) / "memory"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def temp_data_dir(create: bool = False) -> Path:
    """Return the Celune-scoped temporary data directory.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's temporary data directory.
    """
    path = app_data_dir(create=create) / "temp"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def voices_data_dir(create: bool = False) -> Path:
    """Return the user-local directory containing installed voice packs.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: Celune's user-local voice-pack directory.
    """
    path = app_data_dir(create=create) / "voices"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def backend_environments_dir(create: bool = False) -> Path:
    """Return the directory containing isolated backend environments.

    Args:
        create: Whether this directory should be created before being returned.

    Returns:
        Path: The Celune-local backend environment directory.
    """
    path = app_data_dir(create=create) / "environments"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _merge_directory_contents(source: Path, destination: Path) -> None:
    """Move non-conflicting contents from one directory into another."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if target.exists():
            if child.is_dir() and target.is_dir():
                _merge_directory_contents(child, target)
            continue
        shutil.move(str(child), str(target))

    with contextlib.suppress(OSError):
        source.rmdir()


def migrate_legacy_app_data() -> None:
    """Move legacy loose data into Celune's organized AppData layout.

    Existing destination files and directories are preserved. When a destination
    already exists, only non-conflicting legacy contents are moved, so an
    interrupted migration can safely resume on the next startup.
    """
    root = app_data_dir()
    for source_name, destination_parts in _LEGACY_APP_DATA_MIGRATIONS:
        source = root / source_name
        destination = root.joinpath(*destination_parts)
        if not source.is_dir():
            continue
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        elif source.is_dir() and destination.is_dir():
            _merge_directory_contents(source, destination)


def config_path(create_parent: bool = False) -> Path:
    """Return the active user configuration file path.

    Args:
        create_parent: Whether this directory's parents should be created before the path being returned.

    Returns:
        Path: Celune's user configuration file path.
    """
    path = app_data_dir(create=create_parent) / "config.yaml"
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def traceback_path(create_parent: bool = False) -> Path:
    """Return the traceback capture file path.

    Args:
        create_parent: Whether this directory's parents should be created before the path being returned.

    Returns:
        Path: Celune's traceback capture file path.
    """
    path = app_data_dir(create=create_parent) / f"{APP_SLUG}_traceback.txt"
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main_window_log_path(create_parent: bool = False) -> Path:
    """Return the persisted main-window log file path.

    Args:
        create_parent: Whether this directory's parents should be created before the path being returned.

    Returns:
        Path: Celune's main window log file path.
    """
    path = app_data_dir(create=create_parent) / f"{APP_SLUG}.log"
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir(create: bool = False) -> Path:
    """Return the repository-local generated outputs directory.

    Args:
        create: Whether the directory should be created before being returned.

    Returns:
        Path: Celune's generated outputs directory.
    """
    path = project_root() / "outputs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the repository root containing Celune's bundled defaults.

    Returns:
        Path: Celune's repository root directory.
    """
    if running_compiled():
        return _compiled_project_root()

    return Path(__file__).resolve().parent.parent


def core_python_executable() -> Path:
    """Return the Python interpreter that owns Celune's core environment."""
    if running_compiled():
        interpreter_name = "python.exe" if os.name == "nt" else "python"
        return (
            project_root()
            / ".venv"
            / ("Scripts" if os.name == "nt" else "bin")
            / interpreter_name
        )

    return Path(sys.executable).resolve()


def core_site_packages_dir() -> Path:
    """Return the site-packages directory for Celune's core environment."""
    if not running_compiled():
        return Path(sysconfig.get_paths()["purelib"]).resolve()

    interpreter = core_python_executable()
    virtualenv = interpreter.parent.parent
    if os.name == "nt":
        return virtualenv / "Lib" / "site-packages"

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return virtualenv / "lib" / python_version / "site-packages"


def default_config_path() -> Path:
    """Return the bundled default configuration file path.

    Returns:
        Path: Celune's default configuration file path.
    """
    return project_root() / "default_config.yaml"


def legacy_config_path() -> Path:
    """Return the historical repo-root config file path.

    Returns:
        Path: Celune's legacy configuration file path.
    """
    return project_root() / "config.yaml"


def ensure_config_path(
    active_path: Optional[Path] = None,
    default_path: Optional[Path] = None,
    legacy_path: Optional[Path] = None,
) -> tuple[Path, bool]:
    """Ensure Celune's active config file exists, migrating legacy config first.

    Args:
        active_path: Optional explicit active config file path.
        default_path: Optional explicit bundled default config file path.
        legacy_path: Optional explicit legacy repo-root config file path.

    Returns:
        tuple[Path, bool]: The resolved active config path and whether the file had to be created.
    """
    resolved_active = active_path or config_path(create_parent=True)
    resolved_default = default_path or default_config_path()
    resolved_legacy = legacy_path or legacy_config_path()

    resolved_active.parent.mkdir(parents=True, exist_ok=True)
    if resolved_active.exists():
        return resolved_active, False

    source_path = resolved_legacy if resolved_legacy.exists() else resolved_default
    shutil.copy(source_path, resolved_active)
    return resolved_active, True
