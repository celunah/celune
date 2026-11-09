# SPDX-License-Identifier: Apache-2.0
"""Runtime filesystem paths and global Hugging Face runtime setup for Celune."""

from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
import importlib
import sysconfig
import threading
import contextlib
import contextvars
from pathlib import Path
from typing import Union, Literal, Optional
from collections.abc import Callable, Generator

from .constants import APP_NAME, APP_SLUG

_REPO_MARKERS = ("celune", "default_config.yaml", "pyproject.toml")
_HF_HOME_ENV = "HF_HOME"
_HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
_HF_HUB_DISABLE_PROGRESS_BARS_ENV = "HF_HUB_DISABLE_PROGRESS_BARS"
_HF_PROGRESS_PATCH_LOCK = threading.RLock()
_HF_PROGRESS_REPORTER: contextvars.ContextVar[
    Optional[_HuggingFaceProgressReporter]
] = contextvars.ContextVar("celune_huggingface_progress_reporter", default=None)
_HF_PROGRESS_WRAPPER_CLASS: Optional[type] = None
_HF_PROGRESS_PATCH_ORIGINALS: Optional[tuple[type, type]] = None
_HF_PROGRESS_ACTIVE_CONTEXTS = 0
_LEGACY_APP_DATA_MIGRATIONS = (("backends", ("environments",)),)


class _HuggingFaceProgressReporter:
    """Combine concurrent Hugging Face transfer bars into one progress value."""

    _TRANSFER_BAR = "huggingface_hub.snapshot_download.transfer"
    _FILE_BAR = "huggingface_hub.http_get"

    def __init__(
        self,
        callback: Callable[[Optional[float], Optional[float]], None],
    ) -> None:
        self._callback = callback
        self._bars: dict[int, tuple[str, float, Optional[float]]] = {}
        self._lock = threading.Lock()
        self._last_report = 0.0
        self._active = True

    def deactivate(self) -> None:
        """Stop forwarding updates from bars that outlive their download scope."""
        with self._lock:
            self._active = False

    def report(
        self,
        bar_id: int,
        name: str,
        progress: float,
        total: Optional[Union[int, float]],
        *,
        force: bool = False,
    ) -> None:
        """Record one transfer bar and publish its aggregate state."""
        if name not in {self._TRANSFER_BAR, self._FILE_BAR}:
            return
        normalized_total = (
            float(total)
            if isinstance(total, (int, float))
            and not isinstance(total, bool)
            and total > 0
            else None
        )
        with self._lock:
            if not self._active:
                return
            normalized_progress = max(0.0, progress)
            if normalized_total is not None:
                normalized_progress = min(normalized_progress, normalized_total)
            self._bars[bar_id] = (
                name,
                normalized_progress,
                normalized_total,
            )
            now = time.monotonic()
            if not force and now - self._last_report < 0.1:
                return
            self._last_report = now
            transfer_bars = [
                progress_entry
                for progress_entry in self._bars.values()
                if progress_entry[0] == self._TRANSFER_BAR
            ]
            bars = transfer_bars or [
                progress_entry
                for progress_entry in self._bars.values()
                if progress_entry[0] == self._FILE_BAR
            ]
            known_bars = [
                progress_entry
                for progress_entry in bars
                if progress_entry[2] is not None
            ]
            if known_bars:
                reported_total = sum(
                    progress_entry[2]
                    for progress_entry in known_bars
                    if progress_entry[2] is not None
                )
                reported_progress = min(
                    sum(progress_entry[1] for progress_entry in known_bars),
                    reported_total,
                )
            else:
                reported_progress = None
                reported_total = None
        with contextlib.suppress(Exception):
            self._callback(reported_progress, reported_total)


def _make_huggingface_progress_class(base_class: type) -> type:
    """Build the shared Hugging Face bar wrapper without importing it at startup."""

    class CeluneHuggingFaceTqdm(base_class):
        """Keep Hub bars quiet and forward aggregate file-transfer progress."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            initial = kwargs.get("initial", 0)
            self._celune_downloaded = (
                float(initial) if isinstance(initial, (int, float)) else 0.0
            )
            name = kwargs.get("name")
            self._celune_progress_name = name if isinstance(name, str) else None
            self._celune_progress_reporter = _HF_PROGRESS_REPORTER.get()
            self._celune_progress_lock = threading.Lock()
            super().__init__(*args, **kwargs)
            self._report_celune_progress(force=True)

        def update(self, n: Optional[float] = 1) -> Optional[bool]:
            """Add transferred bytes and report the updated aggregate."""
            result = super().update(n)
            if isinstance(n, (int, float)) and not isinstance(n, bool):
                with self._celune_progress_lock:
                    self._celune_downloaded = max(
                        0.0,
                        self._celune_downloaded + float(n),
                    )
            self._report_celune_progress()
            return result

        def close(self) -> None:
            """Report the final byte count before closing the quiet bar."""
            self._report_celune_progress(force=True)
            super().close()

        def _report_celune_progress(
            self,
            force: bool = False,
        ) -> None:
            """Send this bar's transfer state to its scoped progress reporter."""
            reporter = self._celune_progress_reporter
            name = self._celune_progress_name
            if reporter is None or name is None:
                return
            with self._celune_progress_lock:
                progress = self._celune_downloaded
            reporter.report(
                id(self),
                name,
                progress,
                getattr(self, "total", None),
                force=force,
            )

    return CeluneHuggingFaceTqdm


def user_data_dir(
    appname: Optional[str] = None,
    appauthor: Optional[Union[str, Literal[False]]] = None,
    version: Optional[str] = None,
    roaming: bool = False,
    ensure_exists: bool = False,
    use_site_for_root: bool = False,
) -> str:
    """Resolve a platform-specific user-data directory on demand."""
    from platformdirs import user_data_dir as resolve_user_data_dir

    return resolve_user_data_dir(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
        use_site_for_root=use_site_for_root,
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


def configure_huggingface_cache_environment() -> None:
    """Point Hugging Face caches at Celune's user data directory."""
    default_hf_home = str(huggingface_home_dir())
    default_hf_hub_cache = str(huggingface_hub_cache_dir())

    # use app-managed HF cache, so you can easily take her runtime data
    # and apply it elsewhere
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


# noinspection PyPep8Naming
@contextlib.contextmanager
def huggingface_progress(
    callback: Optional[Callable[[Optional[float], Optional[float]], None]],
) -> Generator[None, None, None]:
    """Forward aggregate Hugging Face transfer progress to Celune.

    Args:
        callback: Receiver for aggregate downloaded bytes and their total, or
            ``None`` to leave the Hub unchanged.
    """
    global _HF_PROGRESS_ACTIVE_CONTEXTS
    global _HF_PROGRESS_PATCH_ORIGINALS
    global _HF_PROGRESS_WRAPPER_CLASS
    if callback is None:
        yield
        return

    from huggingface_hub.utils.tqdm import tqdm as HuggingFaceTqdm

    progress_module = importlib.import_module("huggingface_hub.utils.tqdm")
    snapshot_download = importlib.import_module("huggingface_hub._snapshot_download")

    reporter = _HuggingFaceProgressReporter(callback)
    with _HF_PROGRESS_PATCH_LOCK:
        if _HF_PROGRESS_PATCH_ORIGINALS is None:
            previous_progress_tqdm = progress_module.__dict__["tqdm"]
            previous_snapshot_tqdm = snapshot_download.__dict__["hf_tqdm"]
            _HF_PROGRESS_PATCH_ORIGINALS = (
                previous_progress_tqdm,
                previous_snapshot_tqdm,
            )
            _HF_PROGRESS_WRAPPER_CLASS = _make_huggingface_progress_class(
                HuggingFaceTqdm
            )
            progress_module.__dict__["tqdm"] = _HF_PROGRESS_WRAPPER_CLASS
            snapshot_download.__dict__["hf_tqdm"] = _HF_PROGRESS_WRAPPER_CLASS
        _HF_PROGRESS_ACTIVE_CONTEXTS += 1
    context_token = _HF_PROGRESS_REPORTER.set(reporter)
    try:
        yield
    finally:
        _HF_PROGRESS_REPORTER.reset(context_token)
        reporter.deactivate()
        with _HF_PROGRESS_PATCH_LOCK:
            _HF_PROGRESS_ACTIVE_CONTEXTS -= 1
            if _HF_PROGRESS_ACTIVE_CONTEXTS == 0:
                originals = _HF_PROGRESS_PATCH_ORIGINALS
                if originals is not None:
                    progress_module.__dict__["tqdm"] = originals[0]
                    snapshot_download.__dict__["hf_tqdm"] = originals[1]
                _HF_PROGRESS_PATCH_ORIGINALS = None
                _HF_PROGRESS_WRAPPER_CLASS = None


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


def temp_directory_environment() -> dict[str, str]:
    """Return environment variables that route child temp files into Celune data."""
    directory = str(temp_data_dir(create=True))
    return {variable: directory for variable in ("TMPDIR", "TEMP", "TMP")}


def configure_temp_environment() -> None:
    """Route Python and child-process temporary files into Celune's temp directory."""
    environment = temp_directory_environment()
    os.environ.update(environment)
    tempfile.tempdir = environment["TMPDIR"]


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


def api_key_path(create_parent: bool = False) -> Path:
    """Return the private file path used for Celune's generated API key.

    Args:
        create_parent: Whether the app-data directory should be created first.

    Returns:
        Path: Celune's generated API-key file path.
    """
    return app_data_dir(create=create_parent) / "api.key"


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


def model_consent_path() -> Path:
    """Return the app-data path used for model-license acknowledgements."""
    return app_data_dir() / "model-consents.json"


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
