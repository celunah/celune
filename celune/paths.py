# SPDX-License-Identifier: MIT
"""Runtime filesystem paths for Celune user data."""

import sys
import shutil
from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir

from .constants import APP_SLUG

_REPO_MARKERS = ("celune", "default_config.yaml", "pyproject.toml")


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
    path = Path(user_data_dir(APP_SLUG, appauthor=False))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


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


def project_root() -> Path:
    """Return the repository root containing Celune's bundled defaults.

    Returns:
        Path: Celune's repository root directory.
    """
    if running_compiled():
        return _compiled_project_root()

    return Path(__file__).resolve().parent.parent


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
