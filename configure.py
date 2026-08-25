#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a complete Celune source-tree installation."""

import os
import sys
import shutil
import platform
import tomllib
import subprocess
from pathlib import Path
from typing import Optional
from collections.abc import Sequence

APP_NAME = "Celune"
PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIRECTORIES = (
    "environments",
    "huggingface/hub",
    "memory",
    "persona",
    "runtime",
    "temp",
    "voices",
)


def get_version(root: Path = PROJECT_ROOT) -> str:
    """Return the project version from ``pyproject.toml``."""
    with (root / "pyproject.toml").open("rb") as file:
        data = tomllib.load(file)

    return str(data["project"]["version"])


def _run(command: Sequence[str], cwd: Path = PROJECT_ROOT) -> bool:
    """Run one setup command and return whether it completed successfully."""
    print(f"+ {' '.join(command)}")
    try:
        result = subprocess.run(command, cwd=cwd, check=False, text=True)
    except (FileNotFoundError, PermissionError) as error:
        print(f"Could not run {command[0]}: {error}")
        return False
    return result.returncode == 0


def _uv_candidates() -> tuple[Path, ...]:
    """Return common installation paths for uv."""
    candidates = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".local" / "bin" / "uv.exe",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "uv" / "uv.exe")
    scoop = os.environ.get("SCOOP")
    if scoop:
        candidates.append(Path(scoop) / "shims" / "uv.exe")
    return tuple(candidates)


def resolve_uv() -> Optional[Path]:
    """Find the uv executable in PATH or its standard install locations."""
    for name in ("uv.exe", "uv"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    for candidate in _uv_candidates():
        if candidate.is_file():
            return candidate
    return None


def ensure_uv() -> Optional[Path]:
    """Ensure uv is installed and return its executable path."""
    uv = resolve_uv()
    if uv is not None:
        return uv

    print("Installing uv...")
    if platform.system() == "Windows":
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    else:
        command = ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]

    if not _run(command):
        print("uv could not be installed.")
        return None

    uv = resolve_uv()
    if uv is None:
        print("uv was installed, but its executable could not be found.")
    return uv


def ensure_scoop() -> bool:
    """Ensure Scoop and its extras bucket are available on Windows."""
    if shutil.which("scoop"):
        return True

    print("Installing Scoop...")
    if not _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "iwr -useb https://get.scoop.sh | iex",
        ]
    ):
        return False

    return _run(["scoop", "bucket", "add", "extras"])


def resolve_openrgb() -> Optional[str]:
    """Find OpenRGB even when its installer did not add it to PATH."""
    for binary_name in ("openrgb", "OpenRGB"):
        binary_path = shutil.which(binary_name)
        if binary_path:
            return binary_path

    system_name = platform.system()
    if system_name == "Windows":
        candidates = []
        for environment_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(environment_name)
            if base:
                candidates.extend(
                    (
                        Path(base) / "OpenRGB" / "OpenRGB.exe",
                        Path(base) / "OpenRGB" / "openrgb.exe",
                    )
                )
        scoop = os.environ.get("SCOOP") or str(Path.home() / "scoop")
        candidates.extend(
            (
                Path(scoop) / "apps" / "openrgb" / "current" / "OpenRGB.exe",
                Path(scoop) / "apps" / "openrgb" / "current" / "openrgb.exe",
            )
        )
    elif system_name == "Linux":
        candidates = [
            Path("/usr/bin/openrgb"),
            Path("/usr/local/bin/openrgb"),
            Path("/opt/OpenRGB/openrgb"),
            Path("/opt/OpenRGB/OpenRGB"),
        ]
    else:
        candidates = []

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_binary(binary_name: str) -> Optional[str]:
    """Resolve one required system executable."""
    if binary_name == "openrgb":
        return resolve_openrgb()
    return shutil.which(binary_name)


def get_distro_name(system_name: str) -> str:
    """Return a readable operating-system distribution name."""
    if system_name != "Linux":
        return system_name

    try:
        return platform.freedesktop_os_release().get("NAME", "Linux")
    except OSError:
        return "Linux"


def _distro_package_key(distro_name: str) -> Optional[str]:
    """Return the package-manager key for a supported Linux distribution."""
    normalized_name = " ".join(distro_name.casefold().replace("_", " ").split())
    distro_aliases = {
        "debian": "apt",
        "ubuntu": "apt",
        "linux mint": "apt",
        "mint": "apt",
        "pop! os": "apt",
        "arch linux": "pacman",
        "arch": "pacman",
        "manjaro": "pacman",
        "manjaro linux": "pacman",
        "endeavouros": "pacman",
        "fedora": "dnf",
        "fedora linux": "dnf",
        "rocky": "dnf",
        "rocky linux": "dnf",
        "alma": "dnf",
        "almalinux": "dnf",
        "alma linux": "dnf",
        "opensuse": "zypper",
        "opensuse tumbleweed": "zypper",
        "opensuse leap": "zypper",
        "alpine": "apk",
        "alpine linux": "apk",
    }
    package_key = distro_aliases.get(normalized_name)
    if package_key is not None:
        return package_key

    for alias, package_key in distro_aliases.items():
        if normalized_name.startswith(f"{alias} "):
            return package_key
    return None


def _privileged_command(manager: str, arguments: Sequence[str]) -> list[str]:
    """Build a package-manager command with sudo when Linux needs it."""
    command = [manager, *arguments]
    if (
        platform.system() == "Linux"
        and hasattr(os, "geteuid")
        and os.geteuid() != 0
        and shutil.which("sudo")
    ):
        return ["sudo", *command]
    return command


def try_install(manager: str, package_name: str) -> bool:
    """Install one system package with the selected package manager."""
    manager_name = Path(manager).stem.casefold()
    if manager_name == "pacman":
        command = _privileged_command(manager, ("-S", "--noconfirm", package_name))
    elif manager_name in {"apt", "apt-get", "dnf"}:
        command = _privileged_command(manager, ("install", "-y", package_name))
    elif manager_name == "zypper":
        command = _privileged_command(
            manager, ("--non-interactive", "install", "--no-confirm", package_name)
        )
    elif manager_name == "apk":
        command = _privileged_command(manager, ("add", package_name))
    elif manager_name == "scoop":
        command = [manager, "install", package_name]
    else:
        print(f"Unsupported package manager: {manager}")
        return False

    return _run(command)


def _app_data_dir(system_name: str) -> Path:
    """Return Celune's platform-specific application-data directory."""
    if system_name == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / APP_NAME


def _copy_if_missing(source: Path, destination: Path) -> bool:
    """Copy one immutable setup asset without overwriting user data."""
    if destination.exists():
        return True
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as error:
        print(f"Could not copy {source} to {destination}: {error}")
        return False
    print(f"Copied {source.name} to {destination}")
    return True


def initialize_app_data(system_name: str, root: Path = PROJECT_ROOT) -> bool:
    """Create Celune's AppData tree and seed essential runtime assets."""
    data_root = _app_data_dir(system_name)
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        for relative_path in RUNTIME_DIRECTORIES:
            (data_root / relative_path).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Could not create Celune AppData at {data_root}: {error}")
        return False

    assets = (
        (root / "default_config.yaml", data_root / "config.yaml"),
        (root / "voices" / "default.cevoice", data_root / "voices" / "default.cevoice"),
    )
    for source, destination in assets:
        if not source.is_file():
            print(f"Required runtime data is missing: {source}")
            return False
        if not _copy_if_missing(source, destination):
            return False

    print(f"Celune AppData ready at {data_root}")
    return True


def configuration_complete(
    system_name: str,
    required_bins: Sequence[str],
    root: Path = PROJECT_ROOT,
) -> bool:
    """Return whether the environment and essential runtime data are ready."""
    data_root = _app_data_dir(system_name)
    required_directories = all(
        (data_root / path).is_dir() for path in RUNTIME_DIRECTORIES
    )
    required_files = (
        data_root / "config.yaml",
        data_root / "voices" / "default.cevoice",
    )

    return all(
        (
            resolve_uv() is not None,
            (root / ".venv" / "pyvenv.cfg").is_file(),
            (root / "default_config.yaml").is_file(),
            (root / "voices" / "default.cevoice").is_file(),
            required_directories,
            all(path.is_file() for path in required_files),
            all(
                resolve_binary(binary_name) is not None for binary_name in required_bins
            ),
        )
    )


def confirm_repair() -> bool:
    """Ask whether an already configured Celune installation should be repaired."""
    if not sys.stdin.isatty():
        print("Celune is already configured.")
        return False

    try:
        answer = input("You have already configured Celune. Repair Celune? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer.strip().lower() in {"y", "yes"}:
        return True

    print("Celune is already configured.")
    return False


def _sync_arguments(system_name: str, uv: Path) -> list[str]:
    """Return the platform-safe dependency synchronization command."""
    command = [str(uv), "sync", "--dev"]
    if system_name == "Windows":
        command.extend(("--extra", "api"))
    else:
        command.append("--all-extras")
    return command


def _recreate_virtual_environment(uv: Path, root: Path = PROJECT_ROOT) -> bool:
    """Remove and recreate the project virtual environment for repair mode.

    Args:
        uv: Path to the uv executable.
        root: Source-tree root containing the virtual environment.

    Returns:
        bool: Whether the new virtual environment was created successfully.
    """
    virtual_environment = (root / ".venv").resolve()
    if Path(sys.prefix).resolve() == virtual_environment:
        print(
            "Cannot repair the active Python environment; run configure.py outside .venv."
        )
        return False

    if virtual_environment.exists():
        print(f"Removing existing Python environment at {virtual_environment}...")
        try:
            shutil.rmtree(virtual_environment)
        except OSError as error:
            print(f"Could not remove the existing Python environment: {error}")
            return False

    return _run([str(uv), "venv", str(virtual_environment)], cwd=root)


def main() -> int:
    """Run system dependency, Python environment, and AppData setup."""
    system_name = platform.system()
    distro_name = get_distro_name(system_name)
    architecture = platform.machine()
    if system_name not in {"Windows", "Linux"}:
        print(f"Celune does not support {system_name}.")
        return 1

    print(f"Setting up {APP_NAME} {get_version()} on {distro_name} ({architecture})")

    required_packages = {
        "apt": ("sox", "rubberband-cli"),
        "pacman": ("sox", "rubberband", "openrgb"),
        "dnf": ("sox", "rubberband", "openrgb"),
        "zypper": ("sox", "rubberband", "OpenRGB"),
        "apk": ("sox", "rubberband", "openrgb"),
        "windows": ("sox", "rubberband", "openrgb"),
    }
    required_bins = {
        "apt": ("sox", "rubberband"),
        "pacman": ("sox", "rubberband", "openrgb"),
        "dnf": ("sox", "rubberband", "openrgb"),
        "zypper": ("sox", "rubberband", "openrgb"),
        "apk": ("sox", "rubberband", "openrgb"),
        "windows": ("sox", "rubberband", "openrgb"),
    }

    if system_name == "Windows":
        package_manager = shutil.which("scoop") or "scoop"
        ok = True
        package_key = "windows"
    else:
        package_key = _distro_package_key(distro_name)
        if package_key is None:
            print(f"Celune does not support {distro_name}.")
            return 1
        package_manager = shutil.which(package_key) or package_key
        ok = True

        if package_key == "apk":
            print(
                "Alpine Linux support is experimental; some packages may potentially be unsupported."
            )

    repair_requested = False
    if configuration_complete(system_name, required_bins[package_key]):
        repair_requested = confirm_repair()
        if not repair_requested:
            return 0

    if system_name == "Windows":
        ok = ensure_scoop()
    elif package_key == "apt" and not _run(
        _privileged_command(package_manager, ("update",))
    ):
        ok = False

    uv = ensure_uv()
    ok = uv is not None and ok

    for package_name, binary_name in zip(
        required_packages[package_key], required_bins[package_key], strict=True
    ):
        binary_path = resolve_binary(binary_name)
        if binary_path:
            print(f"{binary_name} found at {binary_path}")
            continue

        print(f"{binary_name} not found, installing {package_name}...")
        if not try_install(package_manager, package_name):
            ok = False
            continue
        if resolve_binary(binary_name) is None:
            print(f"{binary_name} is still unavailable after installation.")
            ok = False

    if uv is None:
        print("Celune cannot continue without uv.")
        return 1
    if repair_requested and not _recreate_virtual_environment(uv):
        print("Celune setup did not complete successfully.")
        return 1
    if not _run(_sync_arguments(system_name, uv)):
        print("Celune's Python environment could not be synchronized.")
        ok = False

    if not initialize_app_data(system_name):
        ok = False

    if not ok:
        print("Celune setup did not complete successfully.")
        return 1

    print("Celune is ready to go.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
