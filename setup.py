#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Celune automatic setup utility.

No, this is not a setuptools remnant, this is Celune's bootstrapper.
"""

import sys
import os
import shutil
import tomllib
import platform
import subprocess
import contextlib
from pathlib import Path
from typing import Optional

with contextlib.suppress(IndexError):
    arg = sys.argv[1]

    if arg in [
        "build",
        "install",
        "develop",
        "sdist",
        "bdist_wheel",
        "clean",
        "egg_info",
    ]:
        print("I am not meant to run via `setuptools`. Run me alone.")
        print("I'll proceed anyway, I warned you already.")


def get_version() -> str:
    """Get current Celune version without importing Celune, using Celune's project metadata.

    Returns:
        str: The current Celune version.
    """
    with Path("pyproject.toml").open("rb") as file:
        data = tomllib.load(file)

    return data["project"]["version"]


def try_install(manager: str, package_name: str) -> bool:
    """Attempt to install a given package using the selected package manager.

    Args:
        manager: The package manager to use.
        package_name: The name of the package to install.

    Returns:
        bool: If the package was successfully installed.
    """
    commands = {
        "pacman": [manager, "-S", "--noconfirm", package_name],
        "apt-get": [manager, "install", "-y", package_name],
        "scoop": [manager, "install", package_name],
    }

    cmd = commands.get(Path(manager).stem)
    if cmd is None:
        print("I'm not sure what package manager you have.")
        return False

    try:
        subprocess.run(cmd, text=True, check=True)
        return True
    except FileNotFoundError:
        print("I can't find your package manager.")
    except PermissionError:
        print("I don't have permissions to run your package manager.")
    except subprocess.CalledProcessError:
        print(f"I couldn't install {package_name}.")

    return False


def ensure_uv() -> bool:
    """Ensure that `uv` is installed.

    Returns:
        bool: Whether `uv` is already installed or was installed successfully.
    """
    if shutil.which("uv"):
        return True

    print("Installing uv...")

    if platform.system() == "Windows":
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    else:
        cmd = [
            "sh",
            "-c",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ]

    try:
        subprocess.run(cmd, text=True, check=True)
        return shutil.which("uv") is not None
    except FileNotFoundError:
        print("I can't run the uv installer.")
    except subprocess.CalledProcessError:
        print("I couldn't install uv. I can't run like this.")

    return False


def ensure_scoop() -> bool:
    """Ensure that `scoop` is installed.

    Returns:
        bool: Whether `scoop` is already installed or was installed successfully.
    """
    if shutil.which("scoop"):
        return True

    print("Installing Scoop...")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "iwr -useb https://get.scoop.sh | iex",
    ]

    activate_extras_cmd = ["scoop", "bucket", "add", "extras"]

    try:
        subprocess.run(cmd, check=True, text=True)
        subprocess.run(activate_extras_cmd, check=True, text=True)
        return shutil.which("scoop") is not None
    except FileNotFoundError:
        print("I can't run PowerShell. Are you sure you're on Windows?")
    except subprocess.CalledProcessError:
        print("I couldn't install Scoop successfully.")

    return False


def resolve_openrgb() -> Optional[str]:
    """Find OpenRGB even when its installer did not add it to PATH.

    Returns:
        Optional[str]: The resolved OpenRGB executable path, or None if unavailable.
    """
    for binary_name in ("openrgb", "OpenRGB"):
        binary_path = shutil.which(binary_name)
        if binary_path:
            return binary_path

    system_data = platform.system()
    if system_data == "Windows":
        candidates = []

        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if base:
                candidates.extend(
                    [
                        Path(base) / "OpenRGB" / "OpenRGB.exe",
                        Path(base) / "OpenRGB" / "openrgb.exe",
                    ]
                )

        scoop = os.environ.get("SCOOP") or str(Path.home() / "scoop")
        candidates.extend(
            [
                Path(scoop) / "apps" / "openrgb" / "current" / "OpenRGB.exe",
                Path(scoop) / "apps" / "openrgb" / "current" / "openrgb.exe",
            ]
        )
    elif system_data == "Linux":
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
    """Resolve a required executable.

    Args:
        binary_name: The executable name to resolve.

    Returns:
        Optional[str]: The resolved executable path, or None if unavailable.
    """
    if binary_name == "openrgb":
        return resolve_openrgb()

    return shutil.which(binary_name)


def get_distro_name(system_data: str) -> str:
    """Get current distribution name.

    Args:
        system_data: The system data to pull the distribution name from.

    Returns:
        str: The distribution name.
    """
    if system_data != "Linux":
        return system_data

    try:
        return platform.freedesktop_os_release().get("NAME", "Linux")
    except OSError:
        return "Linux"


system = platform.system()
arch = platform.machine()
distro = get_distro_name(system)
name = distro.split(maxsplit=1)[0].lower()

if system not in {"Windows", "Linux"}:
    print(f"I don't run on {system}.")
    sys.exit(1)


print(f"Setting up Celune {get_version()} on {distro} ({arch})")

required_packages = {
    "ubuntu": ["sox", "rubberband-cli"],
    "debian": ["sox", "rubberband-cli"],
    "arch": ["sox", "rubberband", "openrgb"],
    "windows": ["sox", "rubberband", "openrgb"],
}

required_bins = {
    "ubuntu": ["sox", "rubberband"],
    "debian": ["sox", "rubberband"],
    "arch": ["sox", "rubberband", "openrgb"],
    "windows": ["sox", "rubberband", "openrgb"],
}

ok = True

if system == "Windows":
    package_manager = shutil.which("scoop") or ""
    ok &= ensure_scoop()
elif distro == "Arch Linux":
    package_manager = shutil.which("pacman") or ""
elif distro in {"Debian", "Ubuntu"}:
    package_manager = shutil.which("apt-get") or ""
    print("Warning: OpenRGB needs to be set up manually.")
else:
    print(f"I don't run on {distro}.")
    sys.exit(1)

ok &= ensure_uv()

for package, binary in zip(required_packages[name], required_bins[name], strict=True):
    path = resolve_binary(binary)
    if path:
        print(f"{binary} found at {path}")
        continue

    print(f"{binary} not found, installing {package}...")
    ok &= try_install(package_manager, package)

try:
    subprocess.run(["uv", "sync"], text=True, check=True)
except FileNotFoundError:
    print("I can't find uv.")
    ok = False
except PermissionError:
    print("I don't have permission to run uv.")
    ok = False
except subprocess.CalledProcessError:
    print("uv sync failed")
    ok = False

if not ok:
    print("I couldn't install some packages.")
    sys.exit(1)

print("I'm ready to go.")
