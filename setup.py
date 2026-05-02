#!/usr/bin/env python3

import tomllib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def get_version() -> str:
    with Path("pyproject.toml").open("rb") as file:
        data = tomllib.load(file)

    return data["project"]["version"]

def try_install(manager: str, package: str) -> bool:
    commands = {
        "pacman": [manager, "-S", "--noconfirm", package],
        "apt-get": [manager, "install", "-y", package],
        "scoop": [manager, "install", package],
    }

    cmd = commands.get(manager)
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
        print(f"I couldn't install {package}.")

    return False


def ensure_uv() -> bool:
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

    try:
        subprocess.run(cmd, check=True, text=True)
        return shutil.which("scoop") is not None
    except FileNotFoundError:
        print("I can't run PowerShell. Are you sure you're on Windows?")
    except subprocess.CalledProcessError:
        print("I couldn't install Scoop successfully.")

    return False


def get_distro_name(system: str) -> str:
    if system != "Linux":
        return system

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
    package_manager = "scoop"
    ok &= ensure_scoop()
elif distro == "Arch Linux":
    package_manager = "pacman"
elif distro in {"Debian", "Ubuntu"}:
    package_manager = "apt-get"
    print("Warning: OpenRGB needs to be set up manually.")
else:
    print(f"I don't run on {distro}.")
    sys.exit(1)

ok &= ensure_uv()

for package, binary in zip(required_packages[name], required_bins[name], strict=True):
    path = shutil.which(binary)
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
