# SPDX-License-Identifier: Apache-2.0
"""Tests for the source-tree configuration helper."""

from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import configure
from configure import _sync_arguments, initialize_app_data


def test_sync_arguments_avoid_openzl_on_windows() -> None:
    """Verify Windows setup selects the API extra instead of every extra."""
    command = _sync_arguments("Windows", Path("uv.exe"))

    assert command == ["uv.exe", "sync", "--dev", "--extra", "api"]
    assert "--all-extras" not in command


def test_sync_arguments_include_linux_extras() -> None:
    """Verify Linux setup includes the complete optional dependency set."""
    assert _sync_arguments("Linux", Path("uv")) == [
        "uv",
        "sync",
        "--dev",
        "--all-extras",
    ]


def test_distro_package_keys_cover_supported_linux_distributions() -> None:
    """Verify supported distributions select their requested package managers."""
    expected_package_keys = {
        "Debian": "apt",
        "Ubuntu": "apt",
        "Linux Mint": "apt",
        "Pop!_OS": "apt",
        "Arch Linux": "pacman",
        "Manjaro": "pacman",
        "EndeavourOS": "pacman",
        "Fedora Linux": "dnf",
        "Rocky Linux": "dnf",
        "AlmaLinux": "dnf",
        "openSUSE Tumbleweed": "zypper",
        "Alpine Linux": "apk",
    }

    for distro_name, package_key in expected_package_keys.items():
        assert configure._distro_package_key(distro_name) == package_key


def test_try_install_uses_supported_package_manager_commands(monkeypatch) -> None:
    """Verify each supported Linux manager receives its non-interactive command."""
    commands = []
    monkeypatch.setattr(
        configure,
        "_privileged_command",
        lambda manager, arguments: [manager, *arguments],
    )
    monkeypatch.setattr(
        configure,
        "_run",
        lambda command: commands.append(command) or True,
    )

    for manager in ("apt", "pacman", "dnf", "zypper", "apk"):
        assert configure.try_install(manager, "sox")

    assert commands == [
        ["apt", "install", "-y", "sox"],
        ["pacman", "-S", "--noconfirm", "sox"],
        ["dnf", "install", "-y", "sox"],
        ["zypper", "--non-interactive", "install", "--no-confirm", "sox"],
        ["apk", "add", "sox"],
    ]


def test_initialize_app_data_seeds_only_missing_runtime_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify setup creates AppData and preserves an existing configuration."""
    source_root = tmp_path / "source"
    (source_root / "voices").mkdir(parents=True)
    (source_root / "default_config.yaml").write_text(
        "backend: mini\n", encoding="utf-8"
    )
    (source_root / "voices" / "default.cevoice").write_bytes(b"default voice")
    app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(app_data))

    assert initialize_app_data("Windows", source_root)
    config_path = app_data / "Celune" / "config.yaml"
    voice_path = app_data / "Celune" / "voices" / "default.cevoice"
    assert config_path.read_text(encoding="utf-8") == "backend: mini\n"
    assert voice_path.read_bytes() == b"default voice"

    config_path.write_text("backend: qwen3\n", encoding="utf-8")
    assert initialize_app_data("Windows", source_root)
    assert config_path.read_text(encoding="utf-8") == "backend: qwen3\n"


def test_configuration_complete_requires_environment_tools_and_runtime_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify setup prompts only after every configured resource is present."""
    source_root = tmp_path / "source"
    (source_root / "voices").mkdir(parents=True)
    (source_root / "default_config.yaml").write_text(
        "backend: mini\n", encoding="utf-8"
    )
    (source_root / "voices" / "default.cevoice").write_bytes(b"default voice")
    (source_root / ".venv").mkdir()
    (source_root / ".venv" / "pyvenv.cfg").write_text("home = uv\n", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setattr(configure, "resolve_uv", lambda: Path("uv.exe"))
    monkeypatch.setattr(configure, "resolve_binary", lambda name: name)

    assert initialize_app_data("Windows", source_root)
    assert configure.configuration_complete("Windows", ("sox",), source_root)

    (source_root / ".venv" / "pyvenv.cfg").unlink()
    (source_root / ".venv").rmdir()
    assert not configure.configuration_complete("Windows", ("sox",), source_root)


def test_confirm_repair_accepts_only_explicit_yes(monkeypatch) -> None:
    """Verify repair confirmation defaults to preserving a complete setup."""
    monkeypatch.setattr(configure.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert configure.confirm_repair()

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert not configure.confirm_repair()


def test_recreate_virtual_environment_removes_existing_tree(tmp_path: Path) -> None:
    """Verify repair mode removes the old environment before invoking uv."""
    virtual_environment = tmp_path / ".venv"
    virtual_environment.mkdir()
    (virtual_environment / "pyvenv.cfg").write_text("old", encoding="utf-8")
    uv = tmp_path / "uv.exe"

    with mock.patch.object(configure, "_run", return_value=True) as run:
        assert configure._recreate_virtual_environment(uv, tmp_path)

    assert not virtual_environment.exists()
    run.assert_called_once_with(
        [str(uv), "venv", str(virtual_environment.resolve())],
        cwd=tmp_path,
    )
