# SPDX-License-Identifier: Apache-2.0
"""Tests for the source-tree configuration helper."""

import tomllib
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import configure
from configure import sync_arguments, initialize_app_data


def test_sync_arguments_avoid_openzl_on_windows() -> None:
    """Verify Windows setup selects API and the default CUDA extra."""
    command = sync_arguments("Windows", Path("uv.exe"))

    assert command == [
        "uv.exe",
        "sync",
        "--dev",
        "--extra",
        "api",
        "--extra",
        "cu132",
    ]
    assert "--all-extras" not in command


def test_sync_arguments_include_linux_extras() -> None:
    """Verify Linux setup selects supported extras and one CUDA index."""
    assert sync_arguments("Linux", Path("uv")) == [
        "uv",
        "sync",
        "--dev",
        "--extra",
        "api",
        "--extra",
        "captions",
        "--extra",
        "cechar4-openzl",
        "--extra",
        "cu132",
    ]


def test_sync_arguments_allow_a_different_cuda_index() -> None:
    """Verify setup can use any declared CUDA index."""
    assert sync_arguments("Windows", Path("uv"), "cu128")[-2:] == [
        "--extra",
        "cu128",
    ]


def test_sync_arguments_reject_unknown_cuda_index() -> None:
    """Verify setup does not construct an invalid CUDA extra."""
    try:
        sync_arguments("Linux", Path("uv"), "cu120")
    except ValueError as error:
        assert str(error) == "unsupported CUDA index: cu120"
    else:
        raise AssertionError("unknown CUDA index was accepted")


def test_torch_runtime_dependencies_are_only_in_cuda_extras() -> None:
    """Verify default project syncs cannot resolve another Torch build."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    lock_path = pyproject_path.with_name("uv.lock")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    cuda_indexes = configure.PYTORCH_INDEXES
    torch_runtime_packages = {
        "accelerate",
        "bitsandbytes",
        "silero-vad",
        "torch",
        "torchao",
        "torchaudio",
        "torchvision",
    }

    def requirement_name(requirement: str) -> str:
        """Return the normalized package name from a plain requirement."""
        return requirement.split(">", maxsplit=1)[0].split("=", maxsplit=1)[0]

    default_packages = {
        requirement_name(requirement) for requirement in project["dependencies"]
    }
    assert default_packages.isdisjoint(torch_runtime_packages)

    for cuda_index in cuda_indexes:
        extra_packages = {
            requirement_name(requirement)
            for requirement in project["optional-dependencies"][cuda_index]
        }
        assert torch_runtime_packages <= extra_packages

    sources = pyproject["tool"]["uv"]["sources"]
    for package_name in ("torch", "torchaudio", "torchvision"):
        source_indexes = {
            source["extra"]: source["index"] for source in sources[package_name]
        }
        assert source_indexes == {
            cuda_index: f"pytorch-{cuda_index}" for cuda_index in cuda_indexes
        }

        locked_packages = [
            package for package in lock["package"] if package["name"] == package_name
        ]
        assert locked_packages
        assert all("+cu" in package["version"] for package in locked_packages)


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
        lambda _manager, arguments: [_manager, *arguments],
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
