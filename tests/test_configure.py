# SPDX-License-Identifier: Apache-2.0
"""Tests for the source-tree configuration helper."""

from pathlib import Path
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

    (source_root / ".venv").rmdir()
    assert not configure.configuration_complete("Windows", ("sox",), source_root)


def test_confirm_repair_accepts_only_explicit_yes(monkeypatch) -> None:
    """Verify repair confirmation defaults to preserving a complete setup."""
    monkeypatch.setattr(configure.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert configure.confirm_repair()

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert not configure.confirm_repair()
