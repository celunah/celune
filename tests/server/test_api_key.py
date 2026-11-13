# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune API-key generation and persistence."""

import os
import re
import stat
from pathlib import Path

from celune.config import config_api_settings
from celune.server.auth import (
    generate_api_key,
    resolve_api_host,
    get_or_create_api_key,
    configure_api_security,
)
from celune.server.session import server_session


def test_generate_api_key_uses_the_celune_base36_format() -> None:
    """Verify generated keys use the fixed prefix and uppercase base-36 body."""
    key = generate_api_key()

    assert re.fullmatch(r"C-[0-9A-Z]{25}", key) is not None


def test_get_or_create_api_key_persists_a_private_generated_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a generated key is stable across calls and stored privately."""
    monkeypatch.delenv("CELUNE_API_TOKEN", raising=False)
    key_file = tmp_path / "api.key"

    key = get_or_create_api_key(path=key_file)

    assert get_or_create_api_key(path=key_file) == key
    assert key_file.read_text(encoding="utf-8").strip() == key
    if os.name != "nt":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_get_or_create_api_key_prefers_configured_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify explicit configuration takes precedence and avoids key-file writes."""
    monkeypatch.setenv("CELUNE_API_TOKEN", "environment-key")
    key_file = tmp_path / "api.key"

    assert get_or_create_api_key("configured-key", path=key_file) == "configured-key"
    assert not key_file.exists()


def test_get_or_create_api_key_uses_environment_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify CELUNE_API_TOKEN takes precedence over generated-key storage."""
    monkeypatch.setenv("CELUNE_API_TOKEN", "environment-key")
    key_file = tmp_path / "api.key"

    assert get_or_create_api_key(path=key_file) == "environment-key"
    assert not key_file.exists()


def test_config_api_settings_empty_token_disables_authentication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify an explicit empty token disables auth and overrides the environment."""
    monkeypatch.setenv("CELUNE_API_TOKEN", "environment-key")
    key_file = tmp_path / "api.key"

    settings = config_api_settings(
        {"api": {"token": ""}},
        lambda _message, _severity: None,
        path_factory=lambda _create_parent: key_file,
    )

    assert settings[3] == ""
    assert not key_file.exists()


def test_explicit_empty_token_disables_environment_authentication(monkeypatch) -> None:
    """Verify runtime auth and host resolution preserve an explicit empty token."""
    monkeypatch.setenv("CELUNE_API_TOKEN", "environment-key")
    monkeypatch.setattr(server_session, "auth_token", "previous-key")

    configure_api_security(token="")

    assert server_session.auth_token is None
    assert resolve_api_host(token="") == "127.0.0.1"
