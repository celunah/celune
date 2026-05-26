# SPDX-License-Identifier: MIT
"""Celune-managed Persona runtime helpers."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Iterator
from typing import Callable, Optional

from ..config import Config
from ..constants import JSONSerializable, PERSONA_MODEL_ID
from ..vram import resolve_vram_preset
from .runtime import PersonaRuntime, request_from_json, response_to_json

PERSONA_QUANTIZATION = "4bit"
DevLogCallback = Callable[[str, str], None]


class PersonaClientResponse:
    """Small response shim matching the local HTTP client contract."""

    def __init__(self, payload: dict[str, JSONSerializable]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mirror the ``httpx`` response API for local in-process calls."""

    def json(self) -> dict[str, JSONSerializable]:
        """Return the stored response payload."""
        return dict(self._payload)


class PersonaClient:
    """In-process Persona client adapter used by Celune."""

    def __init__(
        self,
        config: Optional[Mapping[str, JSONSerializable]] = None,
        log_dev: Optional[DevLogCallback] = None,
    ) -> None:
        self.runtime = PersonaRuntime(config=config)
        self.config = config
        self.log_dev = log_dev

    @contextlib.contextmanager
    def _capture_backend_output(self) -> Iterator[None]:
        """Route Persona backend stdout/stderr into Celune developer logs."""
        if self.log_dev is None:
            yield
            return

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            yield

        for line in stderr_buffer.getvalue().splitlines():
            text = line.strip()
            if text:
                self.log_dev(f"[PERSONA] {text}", "warning")

    def load(
        self,
        model_id: str,
        quantization: str = PERSONA_QUANTIZATION,
    ) -> None:
        """Explicitly load the Persona runtime."""
        with self._capture_backend_output():
            self.runtime.load(model_id, quantization)

    def post(self, json: dict[str, JSONSerializable]) -> PersonaClientResponse:
        """Handle a Persona generation request without leaving the process."""
        request = request_from_json(json)
        with self._capture_backend_output():
            response = self.runtime.generate(request)
        return PersonaClientResponse(response_to_json(response))

    def close(self) -> None:
        """Release Persona runtime state."""
        self.runtime.close()


def persona_config(config: Mapping[str, JSONSerializable]) -> Config:
    """Return the normalized configuration block for the persona system.

    Args:
        config: The configuration data for the persona system.

    Returns:
        Config: The normalized configuration data for the persona system.
    """
    raw = config.get("persona", config.get("pyop", {}))
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raw = {}

    return dict(raw)


def persona_enabled(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether Celune should try to use personas."""
    return resolve_vram_preset(config).persona_enabled and bool(
        persona_config(config).get("enabled", True)
    )


def persona_talkback_enabled(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether regular UI input should go through persona talkback."""
    return persona_enabled(config) and bool(
        persona_config(config).get("talkback", True)
    )


def persona_quantization(config: Mapping[str, JSONSerializable]) -> str:
    """Return the Persona quantization mode permitted by the VRAM tier."""
    return resolve_vram_preset(config).persona_quantization


def persona_model_id(config: Optional[Mapping[str, JSONSerializable]] = None) -> str:
    """Return the Persona model ID Celune should load."""
    if config is not None:
        configured = persona_config(config).get("model_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()

    return PERSONA_MODEL_ID


def persona_is_available() -> bool:
    """Check whether the in-process Persona runtime can be used."""
    try:
        PersonaRuntime()
        return True
    except Exception:
        return False


def create_persona_client(
    config: Optional[Mapping[str, JSONSerializable]] = None,
    log_dev: Optional[DevLogCallback] = None,
) -> Optional[PersonaClient]:
    """Create a Celune-managed in-process Persona client when enabled."""
    if config is not None and not persona_enabled(config):
        return None

    if not persona_is_available():
        return None

    return PersonaClient(config=config, log_dev=log_dev)
