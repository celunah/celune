# SPDX-License-Identifier: Apache-2.0
"""Resolve model revisions and store exact-revision license acknowledgements."""

from __future__ import annotations

import os
import re
import json
import tempfile
import threading
from typing import Optional
from pathlib import Path
from contextvars import Token, ContextVar
from collections.abc import Callable

from .types import ModelMetadata
from ..paths import model_consent_path
from ..exceptions import (
    ModelConsentDeclined,
    ModelConsentRequired,
    ModelCommercialUseRestricted,
)

_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_STORE_LOCK = threading.RLock()
_PENDING_LOCK = threading.RLock()
_POLICY_LOCK = threading.Lock()
_PENDING_CONSENTS: dict[tuple[str, str], ModelMetadata] = {}
_CONSENT_HANDLER: Optional[Callable[[ModelMetadata], bool]] = None
_API_MODEL_REQUEST = ContextVar("celune_api_model_request", default=False)
_COMMERCIAL_MODE = False


class ModelConsentStore:
    """Persist acknowledgements by Hugging Face model ID and immutable SHA."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or model_consent_path()
        self._acknowledged: set[tuple[str, str]] = set()
        self._modified_at_ns: Optional[int] = None
        self._load()

    def _load(self) -> None:
        """Read valid consent records while ignoring malformed store contents."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._modified_at_ns = self.path.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            self._modified_at_ns = None
            return
        if not isinstance(payload, dict):
            return
        entries = payload.get("acknowledged")
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("model_id")
            revision = entry.get("revision")
            if (
                isinstance(model_id, str)
                and isinstance(revision, str)
                and _COMMIT_SHA_PATTERN.fullmatch(revision) is not None
            ):
                self._acknowledged.add((model_id, revision))

    def has_acknowledged(self, model_id: str, revision: str) -> bool:
        """Return whether consent was recorded for this exact commit."""
        with _STORE_LOCK:
            try:
                modified_at_ns = self.path.stat().st_mtime_ns
            except OSError:
                modified_at_ns = None
            if modified_at_ns != self._modified_at_ns:
                self._acknowledged.clear()
                self._load()
            return (model_id, revision) in self._acknowledged

    def set_acknowledged(
        self,
        model_id: str,
        revision: str,
        acknowledged: bool,
    ) -> None:
        """Add or remove an acknowledgement and atomically persist the store."""
        if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
            raise ValueError("model consent requires an immutable commit SHA")
        with _STORE_LOCK:
            record = (model_id, revision)
            if acknowledged:
                self._acknowledged.add(record)
            else:
                self._acknowledged.discard(record)
            self._write()

    def _write(self) -> None:
        """Atomically write the current acknowledgement set."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "acknowledged": [
                {"model_id": model_id, "revision": revision}
                for model_id, revision in sorted(self._acknowledged)
            ]
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                json.dump(payload, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
                temporary_path = Path(output.name)
            os.replace(temporary_path, self.path)
            self._modified_at_ns = self.path.stat().st_mtime_ns
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


_STORE = ModelConsentStore()


def _catalog_model(model_id: str) -> Optional[ModelMetadata]:
    """Return registered model metadata without importing backend runtimes."""
    from . import get_model_metadata

    return get_model_metadata(model_id)


def set_commercial_mode(enabled: bool) -> None:
    """Set whether runtime model access must be cleared for commercial use."""
    global _COMMERCIAL_MODE
    with _POLICY_LOCK:
        _COMMERCIAL_MODE = enabled


def commercial_mode_enabled() -> bool:
    """Return whether the active runtime declares commercial use."""
    with _POLICY_LOCK:
        return _COMMERCIAL_MODE


def _require_commercial_compatibility(metadata: ModelMetadata) -> None:
    """Reject models whose catalog terms do not confirm commercial use."""
    if commercial_mode_enabled() and not metadata.commercial_use_allowed:
        raise ModelCommercialUseRestricted(metadata)


def resolve_model_metadata(
    model_id: str,
    revision: Optional[str] = None,
) -> ModelMetadata:
    """Resolve a model to catalog metadata and an immutable commit SHA.

    Built-in models always use the registry pin. Custom repositories resolve
    through Hugging Face before the caller can download or load them.
    """
    registered = _catalog_model(model_id)
    if registered is not None:
        if revision is not None and revision != registered.revision:
            raise ValueError(
                f"built-in model {model_id} must use its reviewed registry revision"
            )
        return registered

    if _COMMIT_SHA_PATTERN.fullmatch(revision or "") is not None:
        resolved_revision = revision
    else:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision)
        resolved_revision = info.sha
    if not isinstance(resolved_revision, str):
        raise TypeError(f"Hugging Face returned no commit SHA for {model_id}")
    if _COMMIT_SHA_PATTERN.fullmatch(resolved_revision) is None:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {model_id}")
    model_url = f"https://huggingface.co/{model_id}"
    return ModelMetadata(
        model_id=model_id,
        display_name=model_id,
        category="other",
        revision=resolved_revision,
        license_id=None,
        license_url=None,
        non_commercial=None,
        research_only=None,
        repository_url=model_url,
    )


def require_model_consent(
    model_id: str,
    revision: Optional[str] = None,
) -> ModelMetadata:
    """Return verified metadata or raise until required consent is recorded."""
    metadata = resolve_model_metadata(model_id, revision)
    _require_commercial_compatibility(metadata)
    if not metadata.requires_consent:
        return metadata
    if _STORE.has_acknowledged(metadata.model_id, metadata.revision):
        return metadata
    key = (metadata.model_id, metadata.revision)
    with _PENDING_LOCK:
        _PENDING_CONSENTS[key] = metadata
        handler = _CONSENT_HANDLER
    if handler is not None and not _API_MODEL_REQUEST.get():
        try:
            accepted = handler(metadata)
            if consent_status(metadata):
                return metadata
            record_model_consent(
                metadata.model_id,
                metadata.revision,
                accepted,
            )
            return metadata
        finally:
            with _PENDING_LOCK:
                _PENDING_CONSENTS.pop(key, None)
    raise ModelConsentRequired(metadata)


def set_model_consent_handler(
    handler: Optional[Callable[[ModelMetadata], bool]],
) -> None:
    """Register the main UI callback used for interactive confirmations."""
    global _CONSENT_HANDLER
    with _PENDING_LOCK:
        _CONSENT_HANDLER = handler


def enter_api_model_request() -> Token[bool]:
    """Mark an API or WebUI request so missing consent becomes an HTTP error."""
    return _API_MODEL_REQUEST.set(True)


def exit_api_model_request(token: Token[bool]) -> None:
    """Restore model-consent behavior after one API or WebUI request."""
    _API_MODEL_REQUEST.reset(token)


def record_model_consent(
    model_id: str,
    revision: str,
    accepted: bool,
) -> ModelMetadata:
    """Record or decline consent for one exact model revision."""
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("model consent requires an immutable commit SHA")
    metadata = _catalog_model(model_id)
    if metadata is None:
        metadata = ModelMetadata(
            model_id=model_id,
            display_name=model_id,
            category="other",
            revision=revision,
            license_id=None,
            license_url=None,
            non_commercial=None,
            research_only=None,
            repository_url=f"https://huggingface.co/{model_id}",
        )
    elif metadata.revision != revision:
        raise ValueError("consent revision does not match the registry pin")
    _require_commercial_compatibility(metadata)
    with _PENDING_LOCK:
        _PENDING_CONSENTS.pop((model_id, revision), None)
    _STORE.set_acknowledged(model_id, revision, accepted)
    if not accepted:
        raise ModelConsentDeclined(metadata)
    return metadata


def pending_model_consents() -> tuple[ModelMetadata, ...]:
    """Return all unresolved model confirmations for interactive clients."""
    with _PENDING_LOCK:
        return tuple(_PENDING_CONSENTS.values())


def consent_status(metadata: ModelMetadata) -> bool:
    """Return whether the exact metadata revision has acknowledged terms."""
    return _STORE.has_acknowledged(metadata.model_id, metadata.revision)


def registered_models() -> tuple[ModelMetadata, ...]:
    """Return every statically cataloged model in category order."""
    from . import ALL_MODELS

    return ALL_MODELS


def consent_prompt_text(metadata: ModelMetadata) -> tuple[str, str]:
    """Return the two localized lines required for the main UI prompt."""
    from ..i18n import string

    source = metadata.license_url or metadata.repository_url
    return (
        string(
            "registry.consent_prompt",
            model_name=metadata.display_name,
            license_url=source,
        ),
        string("registry.consent_instruction"),
    )


def consent_details(metadata: ModelMetadata) -> tuple[str, ...]:
    """Return localized license and restriction details for confirmations."""
    from ..i18n import string
    from ..constants import APP_NAME

    license_text = metadata.license_id or string("registry.license_unknown")
    return (
        string("registry.license_detail", license=license_text),
        string(
            "registry.non_commercial_detail",
            value=string(
                "registry.flag_unknown"
                if metadata.non_commercial is None
                else "registry.flag_yes"
                if metadata.non_commercial
                else "registry.flag_no"
            ),
        ),
        string(
            "registry.research_detail",
            value=string(
                "registry.flag_unknown"
                if metadata.research_only is None
                else "registry.flag_yes"
                if metadata.research_only
                else "registry.flag_no"
            ),
        ),
        string("registry.terms_need_review")
        if metadata.terms_review_required
        else string("registry.terms_reviewed", app_name=APP_NAME),
        string("registry.model_source", url=metadata.repository_url),
    )


def model_metadata_for_revision(model_id: str, revision: str) -> ModelMetadata:
    """Resolve listed API metadata for a static or caller-supplied exact SHA."""
    metadata = _catalog_model(model_id)
    if metadata is not None:
        if metadata.revision != revision:
            raise ValueError("model revision does not match the registry pin")
        return metadata
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("model revision must be a full commit SHA")
    return ModelMetadata(
        model_id=model_id,
        display_name=model_id,
        category="other",
        revision=revision,
        license_id=None,
        license_url=None,
        non_commercial=None,
        research_only=None,
        repository_url=f"https://huggingface.co/{model_id}",
    )
