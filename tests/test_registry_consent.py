# SPDX-License-Identifier: Apache-2.0
"""Tests for the central model catalog and exact-revision consent store."""

import re
import json
import asyncio
from typing import cast
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from starlette.requests import Request

from celune import api
from celune.i18n import string
from celune.backends.vc import seedvc
from celune.registry.consent import (
    ModelConsentStore,
    consent_prompt_text,
    record_model_consent,
    require_model_consent,
    resolve_model_metadata,
)
from celune.registry.other import OTHER_MODELS
from celune.cedts.remote import RemoteBackendProxy
from celune.registry import ALL_MODELS, get_model_metadata
from celune.agent.needle import checkpoints as needle_checkpoints
from celune.exceptions import ModelConsentDeclined, ModelConsentRequired


def test_catalogs_have_unique_pinned_models_and_explicit_unknown_terms() -> None:
    """Verify every built-in has one immutable revision and explicit terms."""
    model_ids = [metadata.model_id for metadata in ALL_MODELS]

    assert len(model_ids) == len(set(model_ids))
    assert all(re.fullmatch(r"[0-9a-f]{40}", model.revision) for model in ALL_MODELS)
    assert all(
        model.license_id for model in ALL_MODELS if not model.terms_review_required
    )
    assert all(
        model.repository_url.startswith("https://huggingface.co/")
        for model in ALL_MODELS
    )

    unknown = get_model_metadata("Plachta/ASTRAL-quantization")
    assert unknown is not None
    assert unknown in OTHER_MODELS
    assert unknown.license_id is None
    assert unknown.non_commercial is None
    assert unknown.research_only is None
    assert unknown.requires_consent


def test_consent_store_is_bound_to_one_exact_revision(tmp_path: Path) -> None:
    """Verify persisted acknowledgement for one SHA does not cover another."""
    store_path = tmp_path / "consent.json"
    first_revision = "a" * 40
    second_revision = "b" * 40
    store = ModelConsentStore(store_path)

    store.set_acknowledged("custom/model", first_revision, True)
    reloaded = ModelConsentStore(store_path)

    assert reloaded.has_acknowledged("custom/model", first_revision)
    assert not reloaded.has_acknowledged("custom/model", second_revision)


def test_custom_models_resolve_to_pinned_unknown_license_metadata() -> None:
    """Verify an unregistered repository remains gated after SHA resolution."""
    revision = "c" * 40
    metadata = resolve_model_metadata("custom/model", revision)

    assert metadata.revision == revision
    assert metadata.license_id is None
    assert metadata.non_commercial is None
    assert metadata.research_only is None
    assert metadata.requires_consent

    with (
        mock.patch("celune.registry.consent._STORE") as store,
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
        mock.patch("celune.registry.consent._CONSENT_HANDLER", None),
    ):
        store.has_acknowledged.return_value = False
        try:
            require_model_consent("custom/model", revision)
        except ModelConsentRequired as error:
            assert error.metadata == metadata
        else:
            raise AssertionError("unknown model terms were not gated")


def test_custom_hugging_face_repositories_resolve_to_commit_shas() -> None:
    """Resolve a mutable branch through Hugging Face before model use."""
    revision = "d" * 40

    with mock.patch(
        "huggingface_hub.HfApi.model_info",
        return_value=SimpleNamespace(sha=revision),
    ) as model_info:
        metadata = resolve_model_metadata("custom/model")

    assert metadata.revision == revision
    assert metadata.requires_consent
    model_info.assert_called_once_with("custom/model", revision=None)


def test_needle_checkpoint_requires_consent_before_downloading() -> None:
    """Reject worker checkpoint preparation before it fetches custom model files."""
    revision = "e" * 40

    with (
        mock.patch("celune.registry.consent._STORE") as store,
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
        mock.patch("celune.registry.consent._CONSENT_HANDLER", None),
        mock.patch.object(needle_checkpoints, "_download") as download,
    ):
        store.has_acknowledged.return_value = False
        try:
            needle_checkpoints.prepare_needle_checkpoint(
                "custom/needle",
                revision,
                Path("unused-cache"),
            )
        except ModelConsentRequired:
            pass
        else:
            raise AssertionError("unacknowledged Needle terms were not rejected")

    download.assert_not_called()


def test_seed_vc_blocks_model_load_when_any_role_needs_consent() -> None:
    """Check Seed-VC licensing before importing the model wrapper."""
    metadata = get_model_metadata("Plachta/ASTRAL-quantization")
    assert metadata is not None
    backend = seedvc.CeluneSeedVCBackend(log=lambda _message, _level: None)

    with (
        mock.patch("celune.registry.consent._STORE") as store,
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
        mock.patch("celune.registry.consent._CONSENT_HANDLER", None),
        mock.patch.object(backend, "_load_wrapper_type") as load_wrapper,
    ):
        store.has_acknowledged.return_value = False
        try:
            backend._get_wrapper()
        except ModelConsentRequired as error:
            assert error.metadata == metadata
        else:
            raise AssertionError("Seed-VC loaded without required model consent")

    load_wrapper.assert_not_called()


def test_main_ui_consent_prompt_matches_the_requested_copy() -> None:
    """Verify the localized TUI prompt includes the exact acceptance phrase."""
    metadata = get_model_metadata("Plachta/ASTRAL-quantization")
    assert metadata is not None

    prompt, instruction = consent_prompt_text(metadata)

    assert prompt == (
        "Please read the license terms for ASTRAL Quantization: "
        "https://huggingface.co/Plachta/ASTRAL-quantization"
    )
    assert instruction == 'Once read, type "I consent" to continue.'
    assert string("registry.waiting_status") == "Waiting for consent"


def test_consent_api_records_and_withdraws_the_exact_revision(tmp_path: Path) -> None:
    """Verify the consent endpoint persists the decision for the catalog pin."""
    metadata = get_model_metadata("Plachta/ASTRAL-quantization")
    assert metadata is not None
    store = ModelConsentStore(tmp_path / "consent.json")

    with (
        mock.patch("celune.registry.consent._STORE", store),
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
    ):
        listed = cast(dict[str, list[dict[str, object]]], api.models())["models"]
        catalog_entry = next(
            item for item in listed if item["model_id"] == metadata.model_id
        )
        assert catalog_entry["revision"] == metadata.revision
        assert catalog_entry["requires_consent"] is True

        accepted = api.submit_model_consent(
            api.ModelConsentRequest(
                model_id=metadata.model_id,
                revision=metadata.revision,
                accepted=True,
            )
        )
        assert accepted["accepted"] is True
        assert store.has_acknowledged(metadata.model_id, metadata.revision)

        try:
            api.submit_model_consent(
                api.ModelConsentRequest(
                    model_id=metadata.model_id,
                    revision=metadata.revision,
                    accepted=False,
                )
            )
        except ModelConsentDeclined:
            pass
        else:
            raise AssertionError("declining model terms was not rejected")

    assert not store.has_acknowledged(metadata.model_id, metadata.revision)


def test_missing_consent_uses_http_428_and_decline_uses_http_403() -> None:
    """Verify REST errors expose exact model metadata and reject a decline."""
    metadata = get_model_metadata("Plachta/ASTRAL-quantization")
    assert metadata is not None

    required = asyncio.run(
        api.model_consent_required_response(
            cast(Request, None),
            ModelConsentRequired(metadata),
        )
    )
    declined = asyncio.run(
        api.model_consent_declined_response(
            cast(Request, None),
            ModelConsentDeclined(metadata),
        )
    )

    assert required.status_code == 428
    required_payload = json.loads(bytes(required.body).decode("utf-8"))
    assert declined.status_code == 403
    assert required_payload["model"]["model_id"] == metadata.model_id


def test_declining_terms_does_not_create_an_acknowledgement(tmp_path: Path) -> None:
    """Verify a declined decision leaves later model operations blocked."""
    metadata = get_model_metadata("Plachta/ASTRAL-quantization")
    assert metadata is not None
    store = ModelConsentStore(tmp_path / "consent.json")

    with (
        mock.patch("celune.registry.consent._STORE", store),
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
    ):
        try:
            record_model_consent(metadata.model_id, metadata.revision, False)
        except ModelConsentDeclined:
            pass
        else:
            raise AssertionError("a declined consent decision was accepted")
        try:
            require_model_consent(metadata.model_id, metadata.revision)
        except ModelConsentRequired:
            pass
        else:
            raise AssertionError("a declined model was allowed to load")


def test_worker_proxy_does_not_load_before_consent(tmp_path: Path) -> None:
    """Verify the parent process gates a CEDTS model load before worker IPC."""
    store = ModelConsentStore(tmp_path / "consent.json")
    proxy = object.__new__(RemoteBackendProxy)
    proxy._request = mock.Mock()

    with (
        mock.patch("celune.registry.consent._STORE", store),
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
    ):
        try:
            proxy.load_model("Plachta/ASTRAL-quantization")
        except ModelConsentRequired:
            pass
        else:
            raise AssertionError("worker-backed load proceeded without consent")

    proxy._request.assert_not_called()
