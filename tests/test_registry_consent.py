# SPDX-License-Identifier: Apache-2.0
"""Tests for the central model catalog and exact-revision consent store."""

import re
import json
import asyncio
from types import SimpleNamespace
from typing import cast
from pathlib import Path
from unittest import mock

from starlette.requests import Request

from celune import api
from celune.i18n import string
from celune.registry import ALL_MODELS, get_model_metadata
from celune.exceptions import (
    ModelConsentDeclined,
    ModelConsentRequired,
    ModelCommercialUseRestricted,
)
from celune.backends.vc import seedvc
from celune.agent.needle import checkpoints as needle_checkpoints
from celune.cedts.remote import RemoteBackendProxy
from celune.registry.other import (
    OTHER_MODELS,
    SOULX_SINGER_RMVPE_MODEL_ID,
    SOULX_SINGER_RMVPE_REVISION,
)
from celune.registry.consent import (
    ModelConsentStore,
    consent_prompt_text,
    set_commercial_mode,
    record_model_consent,
    require_model_consent,
    resolve_model_metadata,
)


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

    astral = get_model_metadata("Plachta/ASTRAL-quantization")
    assert astral is not None
    assert astral in OTHER_MODELS
    assert astral.license_id == "gpl-3.0"
    assert astral.license_url == "https://www.gnu.org/licenses/gpl-3.0.html"
    assert astral.non_commercial is False
    assert astral.research_only is False
    assert not astral.terms_review_required
    assert astral.commercial_use_allowed
    assert not astral.requires_consent

    unknown = resolve_model_metadata("custom/model", "c" * 40)
    assert unknown.license_id is None
    assert unknown.non_commercial is None
    assert unknown.research_only is None
    assert unknown.requires_consent


def test_soulxsinger_rmvpe_checkpoint_has_reviewed_model_metadata() -> None:
    """Register the optional high-precision F0 extractor under its pinned terms."""
    metadata = get_model_metadata(SOULX_SINGER_RMVPE_MODEL_ID)

    assert metadata is not None
    assert metadata.revision == SOULX_SINGER_RMVPE_REVISION
    assert metadata.license_id == "apache-2.0"
    assert metadata.commercial_use_allowed


def test_commercial_mode_requires_reviewed_commercial_terms() -> None:
    """Allow clear catalog entries and block restricted or unknown terms."""
    commercially_allowed = next(
        metadata
        for metadata in ALL_MODELS
        if metadata.commercial_use_allowed and not metadata.requires_consent
    )
    astral = get_model_metadata("Plachta/ASTRAL-quantization")
    firered = get_model_metadata("FireRedTeam/FireRedTTS3")
    dots_mf = get_model_metadata("rednote-hilab/dots.tts-mf")
    unknown = resolve_model_metadata("custom/model", "c" * 40)
    assert astral is not None
    assert firered is not None
    assert dots_mf is not None
    assert astral.license_id == "gpl-3.0"
    assert astral.commercial_use_allowed
    assert not astral.requires_consent
    assert firered.research_only is False
    assert dots_mf.research_only is False
    assert not dots_mf.terms_review_required
    assert firered.commercial_use_allowed
    assert dots_mf.commercial_use_allowed
    assert not firered.requires_consent
    assert not dots_mf.requires_consent

    set_commercial_mode(True)
    try:
        for metadata in (commercially_allowed, astral, firered, dots_mf):
            assert require_model_consent(metadata.model_id) == metadata
        with mock.patch(
            "celune.registry.consent._STORE.has_acknowledged",
            return_value=True,
        ):
            try:
                require_model_consent(unknown.model_id, unknown.revision)
            except ModelCommercialUseRestricted as error:
                assert error.metadata == unknown
            else:
                raise AssertionError(
                    "commercial mode allowed model terms that were not cleared"
                )
        with mock.patch(
            "celune.registry.consent._STORE.set_acknowledged"
        ) as save_acknowledgement:
            try:
                record_model_consent(unknown.model_id, unknown.revision, True)
            except ModelCommercialUseRestricted:
                pass
            else:
                raise AssertionError(
                    "commercially restricted model consent was recorded"
                )
            save_acknowledgement.assert_not_called()
    finally:
        set_commercial_mode(False)


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


def test_seed_vc_reviewed_models_do_not_require_consent() -> None:
    """Verify reviewed Seed-VC dependencies do not trigger consent prompts."""
    with (
        mock.patch("celune.registry.consent._STORE") as store,
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
        mock.patch("celune.registry.consent._CONSENT_HANDLER", None),
    ):
        store.has_acknowledged.return_value = False
        seedvc._require_seedvc_model_consents()

    store.has_acknowledged.assert_not_called()


def test_main_ui_consent_prompt_matches_the_requested_copy() -> None:
    """Verify the localized TUI prompt includes the exact acceptance phrase."""
    metadata = resolve_model_metadata("custom/model", "c" * 40)

    prompt, instruction = consent_prompt_text(metadata)

    assert prompt == (
        "Please read the license terms for custom/model: "
        "https://huggingface.co/custom/model"
    )
    assert instruction == 'Once read, type "I consent" to continue.'
    assert string("registry.waiting_status") == "Waiting for consent"


def test_consent_api_records_and_withdraws_the_exact_revision(tmp_path: Path) -> None:
    """Verify the consent endpoint persists the decision for the catalog pin."""
    metadata = resolve_model_metadata("custom/model", "c" * 40)
    store = ModelConsentStore(tmp_path / "consent.json")

    with (
        mock.patch("celune.registry.consent._STORE", store),
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
    ):
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
    metadata = resolve_model_metadata("custom/model", "c" * 40)

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


def test_commercial_policy_uses_http_403() -> None:
    """Verify a restricted model returns an actionable commercial-mode error."""
    metadata = resolve_model_metadata("custom/restricted-model", "d" * 40)
    astral = get_model_metadata("Plachta/ASTRAL-quantization")
    assert astral is not None
    set_commercial_mode(True)
    try:
        model_listing = api.models()
        response = asyncio.run(
            api.model_commercial_use_restricted_response(
                cast(Request, None),
                ModelCommercialUseRestricted(metadata),
            )
        )
    finally:
        set_commercial_mode(False)

    payload = json.loads(bytes(response.body).decode("utf-8"))
    assert response.status_code == 403
    assert payload["error"] == "commercial_model_restricted"
    assert payload["model"]["commercial_use_allowed"] is False
    assert model_listing["commercial_mode"] is True
    listed_models = cast(list[dict[str, object]], model_listing["models"])
    catalog_models = {cast(str, model["model_id"]): model for model in listed_models}
    assert catalog_models[astral.model_id]["commercial_use_allowed"] is True


def test_declining_terms_does_not_create_an_acknowledgement(tmp_path: Path) -> None:
    """Verify a declined decision leaves later model operations blocked."""
    metadata = resolve_model_metadata("custom/model", "c" * 40)
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
    metadata = resolve_model_metadata("custom/model", "c" * 40)
    proxy = object.__new__(RemoteBackendProxy)
    proxy.request = mock.Mock()

    with (
        mock.patch("celune.registry.consent._STORE", store),
        mock.patch("celune.registry.consent._PENDING_CONSENTS", {}),
        mock.patch(
            "celune.registry.consent.resolve_model_metadata",
            return_value=metadata,
        ),
    ):
        try:
            proxy.load_model(metadata.model_id)
        except ModelConsentRequired:
            pass
        else:
            raise AssertionError("worker-backed load proceeded without consent")

    proxy.request.assert_not_called()


def test_worker_proxy_blocks_commercially_restricted_models() -> None:
    """Verify commercial policy is enforced before worker model-load IPC."""
    proxy = object.__new__(RemoteBackendProxy)
    proxy.request = mock.Mock()
    metadata = resolve_model_metadata("custom/restricted-model", "d" * 40)
    set_commercial_mode(True)
    try:
        try:
            with mock.patch(
                "celune.registry.consent.resolve_model_metadata",
                return_value=metadata,
            ):
                proxy.load_model(metadata.model_id)
        except ModelCommercialUseRestricted:
            pass
        else:
            raise AssertionError("commercially restricted worker model was loaded")
    finally:
        set_commercial_mode(False)

    proxy.request.assert_not_called()
