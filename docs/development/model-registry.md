# Model registry and license consent

This page is for Celune contributors who add model metadata or change a model loading path. It describes the central catalog, pinned revisions, and the consent required before a restricted, unknown, or unreviewed model can load.

## Catalog model metadata

`celune.registry` is the source of truth for built-in model metadata and runtime selections. Catalogs are grouped by major type:

| Module | Contents |
| --- | --- |
| `celune.registry.tts` | TTS models, voice selections, model contracts, and related runtime pins. |
| `celune.registry.llm` | Persona and agent language models, tiers, and revisions. |
| `celune.registry.asr` | Speech recognition models and revisions. |
| `celune.registry.other` | Singleton support models such as alignment, embeddings, emotion, normalization, Needle, and voice conversion. |

Each `ModelMetadata` entry records its model ID, display name, category, immutable Hugging Face commit SHA, repository URL, license identifier and source URL, restriction flags, and any runtime roles. A missing license or restriction value is represented by `None`; Celune treats unknown terms as requiring consent. Set `terms_review_required` when the published terms need human review, even if a license identifier exists.

## Commercial use mode

The top-level `commercial` key in `default_config.yaml` declares the runtime's
use. Its default is `false`, which retains the normal consent flow. When set to
`true`, the shared model access gate permits a model only when the registry has
a license and source URL, both restriction flags are explicitly false, and the
terms do not need review. Unknown, non-commercial, research-only, and unreviewed
models are rejected before consent or any download/load path. Custom models
remain unavailable in this mode until their terms are reviewed and cataloged.

The model listing API reports the active `commercial_mode` and each model's
`commercial_use_allowed` metadata. A blocked operation returns HTTP `403` with
the model details and a localized explanation.

Built-in revisions are reviewed pins. Update a pin only after checking the repository revision and its license information. Custom Hugging Face IDs resolve to a full commit SHA before download, and remain unknown-license models until they are added to a catalog.

Keep backend implementation dispatch and dependency environment manifests in their backend modules. Consumers should import model metadata from `celune.registry` modules rather than maintain a second list.

## Consent behavior

Celune stores acknowledgement in its app-data directory, keyed by the exact model ID and commit SHA. A later revision requires a new acknowledgement. No prior agreement is inferred for existing installations.

Before any cataloged or custom model is downloaded or loaded, the calling path uses `require_model_consent`. The guard covers startup preload, Textual and browser UI actions, REST requests, and CEDTS worker-backed loads. A missing acknowledgement raises `ModelConsentRequired`; declining raises `ModelConsentDeclined`. Neither path permits the model load to continue.

The main Textual UI displays the localized `registry.consent_prompt` and `registry.consent_instruction` messages, and requires the exact phrase from `registry.consent_phrase`. Its bottom-left status uses `registry.waiting_status` while it waits. The browser UI shows the same localized license details, restriction flags, and source link in an obstructing modal, and updates its footer while consent is pending.

## REST consent endpoints

All endpoints use the API's configured authentication. `GET /v1/models` returns catalog metadata, immutable revisions, and whether each model needs or has consent. `POST /v1/models/consent` records a decision for one exact revision:

```json
{
  "model_id": "Plachta/ASTRAL-quantization",
  "revision": "4a2e9679f76eb03753adc8c503e3c23bb9c22f26",
  "accepted": true
}
```

An accepted decision returns the model metadata. If a REST operation reaches a model that lacks consent, Celune returns HTTP `428 Precondition Required` with the model metadata and the operation must be retried after acknowledgement. A declined decision returns HTTP `403 Forbidden` and leaves the model unloaded. Consent applies only to the submitted model ID and commit SHA.

## Verification and recovery

Run the canonical project validation after changing a catalog or any load path:

```powershell
uv run ruff format .
python scripts/run_ci.py
```

For documentation changes, also build MkDocs in strict mode and run MarkdownLint with `.markdownlint.json`.

If an operation returns `428`, read the returned license source and submit consent for its model ID and revision, then retry the operation. If the license is unknown or marked for review, review the repository terms before accepting. If the user declines, select another model or cancel the operation; Celune will continue to require consent for that revision.

## See also

- [REST API](../API.md) for endpoint authentication and request behavior.
- [Licensing](../licensing.md) for Celune source licensing and third-party model terms.
- [Backend development](backends.md) for backend ownership and validation boundaries.
