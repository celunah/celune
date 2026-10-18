# SPDX-License-Identifier: Apache-2.0
"""Runtime model reload helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .exceptions import NotAvailableError

if TYPE_CHECKING:
    from .celune import Celune


def reload_model_for_language(
    engine: Celune,
    target_language: Optional[str],
) -> None:
    """Reload a backend model when the requested language needs a new variant.

    Args:
        engine: Runtime that owns the active model and backend.
        target_language: Language requested by the current speech operation.

    Raises:
        NotAvailableError: A required model reference or active voice is missing.
        Exception: The backend fails while unloading or loading the model.
    """
    backend = engine.backend
    should_reload = getattr(backend, "should_reload_for_language", None)
    if engine.model is None and not callable(should_reload):
        raise NotAvailableError("cannot generate without a model reference")
    if not callable(should_reload) or (
        engine.model is not None and not should_reload(target_language)
    ):
        return

    active_voice = engine.current_voice or backend.default_voice
    if active_voice is None:
        raise NotAvailableError("cannot switch language without an active voice")

    model_id = backend.model_id_for_voice(active_voice)
    engine.log(
        f"[RELOAD] Loading {model_id} for language: {target_language}",
        loglevel="verbose",
    )
    engine.model = None
    backend.unload_model()
    engine.model = backend.load_model(model_id, lang=target_language)
    engine.model_name = model_id
