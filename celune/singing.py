# SPDX-License-Identifier: Apache-2.0
"""Prepare singing notation for TTS backend conversion."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING
from collections.abc import Callable

from .cesn import (
    _NOTE_EVENT_PATTERN,
    _resolve_active_singer,
    _tokens,
    parse_cesn,
    render_cesn,
)
from .exceptions import CESNNotationWarning

if TYPE_CHECKING:
    from .celune import Celune


def _contains_cesn_note_event(text: str) -> bool:
    """Return whether CESN tags contain at least one valid note or rest event."""
    for is_tag, value in _tokens(text):
        if not is_tag or value.strip().startswith("#"):
            continue
        if any(
            _NOTE_EVENT_PATTERN.fullmatch(event.strip()) for event in value.split(",")
        ):
            return True
    return False


def _prepare_synthesis_text(engine: Celune, text: str) -> str:
    """Convert CESN before Celune splits long requests into chunks."""
    if not getattr(engine.backend, "singing", False):
        return text
    if "[" in text:
        with warnings.catch_warnings(record=True) as notation_warnings:
            warnings.simplefilter("always", CESNNotationWarning)
            sections = parse_cesn(
                text,
                supported_languages=tuple(engine.backend.supported_languages),
            )
        if any(section.singer is None for section in sections):
            active_singer = _resolve_active_singer()
            sections = tuple(
                replace(section, singer=active_singer)
                if section.singer is None
                else section
                for section in sections
            )
        for notation_warning in notation_warnings:
            engine.log(str(notation_warning.message), "warning")
        text = render_cesn(sections)
    return engine.backend.cesn_to_model(text)


def _prepare_synthesis_chunks(
    engine: Celune,
    text: str,
    splitter: Callable[[Celune, str], list[str]],
) -> list[str]:
    """Convert CESN once and preserve complete scores as a single request."""
    prepared_text = _prepare_synthesis_text(engine, text)
    if getattr(engine.backend, "singing", False):
        return [prepared_text]
    return splitter(engine, prepared_text)
