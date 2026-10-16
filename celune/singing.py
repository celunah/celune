# SPDX-License-Identifier: Apache-2.0
"""Prepare singing notation for TTS backend conversion."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

from .exceptions import CESNNotationWarning
from .cesn import _resolve_active_singer, parse_cesn, render_cesn

if TYPE_CHECKING:
    from .celune import Celune


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
