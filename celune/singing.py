# SPDX-License-Identifier: Apache-2.0
"""Prepare singing notation for TTS backend conversion."""

from __future__ import annotations

import warnings
from dataclasses import replace
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from .cesn import (
    NOTE_EVENT_PATTERN,
    parse_cesn,
    render_cesn,
    resolve_active_singer,
    tokens,
)
from .i18n import string
from .exceptions import CESNError, CESNNotationWarning


if TYPE_CHECKING:
    from .celune import Celune


def contains_cesn_note_event(text: str) -> bool:
    """Return whether CESN tags contain at least one valid note or rest event."""
    for is_tag, value in tokens(text):
        if not is_tag or value.strip().startswith("#"):
            continue
        if any(
            NOTE_EVENT_PATTERN.fullmatch(event.strip()) for event in value.split(",")
        ):
            return True
    return False


def format_cesn_error(text: str, error: CESNError) -> str:
    """Format a CESN failure with its source line and one-based location."""
    offset = error.source_start
    if offset is None:
        return str(error)
    offset = min(max(offset, 0), len(text))
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end].removesuffix("\r")
    line_number = text.count("\n", 0, line_start) + 1
    column_number = offset - line_start + 1
    caret_prefix = text[line_start:offset].expandtabs()
    header = string(
        "cesn.parse_error_location",
        line=line_number,
        column=column_number,
    )
    return f"{header}\n\n{line}\n{caret_prefix}^ {error}"


def validate_singing_notation(engine: Celune, text: str) -> Optional[str]:
    """Return a user-readable error when input is not valid CESN notation."""
    if not contains_cesn_note_event(text):
        return string("cesn.no_notation_found")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CESNNotationWarning)
            parse_cesn(
                text,
                supported_languages=tuple(engine.backend.supported_languages),
            )
    except CESNError as error:
        return format_cesn_error(text, error)
    return None


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
            active_singer = resolve_active_singer()
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


def prepare_synthesis_chunks(
    engine: Celune,
    text: str,
    splitter: Callable[[Celune, str], list[str]],
) -> list[str]:
    """Convert CESN once and preserve complete scores as a single request."""
    prepared_text = _prepare_synthesis_text(engine, text)
    if getattr(engine.backend, "singing", False):
        return [prepared_text]
    return splitter(engine, prepared_text)
