# SPDX-License-Identifier: Apache-2.0
"""Tests for CESN syntax parsing and note-to-syllable alignment."""

import pytest

from celune.exceptions import (
    CESNAlignmentError,
    CESNMatchError,
    CESNSyntaxError,
)
from celune.cesn import CESNNoteEvent, parse_cesn


def test_parse_cesn_returns_note_event_and_syllable_pairs() -> None:
    """Parse note durations, dotted notes, and ordinary lyric syllables."""
    result = parse_cesn("[c3:4d,d3:8,e3:16] Text to sing.")

    assert result == (
        (CESNNoteEvent("c3", 4, True), "Text "),
        (CESNNoteEvent("d3", 8, False), "to "),
        (CESNNoteEvent("e3", 16, False), "sing."),
    )


def test_parse_cesn_splits_multisyllabic_and_hyphenated_words() -> None:
    """Assign one event to each Syllabreak-detected orthographic syllable."""
    result = parse_cesn(
        "[c3:1,d3:2,e3:4,f3:8,g3:16,a3:32,b3:2] Moonlight resting, silver-blue."
    )

    assert tuple(syllable for _, syllable in result) == (
        "Moon",
        "light ",
        "re",
        "sting, ",
        "sil",
        "ver-",
        "blue.",
    )


def test_parse_cesn_uses_multilingual_syllabreak_rules() -> None:
    """Use Syllabreak language detection for supported non-English lyrics."""
    result = parse_cesn("[c3:4,d3:4] здраво")

    assert tuple(syllable for _, syllable in result) == ("здра", "во")


def test_parse_cesn_aligns_the_default_singing_reference() -> None:
    """Align the bundled singing transcript to Syllabreak's syllables."""
    note_list = ",".join("c3:4" for _ in range(14))
    result = parse_cesn(
        f"[{note_list}] Moonlight resting, silver-blue, I'll keep one quiet light for you."
    )

    assert tuple(syllable for _, syllable in result) == (
        "Moon",
        "light ",
        "re",
        "sting, ",
        "sil",
        "ver-",
        "blue, ",
        "I'll ",
        "keep ",
        "one ",
        "quiet ",
        "light ",
        "for ",
        "you.",
    )


def test_parse_cesn_preserves_leading_punctuation() -> None:
    """Keep opening quotation marks attached to the first syllable."""
    result = parse_cesn("[c3:4,d3:4] “Hello!”")

    assert tuple(syllable for _, syllable in result) == ("“Hel", "lo!”")


@pytest.mark.parametrize(
    "text",
    (
        "c3:4] Text",
        "[c3:3] Text",
        "[c3:4,,d3:8] Text",
        "[c3:4]",
        "[c3:4] Text [again]",
    ),
)
def test_parse_cesn_rejects_malformed_notation(text: str) -> None:
    """Reject malformed note-list delimiters and event fields."""
    with pytest.raises(CESNSyntaxError):
        parse_cesn(text)


def test_parse_cesn_reports_missing_note_events_as_alignment_error() -> None:
    """Report lyric syllables left without note events."""
    with pytest.raises(CESNAlignmentError) as error:
        parse_cesn("[c3:4] Text to sing")

    assert error.value.note_count == 1
    assert error.value.syllable_count == 3


def test_parse_cesn_reports_unmatched_note_events() -> None:
    """Report the first note event that has no remaining lyric syllable."""
    with pytest.raises(CESNMatchError) as error:
        parse_cesn("[c3:4,d3:8] hmm")

    assert error.value.note_index == 1
    assert error.value.note_event == "d3:8"
    assert error.value.syllable_count == 1
