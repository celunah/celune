# SPDX-License-Identifier: Apache-2.0
"""Tests for CESN directives, sections, syllables, and notation rendering."""

from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace

import pytest

from celune.cesn import (
    CESNNoteEvent,
    CESNNotePair,
    CESNRestEvent,
    CESNSinger,
    CESNTuplet,
    _resolve_singer,
    parse_cesn,
    render_cesn,
)
from celune import cevoice
from celune.exceptions import (
    CESNAlignmentError,
    CESNMatchError,
    CESNNotationWarning,
    CESNSyntaxError,
    InvalidCharacterError,
)


def test_parse_cesn_returns_typed_note_syllable_pairs() -> None:
    """Parse dotted durations and force the requested syllabification language."""
    result = parse_cesn("[lang:en][c3:4d,d3:8,e3:16] Text to sing.")

    assert result[0].blocks[0].pairs == (
        CESNNotePair(CESNNoteEvent("c3", 4, True), "Text "),
        CESNNotePair(CESNNoteEvent("d3", 8, False), "to "),
        CESNNotePair(CESNNoteEvent("e3", 16, False), "sing."),
    )


def test_parse_cesn_splits_syllables_with_syllabreak() -> None:
    """Use Syllabreak boundaries while preserving punctuation and separators."""
    result = parse_cesn(
        "[lang:en][c3:1,d3:2,e3:4,f3:8,g3:16,a3:32,b3:2] "
        "Moonlight resting, silver-blue."
    )

    assert tuple(pair.syllable for pair in result[0].blocks[0].pairs) == (
        "Moon",
        "light ",
        "re",
        "sting, ",
        "sil",
        "ver-",
        "blue.",
    )


def test_parse_cesn_syllabifies_non_english_with_selected_language() -> None:
    """Pass the explicit ISO language code through to Syllabreak."""
    result = parse_cesn("[lang:rus][c3:4,d3:4] здраво")

    assert tuple(pair.syllable for pair in result[0].blocks[0].pairs) == (
        "здра",
        "во",
    )


def test_parse_cesn_aligns_default_singing_reference() -> None:
    """Align the default pack's singing transcript using explicit English."""
    note_list = ",".join("c3:4" for _ in range(14))
    result = parse_cesn(
        "[lang:en]"
        f"[{note_list}] Moonlight resting, silver-blue, I'll keep one quiet light for you."
    )

    assert tuple(pair.syllable for pair in result[0].blocks[0].pairs) == (
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


def test_parse_default_cechar_singing_notation() -> None:
    """Parse the default pack's CESN score and retain its tuplet durations."""
    bundle = cevoice.CEVoice.open(Path("voices/default.cevoice"))
    notation = bundle.voices["sing"]["cesn"]
    assert isinstance(notation, str)

    with pytest.warns(CESNNotationWarning) as captured_warnings:
        sections = parse_cesn(notation)
    warning_messages = tuple(str(item.message) for item in captured_warnings)

    events = tuple(
        event
        for section in sections
        for block in section.blocks
        for event in block.events
    )
    duration = sum(
        (
            event.duration_beats
            if isinstance(event, CESNRestEvent)
            else event.event.duration_beats
            for event in events
        ),
        Fraction(0),
    )

    assert duration == Fraction(37, 2)
    assert sections[0].bpm == 108
    assert duration * 60 / sections[0].bpm == Fraction(185, 18)
    assert len(warning_messages) == 2
    assert "note a3:4" in warning_messages[0]
    assert "note c4:4" in warning_messages[1]


def test_parse_cesn_respects_per_word_pattern_override() -> None:
    """Use an explicit pattern for the next word without a fixed dictionary."""
    result = parse_cesn("[lang:en][pattern:qui,et][c3:4,d3:4] Quiet.")

    assert tuple(pair.syllable for pair in result[0].blocks[0].pairs) == (
        "Qui",
        "et.",
    )


def test_parse_cesn_keeps_rests_out_of_note_syllable_pairs() -> None:
    """Keep rest timing in event order without consuming lyric syllables."""
    result = parse_cesn("[lang:en][c3:4,r:1d] Word.")
    block = result[0].blocks[0]

    assert block.events == (
        CESNNotePair(CESNNoteEvent("c3", 4, False), "Word."),
        CESNRestEvent(1, True),
    )
    assert len(block.pairs) == 1


def test_parse_cesn_scales_next_tuplet_events_then_expires() -> None:
    """Scale the next actual note events by the tuplet's exact duration ratio."""
    result = parse_cesn("[lang:en][tuplet:3,2][c3:4]One[d3:4]two[e3:4]three[f3:4]four.")
    pairs = result[0].blocks
    events = tuple(block.pairs[0].event for block in pairs)

    assert tuple(event.tuplet for event in events) == (
        CESNTuplet(3, 2),
        CESNTuplet(3, 2),
        CESNTuplet(3, 2),
        None,
    )
    assert tuple(event.tuplet_start for event in events) == (True, False, False, False)
    assert tuple(event.duration_beats for event in events) == (
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(1, 1),
    )


def test_parse_cesn_counts_rests_in_tuplets_and_renders_groups() -> None:
    """Count rests as tuplet events and preserve repeated groups when rendered."""
    notation = (
        "[lang:en][tuplet:3,2][c3:4,r:4,d3:4]One two."
        "[tuplet:3,2][f3:4,g3:4,a3:4]three four five."
    )
    sections = parse_cesn(notation)
    rendered = render_cesn(sections)
    reparsed = parse_cesn(rendered)
    events = tuple(
        item.event if isinstance(item, CESNNotePair) else item
        for block in reparsed[0].blocks
        for item in block.events
    )

    assert sum(event.tuplet_start for event in events) == 2
    assert tuple(event.duration_beats for event in events) == (
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
    )


def test_parse_cesn_rejects_incomplete_tuplets_and_section_changes() -> None:
    """Require every tuplet to finish before the notation or section ends."""
    with pytest.raises(CESNSyntaxError, match="missing 2 event"):
        parse_cesn("[lang:en][tuplet:3,2][c3:4]One.")

    with pytest.raises(CESNSyntaxError, match="section boundary"):
        parse_cesn(
            "[lang:en][tuplet:3,2][c3:4]One[bpm:140,key:c,scale:major][d3:4]Two."
        )


def test_parse_cesn_ignores_comments_and_accepts_multiple_note_blocks() -> None:
    """Ignore comment tags and retain successive note/lyric blocks in a section."""
    result = parse_cesn("[lang:en][#intro][c3:4]Word. [#ignored][d3:4]Quiet.")

    assert len(result) == 1
    assert tuple(block.lyrics for block in result[0].blocks) == (
        "Word. ",
        "Quiet.",
    )


def test_parse_cesn_starts_sections_when_musical_state_changes() -> None:
    """Split inference sections at tempo, key, scale, and language changes."""
    result = parse_cesn(
        "[lang:en][bpm:120,key:c,scale:major][c3:4]Word."
        "[bpm:160,key:e,scale:minor][e3:4]Quiet."
        "[lang:de][g3:4]Wort."
    )

    assert tuple(
        (section.bpm, section.key, section.scale, section.language)
        for section in result
    ) == (
        (120, "c", "major", "en"),
        (160, "e", "minor", "en"),
        (160, "e", "minor", "de"),
    )


def test_parse_cesn_applies_directives_between_note_tag_and_lyrics() -> None:
    """Allow musical and language tags anywhere before the next lyric text."""
    result = parse_cesn("[d3:4][lang:en][bpm:140,key:d,scale:major]Word.")

    assert (result[0].language, result[0].bpm, result[0].key) == (
        "en",
        140,
        "d",
    )


def test_parse_cesn_warns_on_out_of_scale_notes() -> None:
    """Report a pitch that is not in the current key and scale."""
    with pytest.warns(CESNNotationWarning, match="outside c major"):
        parse_cesn("[lang:en][bpm:120,key:c,scale:major][c#3:4]Word.")


def test_parse_cesn_rejects_language_missing_from_syllabreak_or_model() -> None:
    """Require an explicit language supported by both notation layers."""
    with pytest.raises(CESNSyntaxError, match="language must be specified"):
        parse_cesn("[c3:4] Word.")
    with pytest.raises(CESNSyntaxError, match="selected model"):
        parse_cesn(
            "[lang:en][c3:4] Word.",
            supported_languages=("fr",),
        )
    with pytest.raises(CESNSyntaxError, match="syllabreak"):
        parse_cesn("[lang:ja][c3:4] ことば")


def test_parse_cesn_resolves_singing_character_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach the resolved CECHAR v5 pack identity to its section."""
    singer = CESNSinger("Celune", Path("celune.cevoice"))
    monkeypatch.setattr("celune.cesn._resolve_singer", lambda name: singer)

    result = parse_cesn("[lang:en][singer:celune][c3:4]Word.")

    assert result[0].singer == singer


def test_parse_cesn_propagates_invalid_singing_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a character selection that has no compatible singing voice."""

    def reject_singer(name: str) -> CESNSinger:
        raise InvalidCharacterError(name)

    monkeypatch.setattr("celune.cesn._resolve_singer", reject_singer)
    with pytest.raises(InvalidCharacterError):
        parse_cesn("[lang:en][singer:classic][c3:4]Word.")


def test_resolve_singer_requires_a_v5_pack_with_singing_voice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve character names only to compatible singing packs."""
    pack_path = tmp_path / "celune.cevoice"
    pack_path.touch()
    bundle = SimpleNamespace(
        metadata={"version": cevoice.V5_VERSION},
        singing_voice="sing",
    )
    monkeypatch.setattr(cevoice, "bundled_voices_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cevoice.CEVoice,
        "open",
        classmethod(lambda _cls, _path: bundle),
    )
    monkeypatch.setattr(cevoice, "bundle_character_name", lambda _bundle: "Celune")

    assert _resolve_singer("celune") == CESNSinger("Celune", pack_path.resolve())

    bundle.singing_voice = None
    with pytest.raises(InvalidCharacterError):
        _resolve_singer("Celune")


def test_render_cesn_preserves_sections_patterns_and_rests() -> None:
    """Render one canonical notation string without losing section boundaries."""
    sections = parse_cesn(
        "[lang:en][bpm:120,key:c,scale:major]"
        "[pattern:qui,et][c3:4,r:1,d3:4]Quiet."
        "[bpm:160,key:e,scale:minor][e3:4]Word."
    )

    rendered = render_cesn(sections)
    reparsed = parse_cesn(rendered)

    assert len(reparsed) == 2
    assert tuple(
        event for block in reparsed[0].blocks for event in block.events
    ) == tuple(event for block in sections[0].blocks for event in block.events)


@pytest.mark.parametrize(
    "text",
    (
        "[c3:3] Word.",
        "[lang:en][c3:3] Word.",
        "[lang:en][c3:4,,d3:8] Word.",
        "[lang:en][c3:4] Word. [again]",
        "[lang:en][pattern:qui,far][c3:4,d3:4] Quiet.",
        "[lang:en][bpm:0,key:c,scale:major][c3:4] Word.",
        "[lang:en][bpm:120,key:c,scale:unknown][c3:4] Word.",
        "[lang:en][tuplet:1,1][c3:4] Word.",
        "[lang:en][tuplet:3,0][c3:4] Word.",
        "[lang:en][tuplet:3,x][c3:4] Word.",
    ),
)
def test_parse_cesn_rejects_malformed_notation(text: str) -> None:
    """Reject malformed directives, note tags, and syllable patterns."""
    with pytest.raises(CESNSyntaxError):
        parse_cesn(text)


def test_parse_cesn_reports_missing_note_events_as_alignment_error() -> None:
    """Report lyric syllables left without pitched note events."""
    with pytest.raises(CESNAlignmentError) as error:
        parse_cesn("[lang:en][c3:4] Text to sing")

    assert error.value.note_count == 1
    assert error.value.syllable_count == 3


def test_parse_cesn_reports_unmatched_note_events() -> None:
    """Report the first pitched event that has no lyric syllable."""
    with pytest.raises(CESNMatchError) as error:
        parse_cesn("[lang:en][c3:4,d3:8] hmm")

    assert error.value.note_index == 1
    assert error.value.note_event == "d3:8"
    assert error.value.syllable_count == 1


def test_parse_cesn_reports_note_without_lyrics_as_match_error() -> None:
    """Report a pitched note that is not followed by any lyric syllable."""
    with pytest.raises(CESNMatchError) as error:
        parse_cesn("[lang:en][c3:4]")

    assert error.value.note_index == 0
    assert error.value.syllable_count == 0
