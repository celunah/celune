# SPDX-License-Identifier: Apache-2.0
"""Tests for CESN directives, IPA alignment, sections, and rendering."""

from types import SimpleNamespace
from pathlib import Path
from fractions import Fraction

import pytest

from celune import cevoice
from celune.cesn import (
    CESNSinger,
    CESNTuplet,
    CESNVibrato,
    CESNNotePair,
    CESNNoteEvent,
    CESNRestEvent,
    parse_cesn,
    render_cesn,
    resolve_singer,
    format_cesn_error,
)
from celune.exceptions import (
    CESNMatchError,
    CESNSyntaxError,
    CESNAlignmentError,
    CESNNotationWarning,
    CESNValidationError,
    InvalidCharacterError,
    CESNBackendFormatError,
    CESNUnsupportedFeatureError,
)


def test_parse_cesn_returns_typed_note_ipa_pairs() -> None:
    """Align each explicit IPA syllable with one pitched note."""
    result = parse_cesn("[lang:en][c3:4d,d3:8,e3:16][tɛkst.tu.sɪŋ] Text to sing.")

    assert result[0].blocks[0].pairs == (
        CESNNotePair(CESNNoteEvent("c3", 4, True), "tɛkst"),
        CESNNotePair(CESNNoteEvent("d3", 8, False), "tu"),
        CESNNotePair(CESNNoteEvent("e3", 16, False), "sɪŋ"),
    )
    assert result[0].blocks[0].lyrics == "Text to sing."


def test_joined_notes_require_equal_pitch_and_render_round_trip() -> None:
    """Retain same-pitch ties and locate invalid joins at their marker."""
    notation = "[lang:en][c4:1,j,c4:1][ɑ.ɑ]Ah"

    sections = parse_cesn(notation)
    pairs = sections[0].blocks[0].pairs

    assert tuple(pair.event.tie_from_previous for pair in pairs) == (False, True)
    rendered = render_cesn(sections)
    assert "[c4:1,j,c4:1]" in rendered
    assert parse_cesn(rendered) == sections

    invalid_notation = "[lang:en][c4:1,j,d4:1][ɑ.ɑ]Ah"
    with pytest.raises(
        CESNValidationError,
        match="can't join different pitches together",
    ) as caught:
        parse_cesn(invalid_notation)

    assert caught.value.source_start == invalid_notation.index("j")
    assert format_cesn_error(invalid_notation, caught.value).endswith(
        "can't join different pitches together"
    )


@pytest.mark.parametrize(
    "events",
    (
        "j,c4:1",
        "c4:1,j",
        "c4:1,j,r:1",
    ),
)
def test_join_marker_requires_pitched_notes_on_both_sides(events: str) -> None:
    """Reject leading, trailing, and rest-connected join markers."""
    with pytest.raises(
        CESNValidationError,
        match="can't join different pitches together",
    ):
        parse_cesn(f"[lang:en][{events}][ɑ]Ah")


def test_vibrato_targets_only_the_next_note_and_renders_round_trip() -> None:
    """Attach each vibrato tag to one following note in a grouped melody."""
    notation = "[lang:en][v:4,5,20] [c4:4,d4:4,e4:4,f4:4][ˈsɛl.oʊ.ɪŋ.ɪŋ] singing"

    sections = parse_cesn(notation)
    pairs = sections[0].blocks[0].pairs

    assert tuple(pair.event.vibrato for pair in pairs) == (
        CESNVibrato(4, 5.0, 20),
        None,
        None,
        None,
    )
    rendered = render_cesn(sections)
    assert "[v:4,5,20][c4:4][d4:4][e4:4][f4:4]" in rendered
    assert parse_cesn(rendered) == sections


def test_zero_delay_vibrato_starts_immediately_and_renders_round_trip() -> None:
    """Preserve zero as an immediate vibrato delay through parsing and rendering."""
    sections = parse_cesn("[lang:en][v:0,5,40][c4:1][ɑ]Ah")
    vibrato = sections[0].blocks[0].pairs[0].event.vibrato

    assert vibrato == CESNVibrato(0, 5.0, 40)
    assert vibrato is not None
    assert vibrato.delay_beats == 0
    assert parse_cesn(render_cesn(sections)) == sections


def test_vibrato_tag_between_note_groups_targets_the_next_note() -> None:
    """Keep a vibrato tag after one note queued for the next pitched event."""
    sections = parse_cesn(
        "[lang:en][c4:4][d4:4][v:8,5.5,40][e4:4,f4:4][ˈsɛl.oʊ.ɪŋ.ɪŋ] singing"
    )
    pairs = sections[0].blocks[0].pairs

    assert tuple(pair.event.vibrato for pair in pairs) == (
        None,
        None,
        CESNVibrato(8, 5.5, 40),
        None,
    )
    assert parse_cesn(render_cesn(sections)) == sections


@pytest.mark.parametrize(
    "tag",
    (
        "[v:3,5,20]",
        "[v:4,-5,20]",
        "[v:4,nan,20]",
        "[v:4,5,10]",
        "[v:4,5,100]",
        "[v:4,5]",
        "[v:4,5,20,40]",
    ),
)
def test_parse_cesn_rejects_invalid_vibrato_tags(tag: str) -> None:
    """Require a supported delay note value, positive speed, and depth."""
    with pytest.raises(CESNValidationError):
        parse_cesn(f"[lang:en]{tag}[c4:1][ɑ]Ah")


def test_parse_cesn_rejects_vibrato_without_a_following_pitched_note() -> None:
    """Report a vibrato directive that reaches the end without a target note."""
    with pytest.raises(CESNValidationError, match="must precede a pitched note"):
        parse_cesn("[lang:en][c4:1][ɑ]Ah[v:4,5,20]")


def test_parse_cesn_converts_xsampa_input_to_canonical_ipa() -> None:
    """Convert explicitly marked ASCII X-SAMPA before note alignment."""
    notation = '[lang:en][c4:4,d4:4][x-sampa:"tSEl.oU]cello'
    sections = parse_cesn(notation)

    assert tuple(pair.ipa for pair in sections[0].blocks[0].pairs) == (
        "ˈtʃɛl",
        "oʊ",
    )
    assert "[ˈtʃɛl.oʊ]cello" in render_cesn(sections)


def test_parse_cesn_expands_slide_into_one_onset_and_a_slurred_vowel() -> None:
    """Use one consonant onset and continue its vowel across the note group."""
    notation = "[lang:en][slide:lA,c4:1,d4:1,e4:1,f4:1]la"

    sections = parse_cesn(notation)
    pairs = sections[0].blocks[0].pairs

    assert tuple(pair.ipa for pair in pairs) == ("lɑ", "ɑ", "ɑ", "ɑ")
    assert tuple(pair.slur_from_previous for pair in pairs) == (
        False,
        True,
        True,
        True,
    )
    rendered = render_cesn(sections)
    assert "[slur:4][c4:1,d4:1,e4:1,f4:1][lɑ.ɑ.ɑ.ɑ]la" in rendered
    assert parse_cesn(rendered) == sections

    vowel_only = parse_cesn("[lang:en][slide:A,c4:1,d4:1]ah")
    assert tuple(pair.ipa for pair in vowel_only[0].blocks[0].pairs) == ("ɑ", "ɑ")


@pytest.mark.parametrize(
    "slide",
    (
        "lA,c4:1",
        "lAk,c4:1,d4:1",
        "lA,c4:1,r:4,d4:1",
        "l,c4:1,d4:1",
        "lA.A,c4:1,d4:1",
    ),
)
def test_parse_cesn_rejects_invalid_slide_shorthand(slide: str) -> None:
    """Require one final vowel nucleus and at least two pitched slide notes."""
    with pytest.raises(CESNValidationError):
        parse_cesn(f"[lang:en][slide:{slide}]la")


@pytest.mark.parametrize(
    ("pronunciation", "message"),
    (
        ("ˈtSEl.oU", "ASCII characters only"),
        ("/tSEl.oU/", "without slash delimiters"),
        ("tSEl$", "unsupported X-SAMPA symbol"),
    ),
)
def test_parse_cesn_rejects_invalid_xsampa(
    pronunciation: str,
    message: str,
) -> None:
    """Reject non-ASCII, delimited, or unknown X-SAMPA transcription."""
    notation = f"[lang:en][c4:4][x-sampa:{pronunciation}]cello"
    with pytest.raises(CESNValidationError, match=message) as caught:
        parse_cesn(notation)
    assert caught.value.source_start == notation.index("[x-sampa:")
    assert caught.value.source_end == notation.index("]", caught.value.source_start) + 1


def test_parse_cesn_preserves_lyrics_without_syllabifying_them() -> None:
    """Keep whole lyric spelling while IPA provides note-aligned syllables."""
    result = parse_cesn("[lang:en][c3:4,d3:4,e3:4][ˈsɪl.vɚ.bluː] silver-blue.")

    assert result[0].blocks[0].lyrics == "silver-blue."
    assert tuple(pair.ipa for pair in result[0].blocks[0].pairs) == (
        "ˈsɪl",
        "vɚ",
        "bluː",
    )


def test_parse_cesn_adds_punctuation_rests_and_preserves_explicit_rests() -> None:
    """Append phrase rests after punctuation without duplicating explicit rests."""
    period_sections = parse_cesn("[lang:en][c3:4][a]Finished.")
    comma_sections = parse_cesn("[lang:en][c3:4][a]Continue,")
    explicit_sections = parse_cesn("[lang:en][c3:4][a]Finished.[r:2]")

    assert period_sections[0].blocks[-1].events == (CESNRestEvent(2, False),)
    assert comma_sections[0].blocks[-1].events == (CESNRestEvent(4, False),)
    assert tuple(
        event
        for block in explicit_sections[0].blocks
        for event in block.events
        if isinstance(event, CESNRestEvent)
    ) == (CESNRestEvent(2, False),)
    rendered = render_cesn(period_sections)
    assert "Finished.[r:2]" in rendered
    assert render_cesn(parse_cesn(rendered)) == rendered


def test_parse_cesn_aligns_explicit_ipa_for_a_lyric_phrase() -> None:
    """Require IPA boundaries for each note without changing phrase text."""
    note_list = ",".join("c3:4" for _ in range(14))
    ipa = ".".join("a" for _ in range(14))
    lyrics = "Moonlight resting, silver-blue, I'll keep one quiet light for you."
    result = parse_cesn(f"[lang:en][{note_list}][{ipa}] {lyrics}")

    assert result[0].blocks[0].lyrics == lyrics
    assert len(result[0].blocks[0].pairs) == 14
    assert tuple(pair.ipa for pair in result[0].blocks[0].pairs) == tuple(
        "a" for _ in range(14)
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

    assert duration == Fraction(45, 2)
    assert sections[0].bpm == 108
    assert duration * 60 / sections[0].bpm == Fraction(25, 2)
    assert len(warning_messages) == 2
    assert "note a3:4" in warning_messages[0]
    assert "note c4:4" in warning_messages[1]


def test_parse_cesn_aligns_explicit_ipa_syllables_to_notes() -> None:
    """Use raw IPA after the note list and preserve the unsplit lyric word."""
    result = parse_cesn("[lang:en][c3:4,d3:4][ˈkwaɪ.ət] Quiet.")

    assert tuple(pair.ipa for pair in result[0].blocks[0].pairs) == (
        "ˈkwaɪ",
        "ət",
    )
    assert result[0].blocks[0].lyrics == "Quiet."


def test_parse_cesn_keeps_rests_out_of_note_syllable_pairs() -> None:
    """Keep rest timing in event order without consuming lyric syllables."""
    result = parse_cesn("[lang:en][c3:4,r:1d][wɜːd] Word.")
    block = result[0].blocks[0]

    assert block.events == (
        CESNNotePair(CESNNoteEvent("c3", 4, False), "wɜːd"),
        CESNRestEvent(1, True),
    )
    assert len(block.pairs) == 1


def test_parse_cesn_scales_next_tuplet_events_then_expires() -> None:
    """Scale the next actual note events by the tuplet's exact duration ratio."""
    result = parse_cesn(
        "[lang:en][tuplet:3,2][c3:4][wʌn]One"
        "[d3:4][tu]two[e3:4][θɹi]three[f3:4][fɔɹ]four."
    )
    blocks = tuple(block for block in result[0].blocks if block.pairs)
    events = tuple(block.pairs[0].event for block in blocks)

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
        "[lang:en][tuplet:3,2][c3:4,r:4,d3:4][wʌn.tu]One two."
        "[tuplet:3,2][f3:4,g3:4,a3:4][θɹi.fɔɹ.faɪv]three four five."
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
    assert tuple(event.duration_beats for event in events if event.tuplet) == (
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
        Fraction(2, 3),
    )
    assert tuple(
        event.duration_beats
        for event in events
        if isinstance(event, CESNRestEvent) and event.tuplet is None
    ) == (Fraction(2), Fraction(2))


def test_parse_cesn_rejects_incomplete_tuplets_and_section_changes() -> None:
    """Require every tuplet to finish before the notation or section ends."""
    with pytest.raises(CESNSyntaxError, match="missing 1 event"):
        parse_cesn("[lang:en][tuplet:3,2][c3:4][wʌn]One.")

    with pytest.raises(CESNSyntaxError, match="section boundary"):
        parse_cesn(
            "[lang:en][tuplet:3,2][c3:4][wʌn]One"
            "[bpm:140,key:c,scale:major][d3:4][tu]Two."
        )


def test_parse_cesn_ignores_comments_and_accepts_multiple_note_blocks() -> None:
    """Ignore comment tags and retain successive note/lyric blocks in a section."""
    result = parse_cesn(
        "[lang:en][#intro][c3:4][wɜːd]Word. [#ignored][d3:4][ɡɹeɪt]Great."
    )

    assert len(result) == 1
    assert tuple(block.lyrics for block in result[0].blocks if block.lyrics) == (
        "Word. ",
        "Great.",
    )
    assert tuple(
        event
        for block in result[0].blocks
        for event in block.events
        if isinstance(event, CESNRestEvent)
    ) == (CESNRestEvent(2, False), CESNRestEvent(2, False))


def test_parse_cesn_starts_sections_when_musical_state_changes() -> None:
    """Split inference sections at tempo, key, scale, and language changes."""
    result = parse_cesn(
        "[lang:en][bpm:120,key:c,scale:major][c3:4][wɜːd]Word."
        "[bpm:160,key:e,scale:minor][e3:4][kwaɪt]Quiet."
        "[lang:de][g3:4][vɔɹt]Wort."
    )

    assert tuple(
        (section.bpm, section.key, section.scale, section.language)
        for section in result
    ) == (
        (120, "c", "major", "en"),
        (160, "e", "minor", "en"),
        (160, "e", "minor", "de"),
    )


def test_parse_cesn_requires_settings_before_note_list() -> None:
    """Require extra tags before notes and IPA immediately after the notes."""
    with pytest.raises(CESNValidationError, match="must precede the note list"):
        parse_cesn("[d3:4][lang:en][bpm:140,key:d,scale:major][wɜːd]Word.")


def test_parse_cesn_warns_on_out_of_scale_notes() -> None:
    """Report a pitch that is not in the current key and scale."""
    with pytest.warns(CESNNotationWarning, match="outside c major"):
        parse_cesn("[lang:en][bpm:120,key:c,scale:major][c#3:4][wɜːd]Word.")


def test_parse_cesn_checks_only_explicit_language_and_model_support() -> None:
    """Require a language tag and check it against the selected model."""
    with pytest.raises(CESNValidationError, match="language must be specified"):
        parse_cesn("[c3:4][wɜːd] Word.")
    with pytest.raises(CESNUnsupportedFeatureError, match="selected model"):
        parse_cesn(
            "[lang:en][c3:4][wɜːd] Word.",
            supported_languages=("fr",),
        )
    parse_cesn("[lang:ja][c3:4,d3:4,e3:4][ko.to.ba] ことば")
    text = "[lang:en][bpm:120,key:c,scale:supermajor][c3:4][wɜːd] Word."
    with pytest.raises(
        CESNUnsupportedFeatureError,
        match="unsupported CESN scale",
    ) as error:
        parse_cesn(text)

    assert error.value.source_start == text.index("[bpm:")


def test_parse_cesn_classifies_malformed_values_as_validation_errors() -> None:
    """Represent malformed CESN values with the validation subtype."""
    with pytest.raises(CESNValidationError, match="BPM must be a positive integer"):
        parse_cesn("[lang:en][bpm:0,key:c,scale:major][c3:4][wɜːd] Word.")


def test_parse_cesn_resolves_singing_character_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach the resolved CECHAR v5 pack identity to its section."""
    singer = CESNSinger("Celune", Path("celune.cevoice"))
    monkeypatch.setattr("celune.cesn.resolve_singer", lambda name: singer)

    result = parse_cesn("[lang:en][singer:celune][c3:4][wɜːd]Word.")

    assert result[0].singer == singer


def test_parse_cesn_classifies_missing_singer_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locate an unavailable explicit singer as an unsupported CESN feature."""

    def reject_singer(name: str) -> CESNSinger:
        raise InvalidCharacterError(name)

    monkeypatch.setattr("celune.cesn.resolve_singer", reject_singer)
    with pytest.raises(CESNUnsupportedFeatureError) as caught:
        parse_cesn("[lang:en][singer:classic][c3:4][wɜːd]Word.")

    assert isinstance(caught.value.__cause__, InvalidCharacterError)
    assert caught.value.source_start == 9


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
        singing_pitch_range=(0, 127),
    )
    monkeypatch.setattr(cevoice, "bundled_voices_dir", lambda: tmp_path)

    active_pack = tmp_path / "active.cevoice"
    monkeypatch.setattr(cevoice, "active_bundle_path", lambda: active_pack)

    def open_bundle(_cls: type[cevoice.CEVoice], path: Path) -> object:
        if path == active_pack:
            raise OSError("active bundle is not part of this test")
        return bundle

    monkeypatch.setattr(
        cevoice.CEVoice,
        "open",
        classmethod(open_bundle),
    )
    monkeypatch.setattr(cevoice, "bundle_character_name", lambda _bundle: "Celune")

    assert resolve_singer("celune") == CESNSinger("Celune", pack_path.resolve())

    bundle.singing_voice = None
    with pytest.raises(InvalidCharacterError) as error:
        resolve_singer("Celune")
    assert str(error.value) == (
        "Character 'Celune' is missing or has no usable CECHAR v5 singing voice"
    )


def test_render_cesn_preserves_sections_ipa_and_rests() -> None:
    """Render one canonical notation string without losing section boundaries."""
    sections = parse_cesn(
        "[lang:en][bpm:120,key:c,scale:major]"
        "[c3:4,r:1,d3:4][ˈkwaɪ.ət]Quiet."
        "[bpm:160,key:e,scale:minor][e3:4][wɜːd]Word."
    )

    rendered = render_cesn(sections)
    reparsed = parse_cesn(rendered)

    assert len(reparsed) == 2
    assert tuple(
        event for block in reparsed[0].blocks for event in block.events
    ) == tuple(event for block in sections[0].blocks for event in block.events)


def test_parse_and_render_cesn_preserves_slurred_ipa_and_note_timing() -> None:
    """Keep aligned IPA syllables and durations while marking a slur group."""
    sections = parse_cesn("[lang:en][slur:2][c4:4,d4:8][ˈsɛl.oʊ]cello.")

    pairs = sections[0].blocks[0].pairs
    assert tuple(pair.slur_from_previous for pair in pairs) == (False, True)
    assert tuple(pair.event.duration_beats for pair in pairs) == (
        Fraction(1),
        Fraction(1, 2),
    )

    rendered = render_cesn(sections)
    reparsed = parse_cesn(rendered)

    assert "[slur:2]" in rendered
    assert "[c4:4,d4:8][ˈsɛl.oʊ]" in rendered
    assert reparsed[0].blocks[0].pairs == pairs


@pytest.mark.parametrize(
    "text",
    (
        "[c3:3] Word.",
        "[lang:en][c3:3] Word.",
        "[lang:en][c3:4,,d3:8] Word.",
        "[lang:en][c3:4][wɜːd] Word. [again]",
        "[lang:en][pattern:qui,far][c3:4,d3:4] Quiet.",
        "[lang:en][ipa:/ˈkwaɪ.ət/][c3:4,d3:4] Quiet.",
        "[lang:en][c3:4,d3:4][ˈkwaɪ] Quiet.",
        "[lang:en][c3:4,d3:4][/ˈkwaɪ.ət/] Quiet.",
        "[lang:en][c3:4,d3:4][ˈkwaɪ..ət] Quiet.",
        "[lang:en][slur:2][c3:4,d3:4][ˈsɛl] Cello.",
        "[lang:en][slur:3][c3:4,d3:4][ˈsɛl.oʊ] Cello.",
        "[lang:en][slur:2][c3:4,r:4,d3:4][ˈsɛl.oʊ] Cello.",
        "[lang:en][slur:1][c3:4][sɛl] Cell.",
        "[lang:en][bpm:0,key:c,scale:major][c3:4][wɜːd] Word.",
        "[lang:en][bpm:120,key:c,scale:unknown][c3:4][wɜːd] Word.",
        "[lang:en][tuplet:1,1][c3:4][wɜːd] Word.",
        "[lang:en][tuplet:3,0][c3:4][wɜːd] Word.",
        "[lang:en][tuplet:3,x][c3:4][wɜːd] Word.",
    ),
)
def test_parse_cesn_rejects_malformed_notation(text: str) -> None:
    """Reject malformed directives, note tags, and IPA syllable groups."""
    with pytest.raises(CESNSyntaxError):
        parse_cesn(text)


def test_parse_cesn_error_tracks_source_span_and_formats_one_line() -> None:
    """Show a malformed directive with its exact line and one-based column."""
    text = "[lang:en]\n[bpm:120:key:c,scale:major]\n[c4:4] word.\n[d4:4] another line."

    with pytest.raises(CESNSyntaxError) as caught:
        parse_cesn(text)

    error = caught.value
    assert error.source_start == text.index("[bpm:")
    assert error.source_end == text.index("]", error.source_start) + 1
    malformed_tag = text[error.source_start : error.source_end]
    assert format_cesn_error(text, error) == (
        "Parse error in CESN notation (line 2, column 1):\n\n"
        "[bpm:120:key:c,scale:major]\n"
        f"{'^' + '~' * (len(malformed_tag) - 1)} "
        "CESN settings require bpm, key, and scale fields"
    )


def test_format_cesn_error_without_source_span_returns_exception_message() -> None:
    """Do not invent a source location for backend-only CESN failures."""
    error = CESNBackendFormatError("prepared score has an invalid header")

    assert format_cesn_error("plain text", error) == (
        "prepared score has an invalid header"
    )


def test_parse_cesn_reports_missing_note_events_as_alignment_error() -> None:
    """Report IPA syllables left without pitched note events."""
    notation = "[lang:en][c3:4][tɛkst.tu.sɪŋ] Text to sing"
    with pytest.raises(CESNAlignmentError) as error:
        parse_cesn(notation)

    assert error.value.note_count == 1
    assert error.value.syllable_count == 3
    assert error.value.source_start == notation.index("[tɛkst")


def test_parse_cesn_reports_unmatched_note_events() -> None:
    """Report the first pitched event that has no corresponding IPA syllable."""
    with pytest.raises(CESNMatchError) as error:
        parse_cesn("[lang:en][c3:4,d3:8][hɪm] hmm")

    assert error.value.note_index == 1
    assert error.value.note_event == "d3:8"
    assert error.value.syllable_count == 1


def test_parse_cesn_reports_note_without_lyrics_as_match_error() -> None:
    """Report a pitched note group that has no following IPA or lyric text."""
    with pytest.raises(CESNMatchError) as error:
        parse_cesn("[lang:en][c3:4]")

    assert error.value.note_index == 0
    assert error.value.syllable_count == 0
