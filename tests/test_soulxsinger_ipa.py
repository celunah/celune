# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit CESN IPA syllable alignment."""

import json
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from typing import Optional, cast

import pytest
import numpy as np

from celune.exceptions import (
    CESNIPAError,
    CESNRangeError,
    CESNValidationError,
    CESNBackendFormatError,
)
from celune.cesn import CESNSinger, format_cesn_error
from celune.typing.common import JSON
from celune.cevoice import CEVoiceLoader
from celune.backends.tts.soulxsinger import (
    SoulXSinger,
    decode_score,
    _SoulXRuntime,
    pitch_to_midi,
)


@pytest.mark.parametrize(
    ("ipa", "expected_phonemes", "expected_character"),
    (
        ("ˈkwaɪ.ət", "en_K-W-AY1 en_AH0-T", None),
        ("ˈkwaɪ.ɲ", None, "ɲ"),
    ),
)
@pytest.mark.parametrize(
    ("high_precision", "expected_control"),
    ((False, "score"), (True, "melody")),
)
def test_generate_stream_uses_or_rejects_explicit_ipa(
    ipa: str,
    expected_phonemes: Optional[str],
    expected_character: Optional[str],
    high_precision: bool,
    expected_control: str,
) -> None:
    """Convert each explicit IPA syllable and reject unsupported phone symbols."""
    backend = object.__new__(SoulXSinger)
    backend.name = "soulxsinger"
    backend.high_precision = high_precision
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    processed_metadata: list[dict[str, object]] = []

    def process(metadata: dict[str, object], *_args: object) -> dict[str, object]:
        processed_metadata.append(dict(metadata))
        for key in ("duration", "phoneme", "note_pitch", "note_type"):
            value = metadata[key]
            assert isinstance(value, str)
            metadata[key] = value.split()
        return metadata

    processor = SimpleNamespace(
        phone2idx={
            "en_K": 0,
            "en_W": 1,
            "en_AY1": 2,
            "en_AH0": 3,
            "en_T": 4,
        },
        process=mock.Mock(side_effect=process),
        hop_size=480,
    )
    inference_model = SimpleNamespace(
        infer=mock.Mock(
            side_effect=[
                np.concatenate(
                    (
                        np.full(48000, -0.5, dtype=np.float32),
                        np.full(48000, 0.5, dtype=np.float32),
                    )
                ),
                np.concatenate(
                    (
                        np.full(48000, -0.5, dtype=np.float32),
                        np.full(48000, 0.5, dtype=np.float32),
                    )
                ),
            ]
        )
    )
    f0_extractor = SimpleNamespace(
        process=mock.Mock(return_value=np.full(200, 200.0, dtype=np.float32))
    )
    runtime = SimpleNamespace(
        model=inference_model,
        processor=processor,
        f0_extractor=f0_extractor,
        sample_rate=24000,
        inference_steps=4,
        cfg=1.5,
    )
    singer_bundle = SimpleNamespace(
        path=Path("singer.cevoice"),
        singing_pitch_range=(0, 127),
    )
    singer_loader = SimpleNamespace(bundle=singer_bundle)
    with mock.patch.object(backend, "load_model", return_value=runtime):
        loaded_runtime = backend.load_model("soulxsinger")
    notation = f"[lang:en][g3:4,a3:4][{ipa}]Quiet."

    with (
        mock.patch(
            "celune.backends.tts.soulxsinger.resolve_active_singer",
            return_value=singer,
        ),
        mock.patch(
            "celune.backends.tts.soulxsinger.bundle_character_name",
            return_value="Test Singer",
        ),
        mock.patch.object(backend, "_apply_seed"),
        mock.patch.object(
            SoulXSinger,
            "_singer_loader",
            return_value=singer_loader,
        ),
        mock.patch.object(
            SoulXSinger,
            "_validate_singing_voice",
            return_value=("[lang:en][g3:4][ɑ]Ah", Path("sing.wav")),
        ),
        mock.patch.object(SoulXSinger, "_prompt_data", return_value={}),
        mock.patch(
            "celune.backends.tts.soulxsinger.to_numpy_audio",
            side_effect=lambda audio: audio,
        ),
        mock.patch.dict(
            "sys.modules",
            {"soundfile": SimpleNamespace(write=mock.Mock())},
        ),
    ):
        if expected_character is None:
            chunks = list(backend.generate_stream(loaded_runtime, text=notation))
            assert chunks[0][0].shape == (48000,)
            assert chunks[0][0][0] == pytest.approx(0.5)
        else:
            with pytest.raises(CESNIPAError) as error:
                list(backend.generate_stream(loaded_runtime, text=notation))
            assert error.value.character == expected_character
            assert error.value.source_start == notation.index(expected_character)
            assert error.value.source_end == notation.index(expected_character) + 1
            assert str(error.value) == (
                f"SoulX-Singer cannot encode IPA character {expected_character}."
            )
            assert format_cesn_error(notation, error.value) == (
                "Parse error in CESN notation (line 1, column "
                f"{notation.index(expected_character) + 1}):\n\n"
                f"{notation}\n"
                f"{' ' * notation.index(expected_character)}^ "
                "SoulX-Singer cannot encode IPA character ɲ."
            )

    if expected_phonemes is not None:
        assert processor.process.call_count == (2 if high_precision else 1)
        target_metadata = processed_metadata[-1]
        assert target_metadata["phoneme"] == (f"<SP> {expected_phonemes} <SP>")
        assert "text" not in target_metadata
        duration_metadata = target_metadata["duration"]
        pitch_metadata = target_metadata["note_pitch"]
        note_type_metadata = target_metadata["note_type"]
        assert isinstance(duration_metadata, str)
        assert isinstance(pitch_metadata, str)
        assert isinstance(note_type_metadata, str)
        assert duration_metadata.split()[0] == "2"
        assert pitch_metadata.split()[0] == "0"
        assert note_type_metadata.split()[0] == "1"
        assert [
            call.kwargs["control"] for call in inference_model.infer.call_args_list
        ] == (["score", "melody"] if high_precision else ["score"])
        assert inference_model.infer.call_args.kwargs["control"] == expected_control
        if high_precision:
            f0_metadata = target_metadata["f0"]
            assert isinstance(f0_metadata, str)
            target_f0 = [float(value) for value in f0_metadata.split()]
            expected_frames = int(
                sum(float(value) for value in duration_metadata.split()) * 50
            )
            assert len(target_f0) == expected_frames
            assert target_f0[:100] == [0.0] * 100
            assert target_f0[100] == pytest.approx(196.0, abs=0.003)
            assert any(value == pytest.approx(220.0) for value in target_f0)
            assert target_f0[-1] == 0.0
        else:
            assert "f0" not in target_metadata


def test_generate_stream_emits_one_captioned_chunk_per_cesn_section() -> None:
    """Preserve section lyrics and audio boundaries through generation."""
    backend = object.__new__(SoulXSinger)
    backend.name = "soulxsinger"
    backend.high_precision = False
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    inference_model = SimpleNamespace(
        infer=mock.Mock(
            side_effect=(
                np.full(96000, 0.25, dtype=np.float32),
                np.full(96000, 0.5, dtype=np.float32),
            )
        )
    )
    processor = SimpleNamespace(
        phone2idx={"en_AA0": 0},
        process=mock.Mock(side_effect=lambda metadata: metadata),
        hop_size=480,
    )
    runtime = SimpleNamespace(
        model=inference_model,
        processor=processor,
        f0_extractor=None,
        sample_rate=24000,
        inference_steps=4,
        cfg=1.5,
    )
    singer_loader = SimpleNamespace(
        bundle=SimpleNamespace(
            path=Path("singer.cevoice"),
            singing_pitch_range=(0, 127),
        )
    )
    notation = "[lang:en][c4:4][ɑ]First [bpm:90,key:c,scale:major][d4:4][ɑ]Second."

    with (
        mock.patch.object(backend, "_apply_seed"),
        mock.patch(
            "celune.backends.tts.soulxsinger.resolve_active_singer",
            return_value=singer,
        ),
        mock.patch(
            "celune.backends.tts.soulxsinger.bundle_character_name",
            return_value="Test Singer",
        ),
        mock.patch.object(
            SoulXSinger,
            "_singer_loader",
            return_value=singer_loader,
        ),
        mock.patch.object(
            SoulXSinger,
            "_validate_singing_voice",
            return_value=("[lang:en][c4:4][ɑ]Ah", Path("sing.wav")),
        ),
        mock.patch.object(SoulXSinger, "_prompt_data", return_value={}),
        mock.patch(
            "celune.backends.tts.soulxsinger.to_numpy_audio",
            side_effect=lambda audio: audio,
        ),
    ):
        chunks = list(
            backend.generate_stream(cast(_SoulXRuntime, runtime), text=notation)
        )

    assert len(chunks) == 2
    timings: list[JSON] = []
    for _audio, _sample_rate, timing in chunks:
        assert timing is not None
        timings.append(timing)
    assert [timing["section_caption"] for timing in timings] == ["First", "Second."]
    assert [timing["section_final"] for timing in timings] == [True, True]
    assert [timing["is_final"] for timing in timings] == [False, True]
    assert [chunk[0].shape for chunk in chunks] == [(48000,), (48000,)]


def test_high_precision_prompt_metadata_contains_frame_aligned_f0(
    tmp_path: Path,
) -> None:
    """Prepare frame-aligned reference F0 when melody conditioning is enabled."""
    reference_bundle = tmp_path / "singer.cevoice"
    reference_bundle.write_bytes(b"voice pack")
    processor = SimpleNamespace(
        phone2idx={"en_AA0": 0},
        process=mock.Mock(return_value={"prompt": "prompt"}),
        hop_size=480,
    )
    runtime = SimpleNamespace(
        processor=processor,
        sample_rate=24000,
        prompt_cache={},
        f0_extractor=SimpleNamespace(
            process=mock.Mock(
                return_value=np.array([196.5, 0.0, 200.0], dtype=np.float32)
            )
        ),
    )
    loader = SimpleNamespace(
        bundle=SimpleNamespace(
            path=reference_bundle,
            singing_pitch_range=(0, 127),
        )
    )

    with mock.patch(
        "celune.backends.tts.soulxsinger.bundle_character_name",
        return_value="Test Singer",
    ):
        SoulXSinger._prompt_data(
            cast(_SoulXRuntime, runtime),
            cast(CEVoiceLoader, loader),
            "[lang:en][g3:4][ɑ]Ah",
            tmp_path / "reference.wav",
            high_precision=True,
        )

    prompt_metadata = processor.process.call_args.args[0]
    assert prompt_metadata["f0"] == "196.5 0 200"
    runtime.f0_extractor.process.assert_called_once_with(
        str(tmp_path / "reference.wav"),
        verbose=False,
    )


def test_slurred_cesn_keeps_distinct_syllables_as_note_events() -> None:
    """Keep a new slurred syllable's pitch, phones, and duration."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    notation = "[lang:en][slur:2][c4:4,d4:4][ˈtʃɛl.oʊ]cello."
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        score_text = backend.cesn_to_model(notation)

    section = decode_score(score_text)[0]
    runtime = cast(
        _SoulXRuntime,
        SimpleNamespace(
            processor=SimpleNamespace(
                phone2idx={
                    "en_CH": 0,
                    "en_EH1": 1,
                    "en_L": 2,
                    "en_OW0": 3,
                }
            ),
        ),
    )

    metadata = SoulXSinger._metadata(section.events, section.bpm, runtime)

    assert metadata["note_type"] == "2 2 1"
    assert metadata["duration"] == "0.5 0.5 1"
    assert metadata["note_pitch"] == "60 62 0"
    assert metadata["phoneme"] == "en_CH-EH1-L en_OW0 <SP>"
    assert tuple(event.slur_from_previous for event in section.events) == (
        False,
        True,
        False,
    )
    legacy_payload = json.loads(score_text)
    legacy_payload["sections"][0].pop("display_text")
    for raw_event in legacy_payload["sections"][0]["events"]:
        raw_event.pop("slur_from_previous")
    legacy_section = decode_score(json.dumps(legacy_payload))[0]
    assert not any(event.slur_from_previous for event in legacy_section.events)
    assert legacy_section.display_text == ""


def test_slurred_repeated_phones_use_soulx_continuation_type() -> None:
    """Use SoulX continuation only when the following event repeats the phones."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    notation = "[lang:en][slur:2][c4:4,d4:4][ˈlɑ.ˈlɑ]lala."
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        score_text = backend.cesn_to_model(notation)

    section = decode_score(score_text)[0]
    runtime = cast(
        _SoulXRuntime,
        SimpleNamespace(
            processor=SimpleNamespace(phone2idx={"en_L": 0, "en_AA1": 1}),
        ),
    )

    metadata = SoulXSinger._metadata(section.events, section.bpm, runtime)

    assert metadata["note_type"] == "2 3 1"
    assert metadata["note_pitch"] == "60 62 0"
    assert metadata["phoneme"] == "en_L-AA1 en_L-AA1 <SP>"


def test_slide_shorthand_continues_vowel_after_one_consonant_onset() -> None:
    """Continue a shared vowel after a slide's initial onset consonant."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    notation = "[lang:en][slide:lA,c4:1,d4:1,e4:1,f4:1]la"
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        score_text = backend.cesn_to_model(notation)

    section = decode_score(score_text)[0]
    runtime = cast(
        _SoulXRuntime,
        SimpleNamespace(
            processor=SimpleNamespace(phone2idx={"en_L": 0, "en_AA0": 1}),
        ),
    )

    metadata = SoulXSinger._metadata(section.events, section.bpm, runtime)

    assert metadata["note_type"] == "2 3 3 3"
    assert metadata["note_pitch"] == "60 62 64 65"
    assert metadata["phoneme"] == "en_L-AA0 en_AA0 en_AA0 en_AA0"


@pytest.mark.parametrize(
    "text",
    (
        "not prepared score JSON",
        '{"backend":"soulxsinger","version":2,"sections":[]}',
    ),
)
def test_decode_score_classifies_invalid_backend_payload(
    text: str,
) -> None:
    """Keep backend wire-format failures typed and independent of localization."""
    with pytest.raises(CESNBackendFormatError):
        decode_score(text)


@pytest.mark.parametrize(
    ("pitch", "expected_midi"),
    (("c2", 36), ("f#3", 54), ("c5", 72), ("f5", 77), ("g9", 127)),
)
def test_pitch_to_midi_converts_notes_independently_of_voice_range(
    pitch: str,
    expected_midi: int,
) -> None:
    """Convert score pitches without imposing a backend-wide singing range."""
    assert pitch_to_midi(pitch) == expected_midi


def test_cesn_preprocessing_uses_the_selected_singers_pitch_range() -> None:
    """Reject out-of-range notes with the CECHAR singing voice's named bounds."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Celune", Path("singer.cevoice"), (54, 72))
    notation = "[lang:en][bpm:120,key:f#,scale:major][f3:4][æ]Ah."
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        with pytest.raises(CESNRangeError) as error:
            backend.cesn_to_model(notation)
        payload = backend.cesn_to_model(
            "[lang:en][bpm:120,key:f#,scale:major][f#3:4][æ]Ah."
        )

    assert str(error.value) == "note F3 is outside Celune's singing range (F#3–C5)."
    assert error.value.source_start == notation.index("f3")
    assert error.value.source_end == notation.index("f3") + 2
    assert f"{' ' * notation.index('f3')}^~" in format_cesn_error(
        notation,
        error.value,
    )
    assert isinstance(error.value, CESNValidationError)
    assert decode_score(payload)[0].events[0].pitch == "f#3"


def test_cesn_preprocessing_allows_a_different_characters_pitch_range() -> None:
    """Allow a pitch outside Celune's range when another pack declares it."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Nova", Path("nova.cevoice"), (48, 84))
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        payload = backend.cesn_to_model("[lang:en][f5:4][ɛ]Ah.")

    assert decode_score(payload)[0].events[0].pitch == "f5"
