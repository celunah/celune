# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit CESN IPA pronunciation overrides."""

import json
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from typing import Optional, cast

import pytest
import numpy as np

from celune.exceptions import (
    BackendError,
    CESNBackendFormatError,
    CESNRangeError,
    CESNValidationError,
)
from celune.cesn import CESNSinger
from celune.backends.tts.soulxsinger import (
    SoulXSinger,
    decode_score,
    pitch_to_midi,
    _SoulXRuntime,
)


@pytest.mark.parametrize(
    ("ipa", "expected_phonemes", "expected_error"),
    (
        ("/ˈkwaɪ.ət/", "en_K-W-AY1 en_AH0-T", None),
        ("/ˈkwaɪ.ɲ/", None, "soulx_ipa_unsupported"),
    ),
)
def test_generate_stream_uses_or_rejects_explicit_ipa(
    ipa: str,
    expected_phonemes: Optional[str],
    expected_error: Optional[str],
) -> None:
    """Use IPA phones instead of G2P and reject phones outside SoulX's inventory."""
    backend = object.__new__(SoulXSinger)
    backend.name = "soulxsinger"
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    phonemizer = mock.Mock(side_effect=AssertionError("G2P must be bypassed"))
    processor = SimpleNamespace(
        phone2idx={
            "en_K": 0,
            "en_W": 1,
            "en_AY1": 2,
            "en_AH0": 3,
            "en_T": 4,
        },
        process=mock.Mock(return_value={"target": "target"}),
    )
    inference_model = SimpleNamespace(
        infer=mock.Mock(return_value=np.array([0.25], dtype=np.float32))
    )
    runtime = SimpleNamespace(
        model=inference_model,
        processor=processor,
        phonemizer=phonemizer,
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
    notation = f"[lang:en][pattern:qui,et][ipa:{ipa}][g3:4,a3:4]Quiet."

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
            return_value=("[lang:en][g3:4]Ah", Path("sing.wav")),
        ),
        mock.patch.object(SoulXSinger, "_prompt_data", return_value={}),
        mock.patch(
            "celune.backends.tts.soulxsinger.to_numpy_audio",
            side_effect=lambda audio: audio,
        ),
    ):
        if expected_error is None:
            list(backend.generate_stream(loaded_runtime, text=notation))
        else:
            with pytest.raises(BackendError) as error:
                list(backend.generate_stream(loaded_runtime, text=notation))
            assert error.value.error_code == expected_error

    phonemizer.assert_not_called()
    if expected_phonemes is not None:
        processor.process.assert_called_once()
        assert processor.process.call_args.args[0]["phoneme"] == expected_phonemes
        assert processor.process.call_args.args[0]["text"] == "Qui et."


def test_slurred_cesn_keeps_distinct_syllables_as_note_events() -> None:
    """Keep a new slurred syllable's pitch, phones, and duration."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    notation = "[lang:en][pattern:cel,lo][ipa:/ˈtʃɛl.oʊ/][slur:2][c4:4,d4:4]cello."
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        score_text = backend.cesn_to_model(notation)

    section = decode_score(score_text)[0]
    phonemizer = mock.Mock(side_effect=AssertionError("IPA must bypass G2P"))
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
            phonemizer=phonemizer,
        ),
    )

    metadata = SoulXSinger._metadata(section.events, section.bpm, runtime)

    assert metadata["note_type"] == "2 2"
    assert metadata["duration"] == "0.5 0.5"
    assert metadata["note_pitch"] == "60 62"
    assert metadata["phoneme"] == "en_CH-EH1-L en_OW0"
    assert tuple(event.slur_from_previous for event in section.events) == (False, True)
    phonemizer.assert_not_called()

    legacy_payload = json.loads(score_text)
    for raw_event in legacy_payload["sections"][0]["events"]:
        raw_event.pop("slur_from_previous")
    legacy_section = decode_score(json.dumps(legacy_payload))[0]
    assert not any(event.slur_from_previous for event in legacy_section.events)


def test_slurred_repeated_phones_use_soulx_continuation_type() -> None:
    """Use SoulX continuation only when the following event repeats the phones."""
    backend = object.__new__(SoulXSinger)
    singer = CESNSinger("Test Singer", Path("singer.cevoice"))
    notation = "[lang:en][pattern:la,la][ipa:/ˈlɑ.ˈlɑ/][slur:2][c4:4,d4:4]lala."
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
            phonemizer=mock.Mock(side_effect=AssertionError("IPA must bypass G2P")),
        ),
    )

    metadata = SoulXSinger._metadata(section.events, section.bpm, runtime)

    assert metadata["note_type"] == "2 3"
    assert metadata["note_pitch"] == "60 62"
    assert metadata["phoneme"] == "en_L-AA1 en_L-AA1"


@pytest.mark.parametrize(
    "text",
    (
        "not prepared score JSON",
        '{"backend":"soulxsinger","version":1,"sections":[]}',
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
    with mock.patch(
        "celune.backends.tts.soulxsinger.resolve_active_singer",
        return_value=singer,
    ):
        with pytest.raises(CESNRangeError) as error:
            backend.cesn_to_model("[lang:en][bpm:120,key:f#,scale:major][f3:4]Ah.")
        payload = backend.cesn_to_model(
            "[lang:en][bpm:120,key:f#,scale:major][f#3:4]Ah."
        )

    assert str(error.value) == (
        "note F3 (MIDI 53) is outside Celune's acceptable range: F#3–C5 (MIDI 54–72)"
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
        payload = backend.cesn_to_model("[lang:en][f5:4]Ah.")

    assert decode_score(payload)[0].events[0].pitch == "f5"
