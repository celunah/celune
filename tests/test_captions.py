# SPDX-License-Identifier: Apache-2.0
"""Tests for optional IPA caption alignment."""

import json
import threading
from typing import cast
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np

from celune import captions
from celune.celune import Celune
from celune.dataclasses.pipeline import (
    CaptionPlaybackSegment,
    CaptionPlaybackState,
    SpeechTiming,
)


def test_map_word_timings_keeps_aligned_boundaries() -> None:
    """Keep one-to-one word timings from the forced aligner unchanged."""
    aligned = ((0.1, 0.3), (0.45, 0.8))

    assert captions.map_word_timings(("one", "two"), aligned, 1.0) == aligned


def test_map_word_timings_maps_normalized_text_within_its_chunk() -> None:
    """Split alignment timing across display words when normalization differs."""
    assert captions.map_word_timings(
        ("one", "two", "three", "four"),
        ((0.1, 0.3), (0.5, 0.9)),
        1.0,
    ) == ((0.1, 0.25), (0.25, 0.4), (0.4, 0.65), (0.65, 0.9))


def test_align_chunk_word_start_frames_reports_alignment_errors() -> None:
    """Report the reason and return the fallback signal on alignment failure."""
    aligner = mock.Mock(align_words=mock.Mock(side_effect=ValueError("bad transcript")))
    logger = mock.Mock()

    with mock.patch("celune.captions.get_caption_aligner", return_value=aligner):
        assert (
            captions.align_chunk_word_start_frames(
                (np.ones(12, dtype=np.float32),),
                48000,
                "hello",
                ("hello",),
                "en-US",
                logger,
                "debug",
            )
            is None
        )

    assert "bad transcript" in logger.call_args.args[0]
    assert logger.call_args.args[1] == "warning"


def test_caption_timing_is_registered_before_audio_is_queued(monkeypatch) -> None:
    """Make caption ranges visible before playback can advance through audio."""
    engine = SimpleNamespace(
        queue_lock=threading.RLock(),
        _playback_source_meta={3: {"total_frames": 4.0}},
        _playback_caption_states={3: CaptionPlaybackState(total_words=2)},
    )
    worker = captions.CaptionAlignmentWorker.__new__(captions.CaptionAlignmentWorker)
    worker._state_lock = threading.Lock()
    worker._cancelled_sources = set()
    worker._logger = mock.Mock()
    worker._log_level = "info"

    def flush_audio(*_args, **_kwargs) -> bool:
        assert engine._playback_caption_states[3].segments == [
            CaptionPlaybackSegment(
                start_frame=4,
                end_frame=9,
                word_start=0,
                word_end=2,
                timing_words=("one", "two"),
                word_start_frames=(1, 3),
            )
        ]
        return True

    monkeypatch.setattr(
        "celune.captions.align_chunk_word_start_frames",
        lambda *_args: (1, 3),
    )
    monkeypatch.setattr("celune.playback._flush_buffered_speech_chunks", flush_audio)

    worker._align_and_flush_chunk(
        cast(Celune, engine),
        3,
        [np.ones(5, dtype=np.float32)],
        SpeechTiming(start_time=0.0),
        None,
        "one two",
        "one two",
        ("one", "two"),
        "en-US",
        0,
        2,
        ("one", "two"),
        [False],
        [False],
        48000,
    )


def test_missing_espeak_logs_a_warning_without_traceback() -> None:
    """Report a missing eSpeak executable as a concise warning."""
    logger = mock.Mock()
    aligner = captions.CaptionAligner()

    with (
        mock.patch("celune.captions.get_caption_aligner", return_value=aligner),
        mock.patch(
            "celune.captions.subprocess.run",
            side_effect=FileNotFoundError("espeak-ng"),
        ),
    ):
        assert (
            captions.align_chunk_word_start_frames(
                (np.ones(12, dtype=np.float32),),
                48000,
                "hello",
                ("hello",),
                "en-US",
                logger,
                "info",
            )
            is None
        )

    assert logger.call_args.args == (
        captions.string("ui.caption_espeak_not_found"),
        "warning",
    )


def test_missing_onnxruntime_logs_a_warning_without_traceback() -> None:
    """Report an absent ONNX Runtime extra as a concise warning."""
    logger = mock.Mock()
    aligner = captions.CaptionAligner()

    def load_module(module_name: str) -> object:
        if module_name == "huggingface_hub":
            return SimpleNamespace()
        if module_name == "onnxruntime":
            raise ModuleNotFoundError(
                "No module named 'onnxruntime'",
                name="onnxruntime",
            )
        raise AssertionError(f"unexpected optional module: {module_name}")

    with (
        mock.patch("celune.captions.get_caption_aligner", return_value=aligner),
        mock.patch(
            "celune.captions.subprocess.run",
            return_value=SimpleNamespace(stdout="hɛloʊ\n"),
        ),
        mock.patch.object(
            captions.importlib,
            "import_module",
            side_effect=load_module,
        ),
    ):
        assert (
            captions.align_chunk_word_start_frames(
                (np.ones(12, dtype=np.float32),),
                48000,
                "hello",
                ("hello",),
                "en-US",
                logger,
                "info",
            )
            is None
        )

    assert logger.call_args.args == (
        captions.string("ui.caption_onnxruntime_not_found"),
        "warning",
    )


def test_caption_aligner_preloads_model_in_background_once() -> None:
    """Prepare the alignment model asynchronously and only once."""
    aligner = captions.CaptionAligner()
    loading_started = threading.Event()
    allow_load_to_finish = threading.Event()
    model_ready = threading.Event()
    messages: list[tuple[str, str]] = []

    def load_model() -> tuple[mock.Mock, dict[str, int]]:
        loading_started.set()
        allow_load_to_finish.wait()
        return mock.Mock(), {}

    def log(message: str, severity: str) -> None:
        messages.append((message, severity))
        if message == captions.string("ui.caption_alignment_ready"):
            model_ready.set()

    with mock.patch.object(aligner, "_load_model", side_effect=load_model) as load:
        aligner.preload(log, "info")
        try:
            assert loading_started.wait(timeout=2)
            aligner.preload(log, "info")
            assert load.call_count == 1
        finally:
            allow_load_to_finish.set()

    assert model_ready.wait(timeout=2)
    assert messages == [
        (captions.string("ui.caption_alignment_loading"), "info"),
        (captions.string("ui.caption_alignment_ready"), "info"),
    ]


def test_caption_aligner_loads_model_lazily_and_aligns_ipa(
    tmp_path: Path,
) -> None:
    """Load the acoustic model on demand when startup preload is not used."""
    vocab = {"<pad>": 0, "h": 1, "ɛ": 2, "l": 3, "o": 4, "ʊ": 5}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    logits = np.zeros((1, 11, len(vocab)), dtype=np.float32)
    frame_tokens = (0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0)
    for frame, token_id in enumerate(frame_tokens):
        logits[0, frame, token_id] = 10.0

    session = mock.Mock()
    session.get_inputs.return_value = (SimpleNamespace(name="audio"),)
    session.get_outputs.return_value = (SimpleNamespace(name="logits"),)
    session.run.return_value = (logits,)
    inference_session = mock.Mock(return_value=session)
    session_options = SimpleNamespace(log_severity_level=2)
    download = mock.Mock(side_effect=("model.onnx", str(vocab_path)))
    hub = SimpleNamespace(hf_hub_download=download)
    runtime = SimpleNamespace(
        InferenceSession=inference_session,
        SessionOptions=mock.Mock(return_value=session_options),
    )
    aligner = captions.CaptionAligner()
    waveform = np.linspace(-0.5, 0.5, 14400, dtype=np.float32)

    def load_module(module_name: str) -> object:
        if module_name == "huggingface_hub":
            return hub
        if module_name == "onnxruntime":
            return runtime
        raise AssertionError(f"unexpected optional module: {module_name}")

    with (
        mock.patch.object(
            captions.importlib,
            "import_module",
            side_effect=load_module,
        ) as import_module,
        mock.patch.object(
            captions.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="hɛloʊ\n"),
        ) as espeak,
    ):
        assert import_module.call_count == 0
        assert aligner.align_words(waveform, 48000, "hello", "en-US") == ((0.02, 0.2),)
        assert aligner.align_words(waveform, 48000, "hello", "en-US") == ((0.02, 0.2),)

    assert import_module.call_args_list == [
        mock.call("huggingface_hub"),
        mock.call("onnxruntime"),
    ]
    assert download.call_args_list == [
        mock.call("sadda-speech/wav2vec2-espeak-ctc", "model.onnx"),
        mock.call("sadda-speech/wav2vec2-espeak-ctc", "vocab.json"),
    ]
    inference_session.assert_called_once_with(
        "model.onnx",
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    assert session_options.log_severity_level == 3
    assert session.run.call_count == 2
    assert espeak.call_args.args[0] == [
        "espeak-ng",
        "-q",
        "--ipa",
        "-v",
        "en-us",
        "hello",
    ]
