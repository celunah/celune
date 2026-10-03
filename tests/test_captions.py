# SPDX-License-Identifier: Apache-2.0
"""Tests for optional IPA caption alignment."""

import json
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np

from celune import captions


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


def test_caption_aligner_loads_model_lazily_and_aligns_ipa(
    tmp_path: Path,
) -> None:
    """Load the selected acoustic model only when IPA alignment is requested."""
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
    download = mock.Mock(side_effect=("model.onnx", str(vocab_path)))
    hub = SimpleNamespace(hf_hub_download=download)
    runtime = SimpleNamespace(InferenceSession=inference_session)
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
        providers=["CPUExecutionProvider"],
    )
    assert session.run.call_count == 2
    assert espeak.call_args.args[0] == [
        "espeak-ng",
        "-q",
        "--ipa",
        "-v",
        "en-us",
        "hello",
    ]
