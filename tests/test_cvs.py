# SPDX-License-Identifier: Apache-2.0
"""Tests for Cel Voice Score calculations and reporting."""

import sys
import types
import sqlite3
import logging
from pathlib import Path
from unittest import mock
from contextlib import nullcontext

import numpy as np
import torch
import pytest

from celune import analysis
from celune.cvs import (
    CVS_DATABASE_NAME,
    language_key,
    calculate_cvs,
    record_cvs_utterance,
    word_error_counts,
    word_error_rate_percent,
    supports_word_error_rate,
)
from celune.persona.asr import WhisperTranscriber, _GenerationConfigWarningFilter


@pytest.mark.parametrize(
    ("similarity", "expected"),
    ((97.0, 100.0), (92.0, 50.0), (87.0, 0.0), (110.0, 100.0)),
)
def test_cvs_voice_match_thresholds(similarity: float, expected: float) -> None:
    """Score voice match against the specified cosine thresholds."""
    assert calculate_cvs(similarity, 0.0, 500.0, 1.2).voice_match == expected


def test_cvs_weights_and_missing_components() -> None:
    """Combine four component scores and omit totals with missing inputs."""
    result = calculate_cvs(92.0, 2.0, 1000.0, 0.6)

    assert result.score == pytest.approx(68.0)
    assert result.voice_match == 50.0
    assert result.wer == 90.0
    assert result.latency == 90.0
    assert result.realtime_speed == 50.0
    assert calculate_cvs(97.0, None, 500.0, 1.2).score is None


def test_cvs_component_limits() -> None:
    """Clamp WER, latency, and realtime speed at their documented limits."""
    assert calculate_cvs(97.0, 1.0, 500.0, 1.2).wer == 95.0
    assert calculate_cvs(97.0, 20.0, 5500.0, 0.0).wer == 0.0
    assert calculate_cvs(97.0, 0.0, 5500.0, 0.0).latency == 0.0
    assert calculate_cvs(97.0, 0.0, 500.0, 0.0).realtime_speed == 0.0


def test_word_error_rate_normalizes_and_counts_edits() -> None:
    """Normalize punctuation and count substitutions, insertions, and deletions."""
    assert word_error_rate_percent("Hello, world!", "hello world") == 0.0
    assert word_error_rate_percent("hello world", "hello there") == 50.0
    assert word_error_rate_percent("hello", "hello there") == 100.0
    assert word_error_rate_percent("hello there", "hello") == 50.0
    assert word_error_rate_percent("...", "anything") is None
    assert word_error_counts("Hello, world!", "hello world") == (0, 2)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    (
        ("DON'T stop.", "dont STOP"),
        ("A well-known phrase", "a well known phrase!"),
        ("There are 1,000 items.", "there are 1000 items"),
    ),
)
def test_word_error_rate_ignores_formatting_differences(
    reference: str,
    hypothesis: str,
) -> None:
    """Ignore casing and common punctuation differences in ASR transcripts."""
    assert word_error_rate_percent(reference, hypothesis) == 0.0


def test_whisper_filter_only_suppresses_generation_config_warning() -> None:
    """Suppress the known upstream warning while preserving other warnings."""
    warning_filter = _GenerationConfigWarningFilter()
    target = logging.LogRecord(
        "transformers.generation.utils",
        logging.WARNING,
        __file__,
        1,
        "Passing `generation_config` together with generation-related arguments=('suppress_tokens',)",
        (),
        None,
    )
    unrelated = logging.LogRecord(
        "transformers.generation.utils",
        logging.WARNING,
        __file__,
        1,
        "A different warning",
        (),
        None,
    )

    assert warning_filter.filter(target) is False
    assert warning_filter.filter(unrelated) is True


def test_word_error_rate_rejects_unsegmented_scripts() -> None:
    """Avoid labeling character-level comparisons as word error rates."""
    reference = "今日はいい天気です"
    assert not supports_word_error_rate(reference)
    assert word_error_rate_percent(reference, reference) is None
    assert supports_word_error_rate("This is English.")


def test_language_key_normalizes_language_tags() -> None:
    """Normalize locale tags for CVS labels and aggregates."""
    assert language_key("EN_us") == "en"
    assert language_key("unknown") == "unknown"
    assert language_key(None) == "unknown"


def test_cvs_database_averages_complete_scores_by_backend(tmp_path: Path) -> None:
    """Persist hashed utterance identifiers and calculate backend means."""
    database_path = tmp_path / CVS_DATABASE_NAME
    complete = calculate_cvs(97.0, 0.0, 500.0, 1.2)
    lower = calculate_cvs(92.0, 1.0, 750.0, 0.6)

    aggregate, count = record_cvs_utterance(
        "private/audio/one.wav",
        "pocket_tts",
        "en",
        complete,
        similarity_percent=97.0,
        wer_percent=0.0,
        latency_ms=500.0,
        realtime_speed=1.2,
        database_path=database_path,
    )
    assert aggregate == 100.0
    assert count == 1

    aggregate, count = record_cvs_utterance(
        "private/audio/two.wav",
        "pocket_tts",
        "fr",
        lower,
        similarity_percent=92.0,
        wer_percent=1.0,
        latency_ms=750.0,
        realtime_speed=0.6,
        database_path=database_path,
    )
    assert aggregate == pytest.approx(85.125)
    assert count == 2

    record_cvs_utterance(
        "private/audio/two.wav",
        "pocket_tts",
        "fr",
        complete,
        similarity_percent=97.0,
        wer_percent=0.0,
        latency_ms=500.0,
        realtime_speed=1.2,
        database_path=database_path,
    )
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT utterance_key FROM cvs_utterances").fetchall()
    assert len(rows) == 2
    assert all(len(row[0]) == 64 for row in rows)
    assert all("private/audio" not in row[0] for row in rows)


def test_analysis_adds_wer_with_the_reviewed_tiny_model() -> None:
    """Transcribe the generated utterance and always unload the tiny model."""
    transcriber = mock.Mock()
    transcriber.transcribe.return_value = "hello there"
    metrics: dict = {}

    with mock.patch(
        "celune.persona.asr.WhisperTranscriber",
        return_value=transcriber,
    ) as transcriber_factory:
        analysis._add_word_error_metrics(
            metrics,
            np.zeros(16000, dtype=np.float32),
            16000,
            "Hello, there!",
            "en-US",
        )

    assert metrics["wer_ok"] is True
    assert metrics["wer_percent"] == 0.0
    assert transcriber_factory.call_args.kwargs == {
        "language": "en",
        "allow_cpu": True,
    }
    transcriber.unload.assert_called_once_with()


def test_whisper_tiny_can_load_on_cpu_for_analysis() -> None:
    """Use an unquantized float32 model when analysis has no CUDA device."""
    model = mock.Mock()
    model.generation_config = types.SimpleNamespace(is_multilingual=True)
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    model_loader = mock.Mock(return_value=model)
    processor_loader = mock.Mock(return_value=mock.Mock())
    transformers = types.ModuleType("transformers")
    vars(transformers).update(
        {
            "AutoModelForSpeechSeq2Seq": types.SimpleNamespace(
                from_pretrained=model_loader
            ),
            "AutoProcessor": types.SimpleNamespace(from_pretrained=processor_loader),
        }
    )
    transcriber = WhisperTranscriber("openai/whisper-tiny", allow_cpu=True)

    with (
        mock.patch.dict(sys.modules, {"transformers": transformers}),
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch("celune.persona.asr.require_model_consent"),
        mock.patch(
            "celune.persona.asr.huggingface_progress",
            return_value=nullcontext(),
        ),
    ):
        transcriber._load_model()

    assert transcriber._dtype == torch.float32
    assert model_loader.call_args.kwargs["device_map"] == "cpu"
    assert model_loader.call_args.kwargs["dtype"] == torch.float32
    assert "quantization_config" not in model_loader.call_args.kwargs


def test_voice_report_replaces_similarity_with_cvs(tmp_path: Path) -> None:
    """Write the CVS report and omit a standalone cosine similarity report."""
    metrics = {
        "duration_s": 1.5,
        "sample_rate": 16000,
        "rms_mean": 0.2,
        "rms_std": 0.1,
        "dynamic_range_db": 24.0,
        "pitch_extraction_ok": False,
        "voiced_ratio": 0.7,
        "pause_ratio": 0.3,
        "speaking_pace_proxy": 0.8,
        "spectral_centroid_mean": 1200.0,
        "zcr_mean": 0.02,
        "hf_energy_ratio": 0.1,
        "cvs_language": "en",
        "cvs_score": 90.0,
        "cvs_aggregate": 88.0,
        "cvs_aggregate_count": 4,
        "cvs_voice_match_score": 100.0,
        "voice_similarity_percent": 97.0,
        "cvs_wer_score": 95.0,
        "wer_percent": 1.0,
        "wer_errors": 1,
        "wer_reference_words": 100,
        "cvs_latency_score": 100.0,
        "latency_ms": 500.0,
        "cvs_realtime_speed_score": 100.0,
        "realtime_speed": 1.2,
    }
    output_path = tmp_path / "report.txt"

    analysis.write_report(metrics, {}, [], Path("sample.wav"), output_path)

    report = output_path.read_text(encoding="utf-8")
    assert "CVS[en]: 90.0 / 100" in report
    assert "CVS[all]: 88.0 / 100 (mean of 4 utterances)" in report
    assert "Word error rate" in report
    assert "word edits: 1/100" in report
    assert "Cosine similarity" not in report
