# SPDX-License-Identifier: Apache-2.0
"""Tests for pure analysis helpers and RGB glow math."""

from pathlib import Path
from unittest import mock

import torch
import numpy as np
import pytest
from celune.colors import RGB
from celune import i18n, analysis
from celune.chroma import AudioRGBGlow
from celune.constants import BASE_SR, N_A_NUMERIC

from .support import CeluneTestCase


class TestAnalysis(CeluneTestCase):
    """Tests for deterministic analysis helper behavior."""

    def test_embedding_similarity_and_drift_helpers_validate_inputs(self) -> None:
        """Validate embedding conversion, similarity, and drift helper paths.

        Raises:
            AssertionError: An analysis helper returns an unexpected result.
        """
        embedding = np.ones(2048, dtype=np.float32)
        converted = analysis.embedding_tensor_to_numpy(embedding)
        assert converted.shape == (2048,)

        with pytest.raises(ValueError, match="2048-size"):
            analysis.embedding_tensor_to_numpy(np.ones(3, dtype=np.float32))

        cosine, percent = analysis.cosine_similarity_percent(embedding, embedding)
        assert round(abs(cosine - 1.0), 7) == 0
        assert round(abs(percent - 100.0), 7) == 0
        with pytest.raises(ValueError, match="norm is zero"):
            analysis.cosine_similarity_percent(
                np.zeros(2048, dtype=np.float32),
                embedding,
            )

        assert analysis.voice_drift_level(2.0) == "stable"
        assert analysis.voice_drift_level(5.0) == "expressive"
        assert analysis.voice_drift_level(8.0) == "weak"
        assert analysis.voice_drift_level(12.0) == "wrong"

    def test_traits_and_assessment_cover_speech_and_empty_audio_paths(self) -> None:
        """Check trait and assessment output for empty voice extraction data.

        Raises:
            AssertionError: Trait or assessment output changes unexpectedly.
        """
        metrics = {
            "duration_s": 1.0,
            "pitch_extraction_ok": False,
            "pitch_mean_hz": N_A_NUMERIC,
            "pitch_median_hz": N_A_NUMERIC,
            "pitch_std_hz": N_A_NUMERIC,
            "pitch_peak_hz": N_A_NUMERIC,
            "pitch_variance": N_A_NUMERIC,
            "voice_extraction_ok": False,
            "dynamic_range_db": 0.0,
            "speaking_pace_proxy": 0.0,
            "rms_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "hf_energy_ratio": 0.0,
            "zcr_mean": 0.0,
            "voiced_ratio": 0.0,
            "pause_ratio": 1.0,
        }
        traits = analysis.compute_traits(metrics)
        assert set(traits.values()) == {0.0}
        assessment = analysis.generate_assessment(metrics, traits)
        assert "No voicings found." in assessment[1]
        assert "Mean pitch could not be determined" in assessment[3]
        assert "high pause ratio" in assessment[-1].lower()

    @mock.patch("celune.analysis.librosa.stft", return_value=np.ones((4, 2)))
    @mock.patch(
        "celune.analysis.librosa.feature.zero_crossing_rate",
        return_value=np.array([[0.1, 0.2]], dtype=np.float32),
    )
    @mock.patch(
        "celune.analysis.librosa.feature.spectral_centroid",
        return_value=np.array([[1000.0, 1200.0]], dtype=np.float32),
    )
    @mock.patch(
        "celune.analysis.librosa.pyin",
        return_value=(
            np.array([100.0, 200.0, np.nan, 300.0], dtype=np.float32),
            np.array([True, True, False, True]),
            None,
        ),
    )
    @mock.patch(
        "celune.analysis.librosa.feature.rms",
        return_value=np.array([[0.5, 0.25]], dtype=np.float32),
    )
    @mock.patch("celune.analysis.librosa.get_duration", return_value=2.0)
    @mock.patch(
        "celune.analysis.librosa.fft_frequencies",
        return_value=np.array([0.0, 2000.0, 4000.0, 6000.0], dtype=np.float32),
    )
    def test_compute_raw_metrics_adds_extended_pitch_statistics(
        self,
        _fft_frequencies: mock.Mock,
        _duration: mock.Mock,
        _rms: mock.Mock,
        _pyin: mock.Mock,
        _centroid: mock.Mock,
        _zcr: mock.Mock,
        _stft: mock.Mock,
    ) -> None:
        """Verify pitch metrics include median, standard deviation, and voiced peak.

        Args:
            _fft_frequencies: Mocked FFT frequencies.
            _duration: Mocked duration value.
            _rms: Mocked RMS value.
            _pyin: Mocked pyin value.
            _centroid: Mocked centroid value.
            _zcr: Mocked ZCR value.
            _stft: Mocked STFT value.
        """
        metrics = analysis.compute_raw_metrics(
            np.ones(4096, dtype=np.float32),
            16000,
        )
        expected_voiced_f0 = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        assert metrics["pitch_mean_hz"] == 200.0
        assert metrics["pitch_median_hz"] == 200.0
        assert (
            round(abs(metrics["pitch_std_hz"] - float(np.std(expected_voiced_f0))), 6)
            == 0
        )
        assert metrics["pitch_peak_hz"] == 300.0
        assert (
            round(abs(metrics["pitch_variance"] - float(np.var(expected_voiced_f0))), 6)
            == 0
        )

    @mock.patch("celune.analysis.default_loader", return_value=None)
    def test_reference_embeddings_are_unavailable_without_bundle(
        self, _default_loader: mock.Mock
    ) -> None:
        """Verify reference embeddings require an active CEVOICE bundle.

        Args:
            _default_loader: A mock default loader.
        """
        assert analysis.available_reference_voices() == []

    @mock.patch("celune.analysis.default_loader", return_value=None)
    def test_reference_embedding_load_requires_bundle(
        self, _default_loader: mock.Mock
    ) -> None:
        """Verify reference embedding loading fails without a CEVOICE bundle.

        Args:
            _default_loader: A mock default loader.
        """
        with pytest.raises(FileNotFoundError, match="no compatible CEVOICE/CECHAR"):
            analysis.load_reference_embedding("balanced")

    def test_reference_failures_use_active_locale(self) -> None:
        """Verify reference embedding failures and metrics use localized messages."""
        original_strings = dict(i18n.STRINGS)
        original_locale = i18n.get_locale()
        try:
            i18n.STRINGS["zz"] = {
                "analysis.reference_embedding_bundle_missing": "localized bundle missing",
                "analysis.reference_similarity_error": "localized similarity: {reason}",
            }
            i18n.set_locale("zz")

            with (
                mock.patch("celune.analysis.default_loader", return_value=None),
                self.assertRaisesRegex(
                    FileNotFoundError,
                    "localized bundle missing",
                ),
            ):
                analysis.load_reference_embedding("balanced")

            metrics: dict = {}
            with mock.patch(
                "celune.analysis._compute_qwen3_embedding",
                side_effect=RuntimeError("backend detail"),
            ):
                analysis.add_reference_similarity_metrics(
                    metrics,
                    np.zeros(160, dtype=np.float32),
                    BASE_SR,
                    "balanced",
                )

            self.assertEqual(
                metrics["voice_similarity_error"],
                "localized similarity: backend detail",
            )
        finally:
            i18n.set_locale(original_locale)
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original_strings)

    @mock.patch("celune.analysis.torch.load")
    def test_bundle_reference_embedding_is_materialized_when_available(
        self,
        torch_load: mock.Mock,
    ) -> None:
        """Verify bundle-provided .pt references are loaded from a materialized file path.

        Args:
            torch_load: The mocked value of torch.load().
        """
        torch_load.return_value = torch.ones(2048, dtype=torch.float32)
        fake_loader = mock.Mock()
        materialized = Path("C:/Users/user/AppData/Local/Celune/temp/fake/balanced.pt")
        fake_loader.materialize.return_value = materialized

        with mock.patch("celune.analysis.default_loader", return_value=fake_loader):
            embedding = analysis.load_reference_embedding("balanced")

        assert embedding.shape == (2048,)
        fake_loader.materialize.assert_called_once_with("balanced", "pt")
        torch_load.assert_called_once_with(
            materialized,
            map_location="cpu",
            weights_only=True,
        )

    @mock.patch("celune.analysis.torch.load")
    def test_reference_embedding_rejects_unexpected_objects(
        self,
        torch_load: mock.Mock,
    ) -> None:
        """Reject reference files containing non-tensor or unexpected mappings.

        Args:
            torch_load: The mocked value of torch.load().
        """
        fake_loader = mock.Mock()
        fake_loader.materialize.return_value = Path("balanced.pt")

        invalid_values = (
            (np.ones(2048, dtype=np.float32), TypeError),
            ({"unexpected": torch.ones(2048)}, ValueError),
        )
        for value, expected_error in invalid_values:
            torch_load.return_value = value
            with (
                self.subTest(value_type=type(value).__name__),
                mock.patch(
                    "celune.analysis.default_loader",
                    return_value=fake_loader,
                ),
                self.assertRaisesRegex(
                    expected_error,
                    "invalid reference embedding",
                ),
            ):
                analysis.load_reference_embedding("balanced")

    @mock.patch("celune.analysis.torch.load")
    def test_reference_embedding_rejects_invalid_shape_or_dtype(
        self,
        torch_load: mock.Mock,
    ) -> None:
        """Reject reference tensors with an unsupported shape or dtype.

        Args:
            torch_load: The mocked value of torch.load().
        """
        fake_loader = mock.Mock()
        fake_loader.materialize.return_value = Path("balanced.pt")

        for value in (
            torch.ones(2047, dtype=torch.float32),
            torch.ones(2048, dtype=torch.float64),
            torch.ones(2, 1024, dtype=torch.float32),
        ):
            torch_load.return_value = value
            with (
                self.subTest(shape=tuple(value.shape), dtype=value.dtype),
                mock.patch(
                    "celune.analysis.default_loader",
                    return_value=fake_loader,
                ),
                self.assertRaisesRegex(ValueError, "invalid reference embedding"),
            ):
                analysis.load_reference_embedding("balanced")


class TestChroma(CeluneTestCase):
    """Tests for pure RGB glow helper behavior."""

    def test_pure_glow_helpers_process_audio_without_devices(self) -> None:
        """Exercise glow math without connecting to RGB devices.

        Raises:
            AssertionError: A glow helper returns an unexpected value.
        """
        stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mono = AudioRGBGlow.to_mono(stereo)
        assert np.array_equal(mono, np.array([0.5, 0.5], dtype=np.float32))

        fixed = AudioRGBGlow.fix_color_rendering((255, 255, 255))
        assert len(fixed) == 3
        assert max(fixed) <= 255

        glow = AudioRGBGlow(celune=None, color="#ffffff")
        assert glow.speech_level(np.zeros((0, 2), dtype=np.float32)) == 0.0
        assert glow.speech_level(stereo) > 0.0

    def test_sleep_and_wake_preserve_prior_brightness_target(self) -> None:
        """Verify sleep dimming stores and restores the earlier brightness target."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        glow.start = mock.Mock(return_value=True)
        glow._current_brightness = 0.42
        glow._target_brightness = 0.6
        glow._state = "normal"

        glow.sleep()

        assert glow._state == "sleeping"
        assert round(abs(glow._sleep_restore_brightness - 0.6), 7) == 0
        assert round(abs(glow._target_brightness - glow.idle_brightness * 0.25), 7) == 0

        glow.wake()

        assert glow._state == "waking"
        assert round(abs(glow._target_brightness - 0.6), 7) == 0

    def test_sleeping_glow_ignores_audio_reactivity(self) -> None:
        """Verify queued audio cannot knock the glow out of its sleeping state."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        glow._state = "sleeping"
        sleeping_target = glow.idle_brightness * 0.25
        glow._target_brightness = sleeping_target

        glow.process_glow_chunk(np.ones((64, 2), dtype=np.float32), 0.0)

        assert glow._state == "sleeping"
        assert round(abs(glow._target_brightness - sleeping_target), 7) == 0

    def test_schedule_uses_one_chunk_per_display_frame(self) -> None:
        """Verify glow scheduling does not subdivide frames into tiny chunks."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        glow.start = mock.Mock(return_value=True)
        audio = np.zeros((BASE_SR, 2), dtype=np.float32)

        glow.schedule(audio)

        assert len(glow._scheduled_chunks) == glow.fps
        assert len(glow._scheduled_chunks[0][1]) == BASE_SR // glow.fps

    def test_glow_target_follows_smoothed_audio_rms_without_snapping_to_max(
        self,
    ) -> None:
        """Verify glow brightness tracks RMS amplitude smoothly."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        quiet = np.full((glow.fps, 2), 0.05, dtype=np.float32)
        peak = np.zeros((glow.fps, 2), dtype=np.float32)
        peak[0] = 1.0

        glow.process_glow_chunk(quiet, 0.0)
        quiet_target = glow._target_brightness
        assert quiet_target == glow.idle_brightness
        assert quiet_target < glow.max_brightness

        glow.process_glow_chunk(peak, 0.1)
        assert glow._target_brightness < glow.max_brightness

    def test_glow_worker_uses_audio_target_without_fixed_pulse_logic(self) -> None:
        """Verify the normal glow branch follows audio target directly."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        glow._state = "normal"
        glow._current_brightness = glow.idle_brightness
        glow._target_brightness = min(glow.max_brightness, glow.idle_brightness + 0.4)
        glow._current_color = glow.base_color.copy()
        glow._target_color = glow.base_color.copy()

        writes: list[RGB] = []
        glow._set_all_devices = lambda rgb: writes.append(
            (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        )

        def stop_after_two_sleeps(_seconds: float) -> None:
            if len(writes) >= 2:
                glow._stop_event.set()

        with mock.patch("celune.chroma.time.sleep", side_effect=stop_after_two_sleeps):
            glow.run()

        assert len(writes) >= 1
        assert glow._current_brightness > glow.idle_brightness

    def test_reset_audio_reactivity_clears_pending_audio_and_restores_idle(
        self,
    ) -> None:
        """Verify abrupt playback resets the audio-reactive envelope to idle."""
        glow = AudioRGBGlow(celune=None, color="#ffffff")
        glow._worker = mock.Mock()
        glow._worker.is_alive.return_value = True
        glow._state = "normal"
        glow._smoothed_level = 0.8
        glow._target_brightness = min(glow.max_brightness, glow.idle_brightness + 0.4)
        glow._scheduled_chunks.append((0.0, np.ones((32, 2), dtype=np.float32)))

        glow.reset_audio_reactivity()

        assert len(glow._scheduled_chunks) == 0
        assert glow._smoothed_level == 0.0
        assert glow._state == "normal"
        assert round(abs(glow._target_brightness - glow.idle_brightness), 7) == 0
