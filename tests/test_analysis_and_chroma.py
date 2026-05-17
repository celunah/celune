# SPDX-License-Identifier: MIT
"""Tests for pure analysis helpers and RGB glow math."""

import unittest
from unittest import mock

import numpy as np

from celune import analysis
from celune.chroma import AudioRGBGlow
from celune.constants import N_A_NUMERIC


class AnalysisTests(unittest.TestCase):
    """Tests for deterministic analysis helper behavior."""

    def test_embedding_similarity_and_drift_helpers_validate_inputs(self) -> None:
        """Validate embedding conversion, similarity, and drift helper paths.

        Returns:
            None: Assertions verify expected helper behavior.

        Raises:
            AssertionError: An analysis helper returns an unexpected result.
        """
        embedding = np.ones(2048, dtype=np.float32)
        converted = analysis._embedding_tensor_to_numpy(embedding)
        self.assertEqual(converted.shape, (2048,))

        with self.assertRaisesRegex(ValueError, "2048-d"):
            analysis._embedding_tensor_to_numpy(np.ones(3, dtype=np.float32))

        cosine, percent = analysis._cosine_similarity_percent(embedding, embedding)
        self.assertAlmostEqual(cosine, 1.0)
        self.assertAlmostEqual(percent, 100.0)
        with self.assertRaisesRegex(ValueError, "norm is zero"):
            analysis._cosine_similarity_percent(
                np.zeros(2048, dtype=np.float32),
                embedding,
            )

        self.assertEqual(analysis._voice_drift_level(2.0), "stable")
        self.assertEqual(analysis._voice_drift_level(5.0), "expressive")
        self.assertEqual(analysis._voice_drift_level(8.0), "weak")
        self.assertEqual(analysis._voice_drift_level(12.0), "wrong")

    def test_traits_and_assessment_cover_speech_and_empty_audio_paths(self) -> None:
        """Check trait and assessment output for empty voice extraction data.

        Returns:
            None: Assertions verify fallback assessment behavior.

        Raises:
            AssertionError: Trait or assessment output changes unexpectedly.
        """
        metrics = {
            "duration_s": 1.0,
            "pitch_extraction_ok": False,
            "pitch_mean_hz": N_A_NUMERIC,
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
        self.assertEqual(set(traits.values()), {0.0})
        assessment = analysis.generate_assessment(metrics, traits)
        self.assertIn("No voicings found.", assessment[1])
        self.assertIn("Mean pitch could not be determined", assessment[3])
        self.assertIn("high pause ratio", assessment[-1].lower())

    @mock.patch("celune.analysis.default_loader", return_value=None)
    def test_loose_reference_embeddings_are_discovered_without_bundle(
        self, _default_loader: mock.Mock
    ) -> None:
        """Verify loose packaged embeddings remain available without CEVOICE.

        Args:
            _default_loader: A mock default loader.

        Returns:
            None: This function tests the loader to see if embeddings can be found without a CEVOICE bundle.
        """
        self.assertEqual(
            analysis._available_reference_voices(),
            ["balanced", "bold", "calm", "upbeat"],
        )

    @mock.patch("celune.analysis.default_loader", return_value=None)
    @mock.patch("celune.analysis.torch.load")
    def test_loose_reference_embedding_loads_without_bundle(
        self, torch_load: mock.Mock, _default_loader: mock.Mock
    ) -> None:
        """Verify loose packaged embeddings still load by derived voice path.

        Args:
            torch_load: A mock implementation of torch.load().
            _default_loader: A mock default loader.

        Returns:
            None: This function tests the loader if embeddings can be loaded correctly without a CEVOICE bundle.
        """
        torch_load.return_value = np.ones(2048, dtype=np.float32)

        embedding = analysis._load_reference_embedding("balanced")

        self.assertEqual(embedding.shape, (2048,))
        torch_load.assert_called_once()


class ChromaTests(unittest.TestCase):
    """Tests for pure RGB glow helper behavior."""

    def test_pure_glow_helpers_process_audio_without_devices(self) -> None:
        """Exercise glow math without connecting to RGB devices.

        Returns:
            None: Assertions verify pure helper outputs.

        Raises:
            AssertionError: A glow helper returns an unexpected value.
        """
        stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mono = AudioRGBGlow._to_mono(stereo)
        self.assertTrue(np.array_equal(mono, np.array([0.5, 0.5], dtype=np.float32)))

        fixed = AudioRGBGlow._fix_color_rendering((255, 255, 255))
        self.assertEqual(len(fixed), 3)
        self.assertLessEqual(max(fixed), 255)

        glow = object.__new__(AudioRGBGlow)
        glow.input_gain = 4.0
        glow.gamma = 1.4
        self.assertEqual(glow._speech_level(np.zeros((0, 2), dtype=np.float32)), 0.0)
        self.assertGreater(glow._speech_level(stereo), 0.0)
