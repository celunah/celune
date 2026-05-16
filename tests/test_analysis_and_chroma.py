# SPDX-License-Identifier: MIT
"""Tests for pure analysis helpers and RGB glow math."""

import unittest

import numpy as np

from celune import analysis
from celune.chroma import AudioRGBGlow


class AnalysisTests(unittest.TestCase):
    def test_embedding_similarity_and_drift_helpers_validate_inputs(self) -> None:
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
        metrics = {
            "duration_s": 1.0,
            "pitch_extraction_ok": False,
            "pitch_mean_hz": float("nan"),
            "pitch_variance": float("nan"),
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


class ChromaTests(unittest.TestCase):
    def test_pure_glow_helpers_process_audio_without_devices(self) -> None:
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
