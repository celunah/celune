# SPDX-License-Identifier: MIT
"""Tests for color and DSP helpers."""

import unittest

import numpy as np

from celune import colors
from celune import dsp
from celune.constants import UtteranceLoudnessTier
from celune.exceptions import AudioMismatchError, BadAudioError


class ColorTests(unittest.TestCase):
    def tearDown(self) -> None:
        colors.configure_theme()

    def test_default_and_custom_theme_palettes_are_configured(self) -> None:
        colors.configure_theme()
        self.assertEqual(colors.THEME.primary, "#cebaff")
        self.assertEqual(colors.THEME_LIGHT.background, "#ece8ff")

        colors.configure_theme("#101010", "#222222")
        self.assertEqual(colors.THEME.background, "#101010")
        self.assertGreaterEqual(
            colors._contrast_ratio(colors.THEME.primary, colors.THEME.background),
            4.5,
        )
        self.assertEqual(
            colors.SEVERITY_COLORS["celune"]["info"],
            colors.THEME.primary,
        )


class DspTests(unittest.TestCase):
    def test_make_stereo_and_resampling_validate_audio(self) -> None:
        mono = np.array([0.0, 1.0], dtype=np.float32)
        stereo = dsp._make_stereo(mono)
        self.assertEqual(stereo.shape, (2, 2))
        self.assertTrue(np.array_equal(stereo[:, 0], mono))

        with self.assertRaises(AudioMismatchError):
            dsp._make_stereo(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(BadAudioError):
            dsp._resample_audio(stereo, 0)
        self.assertEqual(dsp._resample_audio(stereo, 48000).shape, (2, 2))

    def test_soften_split_and_silence_detection(self) -> None:
        audio = np.ones((10, 2), dtype=np.float32)
        softened = dsp._soften(audio.copy(), sr=10, duration=0.2, start_gain=0.5)
        self.assertAlmostEqual(float(softened[0, 0]), 0.5)
        chunks = list(dsp._split(np.zeros((20, 2), dtype=np.float32), 10, 5))
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 4, 4, 4])

        silent = np.zeros((4, 2), dtype=np.float32)
        suspicious = np.full((4, 2), 0.005, dtype=np.float32)
        normal = np.full((4, 2), 0.1, dtype=np.float32)
        self.assertEqual(
            dsp.is_silent_utterance(silent),
            (True, UtteranceLoudnessTier.SILENT),
        )
        self.assertEqual(
            dsp.is_silent_utterance(suspicious),
            (True, UtteranceLoudnessTier.SUSPICIOUS),
        )
        self.assertEqual(
            dsp.is_silent_utterance(normal),
            (False, UtteranceLoudnessTier.NORMAL),
        )
