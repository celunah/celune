# SPDX-License-Identifier: MIT
"""Tests for color and DSP helpers."""

from typing import cast
from unittest import TestCase, mock

import numpy as np

from celune import colors, dsp
from celune.constants import UtteranceLoudnessTier
from celune.exceptions import AudioMismatchError, BadAudioError


class ColorTests(TestCase):
    """Tests for generated Celune theme palettes."""

    def tearDown(self) -> None:
        colors.configure_theme()

    def test_default_and_custom_theme_palettes_are_configured(self) -> None:
        """Verify default palettes and contrast-adjusted custom palettes.

        Raises:
            AssertionError: Theme behavior changes unexpectedly.
        """
        colors.configure_theme()
        self.assertEqual(colors.THEME.primary, "#cebaff")
        self.assertEqual(colors.THEME_LIGHT.background, "#ece8ff")

        colors.configure_theme("#101010", "#222222")
        self.assertEqual(colors.THEME.background, "#101010")
        self.assertGreaterEqual(
            colors.contrast_ratio(
                colors.THEME.primary,
                cast(str, colors.THEME.background),
            ),
            4.5,
        )
        self.assertEqual(
            colors.SEVERITY_COLORS["celune"]["info"],
            colors.THEME.primary,
        )
        self.assertEqual(colors.SEVERITY_COLORS["celune"]["sleeping"], "#9c88ce")
        self.assertEqual(colors.SEVERITY_COLORS["celune_light"]["sleeping"], "#6d5f90")

        colors.configure_theme("#101010", "#222222", "#8866cc")
        self.assertEqual(colors.SEVERITY_COLORS["celune"]["sleeping"], "#8866cc")
        self.assertEqual(colors.SEVERITY_COLORS["celune_light"]["sleeping"], "#7558af")


class DspTests(TestCase):
    """Tests for lightweight DSP helpers."""

    def tearDown(self) -> None:
        dsp._SIGNAL_CACHE.clear()

    def test_make_stereo_and_resampling_validate_audio(self) -> None:
        """Verify stereo conversion and sample-rate validation paths.

        Raises:
            AssertionError: DSP behavior changes unexpectedly.
        """
        mono = np.array([0.0, 1.0], dtype=np.float32)
        stereo = dsp.make_stereo(mono)
        self.assertEqual(stereo.shape, (2, 2))
        self.assertTrue(np.array_equal(stereo[:, 0], mono))

        with self.assertRaises(AudioMismatchError):
            dsp.make_stereo(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(BadAudioError):
            dsp.resample_audio(stereo, 0)
        self.assertEqual(dsp.resample_audio(stereo, 48000).shape, (2, 2))

    def test_soften_split_and_silence_detection(self) -> None:
        """Verify softening, chunk splitting, and loudness tiers.

        Raises:
            AssertionError: DSP output changes unexpectedly.
        """
        audio = np.ones((10, 2), dtype=np.float32)
        softened = dsp.soften(audio.copy(), sr=10, duration=0.2, start_gain=0.5)
        self.assertAlmostEqual(float(softened[0, 0]), 0.5)
        chunks = list(dsp.split(np.zeros((20, 2), dtype=np.float32), 10, 5))
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

    def test_pad_generates_rms_normalized_stereo_with_silence(self) -> None:
        """Verify soft pad synthesis, RMS normalization, and silence padding."""
        sample_rate = 1000
        leading_samples = 10
        trailing_samples = 20
        target_dbfs = -36.0

        audio = dsp.pad_note(
            (261.63, 329.63),
            duration=0.25,
            sample_rate=sample_rate,
            target_rms_dbfs=target_dbfs,
            attack_seconds=0.05,
            release_seconds=0.05,
            leading_silence_seconds=leading_samples / sample_rate,
            trailing_silence_seconds=trailing_samples / sample_rate,
        )

        self.assertEqual(audio.shape, (280, 2))
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.all(audio[:leading_samples] == 0))
        self.assertTrue(np.all(audio[-trailing_samples:] == 0))

        audible = audio[leading_samples:-trailing_samples]
        rms = np.sqrt(np.mean(np.square(audible), dtype=np.float64))
        expected_rms = 10 ** (target_dbfs / 20)
        self.assertAlmostEqual(float(rms), expected_rms, places=6)
        self.assertLessEqual(float(np.max(np.abs(audio))), 0.95)

        with self.assertRaises(BadAudioError):
            dsp.pad_note((), duration=0.25, sample_rate=sample_rate)

    def test_transpose_frequencies_uses_equal_tempered_intervals(self) -> None:
        """Verify semitone transposition without waveform pitch shifting."""
        self.assertEqual(dsp._transpose_frequencies((440.0,), 0), (440.0,))
        self.assertAlmostEqual(
            dsp._transpose_frequencies((440.0,), 12)[0],
            880.0,
        )
        self.assertAlmostEqual(
            dsp._transpose_frequencies((440.0,), -12)[0],
            220.0,
        )

    def test_ui_signal_helpers_reuse_cached_audio(self) -> None:
        """Verify UI signal helpers reuse immutable cached buffers."""
        base = np.ones((4, 2), dtype=np.float32)

        with (
            mock.patch("celune.dsp._load_readiness_signal", return_value=base) as load,
            mock.patch(
                "celune.dsp.pad_note",
                return_value=base,
            ) as generate,
        ):
            readiness_first = dsp.readiness_signal()
            readiness_second = dsp.readiness_signal()
            sleeping_first = dsp.sleeping_signal()
            sleeping_second = dsp.sleeping_signal()
            working_first = dsp.working_signal()
            working_second = dsp.working_signal()
            error_first = dsp.error_signal()
            error_second = dsp.error_signal()

        self.assertIs(readiness_first, readiness_second)
        self.assertIs(sleeping_first, sleeping_second)
        self.assertIs(working_first, working_second)
        self.assertIs(error_first, error_second)
        self.assertFalse(readiness_first.flags.writeable)
        self.assertEqual(error_first.shape, (4, 2))
        self.assertAlmostEqual(float(np.max(np.abs(error_first))), 1.0)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(generate.call_count, 3)

    def test_reverb_strength_reduces_dry_level_to_preserve_headroom(self) -> None:
        """Verify stronger reverb keeps the combined dry/wet gain under control."""
        reverb = dsp.StreamingPedalboardReverb()

        reverb.strength = 0.0
        reverb.update_params()
        self.assertAlmostEqual(reverb.reverb.wet_level, 0.0)
        self.assertAlmostEqual(reverb.reverb.dry_level, 1.0)

        reverb.strength = 1.0
        reverb.update_params()
        self.assertAlmostEqual(reverb.reverb.wet_level, 0.16)
        self.assertAlmostEqual(reverb.reverb.dry_level, 0.84)
        self.assertLessEqual(
            reverb.reverb.wet_level + reverb.reverb.dry_level,
            1.0,
        )
