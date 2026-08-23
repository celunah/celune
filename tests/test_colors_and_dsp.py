# SPDX-License-Identifier: Apache-2.0
"""Tests for color and DSP helpers."""

from typing import cast
from unittest import mock

import numpy as np
import pytest
from celune import colors, dsp
from celune.constants import UtteranceLoudnessTier
from celune.exceptions import BadAudioError, AudioMismatchError

from .support import CeluneTestCase


class TestColor(CeluneTestCase):
    """Tests for generated Celune theme palettes."""

    def tearDown(self) -> None:
        """Reset shared audio-reactivity state after each color test."""
        colors.configure_theme()

    def test_default_and_custom_theme_palettes_are_configured(self) -> None:
        """Verify default palettes and contrast-adjusted custom palettes.

        Raises:
            AssertionError: Theme behavior changes unexpectedly.
        """
        colors.configure_theme()
        assert colors.THEME.primary == "#cebaff"
        assert colors.THEME_LIGHT.background == "#deceff"

        colors.configure_theme("#101010", "#222222")
        assert colors.THEME.background == "#101010"
        assert (
            colors.contrast_ratio(
                colors.THEME.primary,
                cast(str, colors.THEME.background),
            )
            >= 4.5
        )
        assert colors.SEVERITY_COLORS["celune"]["info"] == colors.THEME.primary
        assert colors.SEVERITY_COLORS["celune"]["sleeping"] == "#9c88ce"
        assert colors.SEVERITY_COLORS["celune_light"]["sleeping"] == "#6d5f90"

        colors.configure_theme("#101010", "#222222", "#8866cc")
        assert colors.SEVERITY_COLORS["celune"]["sleeping"] == "#8866cc"
        assert colors.SEVERITY_COLORS["celune_light"]["sleeping"] == "#7558af"


class TestDsp(CeluneTestCase):
    """Tests for lightweight DSP helpers."""

    def tearDown(self) -> None:
        """Reset shared audio-reactivity state after each DSP test."""
        dsp._SIGNAL_CACHE.clear()

    def test_make_stereo_and_resampling_validate_audio(self) -> None:
        """Verify stereo conversion and sample-rate validation paths.

        Raises:
            AssertionError: DSP behavior changes unexpectedly.
        """
        mono = np.array([0.0, 1.0], dtype=np.float32)
        stereo = dsp.make_stereo(mono)
        assert stereo.shape == (2, 2)
        assert np.array_equal(stereo[:, 0], mono)

        with pytest.raises(AudioMismatchError):
            dsp.make_stereo(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(BadAudioError):
            dsp.resample_audio(stereo, 0)
        assert dsp.resample_audio(stereo, 48000).shape == (2, 2)

    def test_soften_split_and_silence_detection(self) -> None:
        """Verify softening, chunk splitting, and loudness tiers.

        Raises:
            AssertionError: DSP output changes unexpectedly.
        """
        audio = np.ones((10, 2), dtype=np.float32)
        softened = dsp.soften(audio.copy(), sr=10, duration=0.2, start_gain=0.5)
        assert float(softened[0, 0]) == pytest.approx(0.5)
        chunks = list(dsp.split(np.zeros((20, 2), dtype=np.float32), 10, 5))
        assert [len(chunk) for chunk in chunks] == [4, 4, 4, 4, 4]

        silent = np.zeros((4, 2), dtype=np.float32)
        suspicious = np.full((4, 2), 0.005, dtype=np.float32)
        normal = np.full((4, 2), 0.1, dtype=np.float32)
        assert dsp.is_silent_utterance(silent) == (True, UtteranceLoudnessTier.SILENT)
        assert dsp.is_silent_utterance(suspicious) == (
            True,
            UtteranceLoudnessTier.SUSPICIOUS,
        )
        assert dsp.is_silent_utterance(normal) == (False, UtteranceLoudnessTier.NORMAL)

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

        assert audio.shape == (280, 2)
        assert audio.dtype == np.float32
        assert np.all(audio[:leading_samples] == 0)
        assert np.all(audio[-trailing_samples:] == 0)

        audible = audio[leading_samples:-trailing_samples]
        rms = np.sqrt(np.mean(np.square(audible), dtype=np.float64))
        expected_rms = 10 ** (target_dbfs / 20)
        assert float(rms) == pytest.approx(expected_rms)
        assert float(np.max(np.abs(audio))) <= 0.95

        with pytest.raises(BadAudioError):
            dsp.pad_note((), duration=0.25, sample_rate=sample_rate)

    def test_transpose_frequencies_uses_equal_tempered_intervals(self) -> None:
        """Verify semitone transposition without waveform pitch shifting."""
        assert dsp._transpose_frequencies((440.0,), 0) == (440.0,)
        assert dsp._transpose_frequencies((440.0,), 12)[0] == pytest.approx(880.0)
        assert dsp._transpose_frequencies((440.0,), -12)[0] == pytest.approx(220.0)

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

        assert readiness_first is readiness_second
        assert sleeping_first is sleeping_second
        assert working_first is working_second
        assert error_first is error_second
        assert not readiness_first.flags.writeable
        assert error_first.shape == (4, 2)
        assert float(np.max(np.abs(error_first))) == pytest.approx(1.0)
        assert load.call_count == 1
        assert generate.call_count == 3

    def test_reverb_strength_reduces_dry_level_to_preserve_headroom(self) -> None:
        """Verify stronger reverb keeps the combined dry/wet gain under control."""
        reverb = dsp.StreamingPedalboardReverb()

        reverb.strength = 0.0
        reverb.update_params()
        assert reverb.reverb.wet_level == pytest.approx(0.0)
        assert reverb.reverb.dry_level == pytest.approx(1.0)

        reverb.strength = 1.0
        reverb.update_params()
        assert reverb.reverb.wet_level == pytest.approx(0.16)
        assert reverb.reverb.dry_level == pytest.approx(0.84)
        assert reverb.reverb.wet_level + reverb.reverb.dry_level <= 1.0
