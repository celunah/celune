# SPDX-License-Identifier: Apache-2.0
"""Tests for score-guided F0 melody conditioning."""

import pytest
import numpy as np

from celune.backends.tts.soulxsinger import _ScoreEvent, _natural_f0_contour


def test_natural_f0_contour_recenters_each_note_and_keeps_expression() -> None:
    """Preserve within-note pitch movement while centering notes on CESN pitches."""
    events = (
        _ScoreEvent("c4", 1, 1, False),
        _ScoreEvent("d4", 1, 1, False),
        _ScoreEvent(None, 1, 1, True),
    )
    observed_f0 = np.array(
        [250.0, 260.0, 270.0, 0.0, 300.0, 310.0, 320.0, 0.0, 440.0, 440.0, 0.0, 0.0]
    )

    contour = np.array(
        [
            float(value)
            for value in _natural_f0_contour(
                events,
                ["0.04", "0.04", "0.04"],
                ["60", "62", "0"],
                observed_f0,
                sample_rate=100,
                hop_size=1,
            ).split()
        ]
    )

    assert len(contour) == 12
    assert np.median(contour[:3]) == pytest.approx(440.0 * 2 ** ((60 - 69) / 12))
    assert np.median(contour[4:7]) == pytest.approx(440.0 * 2 ** ((62 - 69) / 12))
    assert contour[0] < contour[1] < contour[2]
    assert contour[3] == 0.0
    assert np.all(contour[8:] == 0.0)


def test_natural_f0_contour_falls_back_for_unvoiced_notes() -> None:
    """Keep a written note pitched when the teacher extractor misses it entirely."""
    event = (_ScoreEvent("g3", 1, 1, False),)

    contour = _natural_f0_contour(
        event,
        ["0.04"],
        ["55"],
        np.zeros(4, dtype=np.float32),
        sample_rate=100,
        hop_size=1,
    )

    expected_frequency = 440.0 * 2 ** ((55 - 69) / 12)
    assert [float(value) for value in contour.split()] == pytest.approx(
        [expected_frequency] * 4
    )
