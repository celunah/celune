#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect intonation of voices."""

import os
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

type DualIntonation = tuple[
    tuple[np.float64, np.float64, np.float64, np.float64, np.float64],
    tuple[np.float64, np.float64, np.float64, np.float64, np.float64],
]


def inspect(audio: Path, ref_audio: Path, percentiles: float) -> DualIntonation:
    """Inspect the audio's intonation."""
    y, sr = librosa.load(audio)
    x, rsr = librosa.load(ref_audio)

    f0, voiced, _ = librosa.pyin(
        y,
        fmin=120,
        fmax=300,
        sr=sr,
        frame_length=2048,
        hop_length=int(sr / 100),
        max_transition_rate=5.0,
    )

    rf0, ref_voiced, _ = librosa.pyin(
        x,
        fmin=120,
        fmax=300,
        sr=rsr,
        frame_length=2048,
        hop_length=int(rsr / 100),
        max_transition_rate=5.0,
    )

    f0 = smooth_f0(f0, voiced, window=5)
    rf0 = smooth_f0(rf0, ref_voiced, window=5)

    print_contour(f0, voiced, "Input contour")
    print_contour(rf0, ref_voiced, "Reference contour")

    f0_hop = int(sr / 100)
    ref_hop = int(rsr / 100)

    pitch_step = p95_pitch_rate(f0, voiced, int(sr), f0_hop)
    ref_pitch_step = p95_pitch_rate(rf0, ref_voiced, int(rsr), ref_hop)

    f0_voiced = f0[voiced & np.isfinite(f0)]
    rf0_voiced = rf0[ref_voiced & np.isfinite(rf0)]

    if f0_voiced.size == 0 or rf0_voiced.size == 0:
        raise ValueError("no voiced segments found")

    median = np.median(f0_voiced)
    ref_median = np.median(rf0_voiced)

    coverage = percentiles
    tail = (1 - coverage) / 2

    pl, ph = np.percentile(
        f0_voiced,
        [tail * 100, (1 - tail) * 100],
    )
    rpl, rph = np.percentile(
        rf0_voiced,
        [tail * 100, (1 - tail) * 100],
    )

    span_semitones = 12 * np.log2(ph / pl)
    ref_span_semitones = 12 * np.log2(rph / rpl)

    return (pl, ph, span_semitones, median, pitch_step), (
        rpl,
        rph,
        ref_span_semitones,
        ref_median,
        ref_pitch_step,
    )


def smooth_f0(
    f0: np.ndarray,
    voiced: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Median-smooth F0 in semitones within each voiced run."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")

    valid = voiced & np.isfinite(f0) & (f0 > 0)
    smoothed = np.full(f0.shape, np.nan)
    log_f0 = np.full(f0.shape, np.nan)
    log_f0[valid] = 12 * np.log2(f0[valid])

    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    half = window // 2

    for start, end in zip(starts, ends):
        run = log_f0[start:end]
        padded = np.pad(run, (half, half), mode="edge")

        for i in range(len(run)):
            smoothed[start + i] = np.median(padded[i : i + window])

    return np.exp2(smoothed / 12)


def p95_pitch_rate(
    f0: np.ndarray,
    voiced: np.ndarray,
    sr: int,
    hop_length: int,
) -> np.float64:
    """Return the 95th percentile for a voiced segment."""
    valid = (
        voiced[:-1]
        & voiced[1:]
        & np.isfinite(f0[:-1])
        & np.isfinite(f0[1:])
        & (f0[:-1] > 0)
        & (f0[1:] > 0)
    )
    if not np.any(valid):
        return np.float64(0)

    left = f0[:-1][valid]
    right = f0[1:][valid]
    steps = np.abs(12 * np.log2(right / left))
    if steps.size == 0:
        raise ValueError("no adjacent voiced frames found")

    seconds_per_frame = hop_length / sr
    rates = steps / seconds_per_frame
    return np.percentile(rates, 95)


def print_contour(
    f0: np.ndarray,
    voiced: np.ndarray,
    label: str,
    width: int = 80,
    semitone_limit: float = 6.0,
) -> None:
    """Print a median-centered pitch contour using block characters."""
    valid = voiced & np.isfinite(f0) & (f0 > 0)
    if not np.any(valid):
        print(f"{label}: no voiced frames")
        return

    median = np.median(f0[valid])
    relative = np.full(f0.shape, np.nan)
    relative[valid] = 12 * np.log2(f0[valid] / median)

    bins = np.array_split(np.arange(len(f0)), min(width, len(f0)))
    values: list[float | None] = []
    for indices in bins:
        bin_values = relative[indices]
        bin_values = bin_values[np.isfinite(bin_values)]
        values.append(float(np.median(bin_values)) if bin_values.size else None)

    rows = 13
    grid = [[" "] * len(values) for _ in range(rows)]
    point_rows: list[int | None] = []
    for column, value in enumerate(values):
        if value is not None:
            row = round(
                (semitone_limit - np.clip(value, -semitone_limit, semitone_limit))
                / (2 * semitone_limit)
                * (rows - 1)
            )
            grid[row][column] = "█"
            point_rows.append(row)
        else:
            point_rows.append(None)

    # Join neighboring voiced bins into a visible trace, but preserve pauses.
    for column in range(1, len(point_rows)):
        previous = point_rows[column - 1]
        current = point_rows[column]
        if previous is not None and current is not None:
            for row in range(min(previous, current), max(previous, current) + 1):
                grid[row][column] = "█"

    print(label)
    for row, cells in enumerate(grid):
        pitch = semitone_limit - row * (2 * semitone_limit / (rows - 1))
        print(f"{pitch:+5.1f} |{''.join(cells)}")
    print("      +" + "-" * len(values))
    print("       time ->")


def validate_file(f) -> str:
    """Validate whether this is a file."""
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"{f} does not exist")

    try:
        sf.info(f)
    except sf.LibsndfileError as sfe:
        raise argparse.ArgumentTypeError(f"{f} is not a supported audio file") from sfe

    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect intonation of a voice sample")
    parser.add_argument(
        "-i",
        "--input",
        dest="filename",
        required=True,
        type=validate_file,
        help="input file",
        metavar="FILE",
    )
    parser.add_argument(
        "-r",
        "--reference",
        dest="reference_filename",
        required=True,
        type=validate_file,
        help="reference file",
        metavar="FILE",
    )
    args = parser.parse_args()
    try:
        target, reference = inspect(
            Path(args.filename), Path(args.reference_filename), 0.5
        )
        lo, hi, semi, med, step = target
        rlo, rhi, ref_semi, ref_med, ref_step = reference

        pitch_offset = abs(12 * np.log2(med / ref_med))
        span_difference = semi - ref_semi
        step_offset = abs(step - ref_step)

        grades = [
            (-0.5, 1.0, "S"),
            (-1.0, 2.0, "A"),
            (-1.5, 3.0, "B"),
            (-2.0, 4.0, "C"),
            (-2.5, 5.0, "D"),
        ]

        for lower, upper, grade in grades:
            if lower <= span_difference <= upper:
                rank = grade
                break
        else:
            rank = "F"

        print(f"Median pitch difference: {pitch_offset:+.2f} sem")
        print(f"Intonation span difference: {span_difference:+.2f} sem")
        print(f"Intonation step difference: {step_offset:+.2f} sem/s")

        print(f"Low: {lo:.2f} Hz (reference: {rlo:.2f} Hz)")
        print(f"High: {hi:.2f} Hz (reference: {rhi:.2f} Hz)")
        print(f"Semitone range: {semi:.2f} sem (reference: {ref_semi:.2f} sem)")
        print(f"Median: {med:.2f} Hz (reference: {ref_med:.2f} Hz)")
        print(f"Pitch step: {step:.2f} sem/s (reference: {ref_step:.2f} sem/s)")

        print(f"Rank: {rank}")
    except ValueError as exc:
        parser.error(str(exc))
