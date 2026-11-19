# SPDX-License-Identifier: Apache-2.0
"""Scoring and persistence for the Cel Voice Score (CVS)."""

import math
import hashlib
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .paths import app_data_dir

CVS_VOICE_WEIGHT = 0.40
CVS_WER_WEIGHT = 0.30
CVS_LATENCY_WEIGHT = 0.15
CVS_REALTIME_WEIGHT = 0.15
CVS_DATABASE_NAME = "cvs_aggregates.db"


@dataclass(frozen=True, slots=True)
class CVSScore:
    """One CVS result and its four normalized component scores."""

    score: Optional[float]
    voice_match: Optional[float]
    wer: Optional[float]
    latency: Optional[float]
    realtime_speed: Optional[float]


def voice_match_score(similarity_percent: Optional[float]) -> Optional[float]:
    """Score cosine similarity, reaching full marks at 97 percent."""
    value = _finite_nonnegative(similarity_percent)
    if value is None:
        return None
    return _clamp((value - 87.0) * 10.0)


def wer_score(wer_percent: Optional[float]) -> Optional[float]:
    """Score word error rate expressed in percentage points."""
    value = _finite_nonnegative(wer_percent)
    if value is None:
        return None
    return max(0.0, 100.0 - (5.0 * value))


def latency_score(latency_ms: Optional[float]) -> Optional[float]:
    """Score time to first usable audio, with a 500 ms full-score target."""
    value = _finite_nonnegative(latency_ms)
    if value is None:
        return None
    excess_ms = max(0.0, value - 500.0)
    return _clamp(100.0 - (excess_ms / 250.0 * 5.0))


def realtime_speed_score(speed: Optional[float]) -> Optional[float]:
    """Linearly score synthesis speed from zero to 1.2 times realtime."""
    value = _finite_nonnegative(speed)
    if value is None:
        return None
    return _clamp(value / 1.2 * 100.0)


def calculate_cvs(
    similarity_percent: Optional[float],
    wer_percent: Optional[float],
    latency_ms: Optional[float],
    realtime_speed: Optional[float],
) -> CVSScore:
    """Calculate the weighted Cel Voice Score when every component exists.

    Args:
        similarity_percent: Speaker cosine similarity on a 0 to 100 scale.
        wer_percent: Word error rate in percentage points.
        latency_ms: Request-to-first-audio latency in milliseconds.
        realtime_speed: Generated audio duration divided by generation time.

    Returns:
        CVSScore: Component scores and the weighted total, or a missing total
            when any required component could not be measured.
    """
    voice_score = voice_match_score(similarity_percent)
    word_score = wer_score(wer_percent)
    response_score = latency_score(latency_ms)
    speed_score = realtime_speed_score(realtime_speed)
    total = None
    if (
        voice_score is not None
        and word_score is not None
        and response_score is not None
        and speed_score is not None
    ):
        total = (
            CVS_VOICE_WEIGHT * voice_score
            + CVS_WER_WEIGHT * word_score
            + CVS_LATENCY_WEIGHT * response_score
            + CVS_REALTIME_WEIGHT * speed_score
        )
    return CVSScore(
        score=total,
        voice_match=voice_score,
        wer=word_score,
        latency=response_score,
        realtime_speed=speed_score,
    )


def word_error_rate_percent(reference: str, hypothesis: str) -> Optional[float]:
    """Return whitespace-token WER after case and punctuation normalization.

    Args:
        reference: Expected spoken text.
        hypothesis: Text recognized from the generated speech.

    Returns:
        Optional[float]: Word errors per 100 reference words, or ``None`` when
            the reference contains no words.
    """
    counts = word_error_counts(reference, hypothesis)
    if counts is None:
        return None
    errors, reference_word_count = counts
    return 100.0 * errors / reference_word_count


def word_error_counts(
    reference: str,
    hypothesis: str,
) -> Optional[tuple[int, int]]:
    """Return minimum word edits and reference word count for WER.

    Args:
        reference: Expected spoken text.
        hypothesis: Text recognized from the generated speech.

    Returns:
        Optional[tuple[int, int]]: Edit count and reference word count, or
            ``None`` when the reference cannot be measured as whitespace words.
    """
    if not supports_word_error_rate(reference):
        return None
    reference_words = _normalized_words(reference)
    hypothesis_words = _normalized_words(hypothesis)
    if not reference_words:
        return None

    previous = list(range(len(hypothesis_words) + 1))
    for ref_index, reference_word in enumerate(reference_words, start=1):
        current = [ref_index]
        for hyp_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = int(reference_word != hypothesis_word)
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1], len(reference_words)


def supports_word_error_rate(reference: str) -> bool:
    """Return whether whitespace tokenization can measure this reference."""
    return bool(_normalized_words(reference)) and not _contains_unspaced_script(
        reference
    )


def language_key(language: Optional[str]) -> str:
    """Return a compact language tag for one utterance score."""
    if not isinstance(language, str) or not language.strip():
        return "unknown"
    normalized = language.strip().replace("_", "-").split("-", maxsplit=1)[0]
    if normalized.casefold() in {"auto", "unknown"}:
        return "unknown"
    return normalized.casefold()


def record_cvs_utterance(
    utterance_id: str,
    backend_id: str,
    language: str,
    cvs: CVSScore,
    *,
    similarity_percent: Optional[float],
    wer_percent: Optional[float],
    latency_ms: Optional[float],
    realtime_speed: Optional[float],
    database_path: Optional[Path] = None,
) -> tuple[Optional[float], int]:
    """Persist one result and return its backend-wide mean and sample count.

    The database stores a one-way hash of ``utterance_id`` rather than a path,
    transcript, or recognized words.
    """
    resolved_database = database_path or (app_data_dir(create=True) / CVS_DATABASE_NAME)
    resolved_database.parent.mkdir(parents=True, exist_ok=True)
    utterance_key = hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()

    with sqlite3.connect(resolved_database, timeout=30.0) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cvs_utterances (
                utterance_key TEXT PRIMARY KEY,
                backend_id TEXT NOT NULL,
                language TEXT NOT NULL,
                cvs_score REAL,
                voice_match_score REAL,
                wer_score REAL,
                latency_score REAL,
                realtime_speed_score REAL,
                similarity_percent REAL,
                wer_percent REAL,
                latency_ms REAL,
                realtime_speed REAL,
                recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO cvs_utterances (
                utterance_key, backend_id, language, cvs_score,
                voice_match_score, wer_score, latency_score,
                realtime_speed_score, similarity_percent, wer_percent,
                latency_ms, realtime_speed, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(utterance_key) DO UPDATE SET
                backend_id=excluded.backend_id,
                language=excluded.language,
                cvs_score=excluded.cvs_score,
                voice_match_score=excluded.voice_match_score,
                wer_score=excluded.wer_score,
                latency_score=excluded.latency_score,
                realtime_speed_score=excluded.realtime_speed_score,
                similarity_percent=excluded.similarity_percent,
                wer_percent=excluded.wer_percent,
                latency_ms=excluded.latency_ms,
                realtime_speed=excluded.realtime_speed,
                recorded_at=CURRENT_TIMESTAMP
            """,
            (
                utterance_key,
                backend_id,
                language,
                cvs.score,
                cvs.voice_match,
                cvs.wer,
                cvs.latency,
                cvs.realtime_speed,
                similarity_percent,
                wer_percent,
                latency_ms,
                realtime_speed,
            ),
        )
        aggregate = connection.execute(
            """
            SELECT AVG(cvs_score), COUNT(cvs_score)
            FROM cvs_utterances
            WHERE backend_id = ?
            """,
            (backend_id,),
        ).fetchone()

    if aggregate is None:
        return None, 0
    mean_value, count_value = aggregate
    mean = float(mean_value) if mean_value is not None else None
    return mean, int(count_value)


def _normalized_words(text: str) -> list[str]:
    """Normalize case and punctuation before whitespace word comparison."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words: list[str] = []
    current: list[str] = []
    apostrophes = {"'", "’", "‘", "ʼ"}
    for index, character in enumerate(normalized):
        if character.isalnum():
            current.append(character)
        elif character in apostrophes:
            previous = normalized[index - 1] if index > 0 else ""
            following = normalized[index + 1] if index + 1 < len(normalized) else ""
            if previous.isalnum() and following.isalnum():
                continue
            if current:
                words.append("".join(current))
                current.clear()
        elif character in {",", "."}:
            previous = normalized[index - 1] if index > 0 else ""
            following = normalized[index + 1] if index + 1 < len(normalized) else ""
            if previous.isdigit() and following.isdigit():
                continue
            if current:
                words.append("".join(current))
                current.clear()
        elif current:
            words.append("".join(current))
            current.clear()
    if current:
        words.append("".join(current))
    return words


def _contains_unspaced_script(text: str) -> bool:
    """Check scripts that need language-specific WER word segmentation."""
    script_ranges = (
        (0x0E00, 0x0E7F),  # Thai
        (0x0E80, 0x0EFF),  # Lao
        (0x1000, 0x109F),  # Myanmar
        (0x1780, 0x17FF),  # Khmer
        (0x3040, 0x30FF),  # Hiragana and Katakana
        (0x3400, 0x9FFF),  # CJK ideographs
    )
    return any(
        start <= ord(character) <= end
        for character in text
        for start, end in script_ranges
    )


def _finite_nonnegative(value: Optional[float]) -> Optional[float]:
    """Return one finite nonnegative numeric metric if it is usable."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        return None
    return float(value)


def _clamp(value: float) -> float:
    """Clamp a component score to the published 0 to 100 range."""
    return min(100.0, max(0.0, value))
