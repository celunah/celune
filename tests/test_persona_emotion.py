# SPDX-License-Identifier: MIT
"""Tests for Persona emotion blending."""

from collections.abc import Sequence
from unittest import TestCase

import numpy as np

from celune.persona.emotion import (
    GOEMOTIONS_LABELS,
    EmotionAnalysis,
    EmotionPrediction,
    PersonaEmotionAnalyzer,
)


class StubPersonaEmotionAnalyzer(PersonaEmotionAnalyzer):
    """Analyzer with deterministic embeddings and labels for emotion tests."""

    def __init__(
        self,
        analyses_by_text: dict[str, EmotionAnalysis],
        prototypes: dict[str, np.ndarray],
    ) -> None:
        super().__init__(model_name="stub")
        self._analyses_by_text = analyses_by_text
        self._prototypes = {
            label: vector.astype(np.float32) for label, vector in prototypes.items()
        }

    def analyze_texts(self, texts: Sequence[str]):
        return [self._analyses_by_text[text] for text in texts]

    def _prototype_embeddings(self):
        return self._prototypes


class PersonaEmotionTests(TestCase):
    """Verify weighted Persona emotion blending stays stable."""

    @staticmethod
    def _analysis(
        embedding: tuple[float, float],
        top_label: str,
        other_label: str,
    ) -> EmotionAnalysis:
        """Build one simple deterministic analysis object."""
        return EmotionAnalysis(
            embedding=np.array(embedding, dtype=np.float32),
            predictions=(
                EmotionPrediction(label=top_label, score=0.9),
                EmotionPrediction(label=other_label, score=0.1),
            ),
        )

    def test_weighted_history_prefers_user_emotion_and_softens_negative_target(
        self,
    ) -> None:
        """Verify user-heavy blending turns negative emotion into soft reinforcement."""
        analyzer = StubPersonaEmotionAnalyzer(
            analyses_by_text={
                "Earlier I was okay.": self._analysis((1.0, 0.0), "sadness", "joy"),
                "I am still here with you.": self._analysis(
                    (0.0, 1.0), "joy", "sadness"
                ),
                "I feel devastated.": self._analysis((1.0, 0.0), "sadness", "joy"),
            },
            prototypes={
                "sadness": np.array((1.0, 0.0), dtype=np.float32),
                "joy": np.array((0.0, 1.0), dtype=np.float32),
            },
        )

        state = analyzer.summarize_history(
            history=(
                {"role": "user", "content": "Earlier I was okay."},
                {"role": "assistant", "content": "I am still here with you."},
            ),
            request="I feel devastated.",
        )

        assert state is not None
        self.assertEqual(state.target_label, "sadness")
        self.assertIn("gently reassuring", state.target_state)
        self.assertEqual(state.user_label, "sadness")

    def test_positive_target_is_mirrored_softly(self) -> None:
        """Verify positive blends preserve the feeling but soften the delivery."""
        analyzer = StubPersonaEmotionAnalyzer(
            analyses_by_text={
                "This is amazing!": self._analysis((0.0, 1.0), "joy", "sadness"),
                "I'm glad you like it.": self._analysis((0.0, 1.0), "joy", "sadness"),
            },
            prototypes={
                "sadness": np.array((1.0, 0.0), dtype=np.float32),
                "joy": np.array((0.0, 1.0), dtype=np.float32),
            },
        )

        state = analyzer.summarize_history(
            history=({"role": "assistant", "content": "I'm glad you like it."},),
            request="This is amazing!",
        )

        assert state is not None
        self.assertEqual(state.target_label, "joy")
        self.assertIn("softly joyful", state.target_state)

    def test_generic_huggingface_labels_remap_to_goemotions_by_index(self) -> None:
        """Verify generic LABEL_n configs do not leak placeholder names into Persona state."""
        config = type(
            "Config",
            (),
            {
                "id2label": {
                    str(index): f"LABEL_{index}"
                    for index in range(len(GOEMOTIONS_LABELS))
                }
            },
        )()

        labels = PersonaEmotionAnalyzer._resolve_labels(config)

        self.assertEqual(labels, GOEMOTIONS_LABELS)
