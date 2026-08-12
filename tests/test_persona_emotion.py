# SPDX-License-Identifier: MIT
"""Tests for Persona emotion blending."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from celune.persona.emotion import (
    GOEMOTIONS_LABELS,
    EmotionAnalysis,
    EmotionPrediction,
    PersonaEmotionAnalyzer,
    compute_emotion_directions,
    compute_emotion_scores,
)
from celune.typing.persona import ModelGenerateKwargValue
from celune.utils import discard

from .support import CeluneTestCase


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


class FakePersonaTokenizer:
    """Small tokenizer double for live-VLM emotion probing tests."""

    eos_token_id: Optional[int]

    def __init__(self) -> None:
        self.eos_token_id = 0

    def __call__(
        self,
        *,
        text: Union[str, Sequence[str]],
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> BatchEncoding:
        """Return a mock batch encoding."""
        discard(return_tensors)
        discard(padding)
        discard(truncation)
        discard(max_length)
        values = [text] if isinstance(text, str) else list(text)
        lengths = torch.tensor([[len(value), len(value) + 1] for value in values])
        return BatchEncoding(
            {
                "input_ids": lengths,
                "attention_mask": torch.ones_like(lengths),
            }
        )

    @staticmethod
    def decode(token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        """Decode mock tokens."""
        discard(token_ids)
        discard(skip_special_tokens)
        return ""


class FakePersonaModel:
    """VLM double exposing hidden states without a separate emotion model."""

    def __init__(self) -> None:
        self.device: Union[torch.device, str] = torch.device("cpu")
        self.calls = 0
        self.config = SimpleNamespace(id2label={})

    def __call__(
        self,
        **kwargs: Union[torch.Tensor, bool],
    ) -> SimpleNamespace:
        """Return a mock emotional vector."""
        self.calls += 1
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        input_ids = input_ids.float()
        hidden = torch.stack((input_ids, input_ids * 2), dim=-1)
        return SimpleNamespace(hidden_states=(hidden,))

    @staticmethod
    def generate(**kwargs: ModelGenerateKwargValue) -> torch.Tensor:
        """Generate mock tokens."""
        discard(kwargs)
        return torch.empty((1, 0), dtype=torch.long)

    def eval(self) -> None:
        """Evaluate mock tokens."""


class TestPersonaEmotion(CeluneTestCase):
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
        assert state.target_label == "sadness"
        assert "gently reassuring" in state.target_state
        assert "Emotion direction: sadness" in state.target_state
        assert "Response behavior:" in state.target_state
        assert state.target_intensity > 0.0
        assert state.user_label == "sadness"

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
        assert state.target_label == "joy"
        assert "softly joyful" in state.target_state

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

        assert labels == GOEMOTIONS_LABELS

    def test_live_vlm_hidden_states_produce_emotion_vectors(self) -> None:
        """Verify emotion probing reuses the loaded Persona VLM backend."""
        tokenizer = FakePersonaTokenizer()
        model = FakePersonaModel()
        analyzer = PersonaEmotionAnalyzer()

        analyzer.bind_vlm(tokenizer, model)
        analyses = analyzer.analyze_texts(("The user is curious.",))

        assert analyses is not None
        assert analyses is not None
        assert model.calls == 1
        assert round(abs(float(np.linalg.norm(analyses[0].embedding)) - 1.0), 5) == 0
        assert analyzer._prototype_embeddings() is not None
        assert model.calls > 1

    def test_emotion_vectors_map_back_to_existing_labels(self) -> None:
        """Verify contrastive floating-point vectors retain label mapping."""
        baseline = np.array((1.0, 0.0), dtype=np.float32)
        directions = compute_emotion_directions(
            {
                "sadness": np.array((1.0, 1.0), dtype=np.float32),
                "joy": np.array((2.0, 0.0), dtype=np.float32),
            },
            baseline,
        )

        scores = compute_emotion_scores(
            np.array((0.0, 1.0), dtype=np.float32),
            directions,
        )

        assert scores["sadness"] > scores["joy"]
