# SPDX-License-Identifier: MIT
"""Emotion analysis helpers for Persona conversation state."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Optional, Protocol, Union, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from ..utils import discard
from ..typing.aliases import EmbeddingVector
from ..constants import JSONSerializable, PERSONA_EMOTION_MODEL

GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def _looks_like_generic_label(value: str, index: int) -> bool:
    """Return whether one config label is a placeholder such as ``LABEL_0``."""
    normalized = value.strip().casefold()
    return normalized in {
        f"label_{index}",
        f"label {index}",
        str(index),
    }


_NEGATIVE_EMOTION_TARGETS: dict[str, str] = {
    "anger": "calm and grounding",
    "annoyance": "patient and easing tension",
    "confusion": "clear and grounding",
    "disappointment": "softly encouraging",
    "disapproval": "gentle and understanding",
    "disgust": "careful and composed",
    "embarrassment": "kindly reassuring",
    "fear": "protective and reassuring",
    "grief": "tenderly consoling",
    "nervousness": "calm and supportive",
    "remorse": "warmly forgiving",
    "sadness": "gently reassuring",
}
_POSITIVE_EMOTION_TARGETS: dict[str, str] = {
    "admiration": "softly admiring",
    "amusement": "lightly playful",
    "approval": "warmly approving",
    "caring": "gently caring",
    "curiosity": "gently curious",
    "desire": "softly eager",
    "excitement": "gently excited",
    "gratitude": "warmly grateful",
    "joy": "softly joyful",
    "love": "warmly affectionate",
    "optimism": "softly hopeful",
    "pride": "quietly proud",
    "realization": "softly awakened",
    "relief": "softly relieved",
    "surprise": "gently surprised",
}


@dataclass(frozen=True)
class EmotionPrediction:
    """One scored emotion label."""

    label: str
    score: float


@dataclass(frozen=True)
class EmotionAnalysis:
    """Emotion analysis result for one message."""

    embedding: EmbeddingVector
    predictions: tuple[EmotionPrediction, ...]


@dataclass(frozen=True)
class PersonaEmotionState:
    """Weighted target emotion state for Persona prompting."""

    target_label: str
    target_state: str
    user_label: str
    assistant_label: str


@dataclass(frozen=True)
class _EmotionBackend:
    """Loaded tokenizer/model pair for one emotion model."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    labels: tuple[str, ...]


class _EmotionModelConfig(Protocol):
    """Protocol for model configs that expose emotion label mappings."""

    id2label: Mapping[Union[int, str], str]


class PersonaEmotionAnalyzer:
    """Analyze conversation emotion and produce a softened Persona target state."""

    def __init__(
        self,
        model_name: str = PERSONA_EMOTION_MODEL,
        *,
        user_weight: float = 0.75,
        assistant_weight: float = 0.25,
        history_decay_power: float = 3.0,
    ) -> None:
        self.model_name = model_name.strip() or PERSONA_EMOTION_MODEL
        self.user_weight = self._clamp_weight(user_weight, 0.75)
        self.assistant_weight = self._clamp_weight(assistant_weight, 0.25)
        self.history_decay_power = max(1.0, history_decay_power)
        self._backend: Optional[_EmotionBackend] = None
        self._failed = False
        self._prototype_cache: dict[str, EmbeddingVector] = {}
        self.last_error: str = ""

    @staticmethod
    def _clamp_weight(value: float, fallback: float) -> float:
        """Return one normalized role weight."""
        if isinstance(value, bool):
            return fallback
        if not isinstance(value, (int, float)):
            return fallback
        return max(0.0, float(value))

    def _load_backend(self) -> Optional[_EmotionBackend]:
        """Load the underlying model backend when available."""
        if self._failed:
            return None
        if self._backend is not None:
            return self._backend

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
            except (RuntimeError, AssertionError, ValueError, OSError):
                model = AutoModel.from_pretrained(self.model_name)
            model.eval()
            model.to(torch.device("cpu"))
            labels = self._resolve_labels(getattr(model, "config", None))
            self.last_error = ""
        except (RuntimeError, AssertionError, ValueError, OSError) as error:
            self._failed = True
            self.last_error = str(error)
            return None

        self._backend = _EmotionBackend(
            tokenizer=tokenizer,
            model=model,
            labels=labels,
        )
        return self._backend

    @staticmethod
    def _resolve_labels(config: Optional[_EmotionModelConfig]) -> tuple[str, ...]:
        """Return label names from model config or GoEmotions defaults."""
        raw = None if config is None else config.id2label
        if isinstance(raw, Mapping):
            labels: list[tuple[int, str]] = []
            generic_count = 0
            for key, value in raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(value, str) or not value.strip():
                    continue
                normalized = value.strip().casefold()
                if _looks_like_generic_label(normalized, idx):
                    generic_count += 1
                labels.append((idx, normalized))
            if labels:
                labels.sort(key=lambda item: item[0])
                if (
                    generic_count == len(labels)
                    and len(labels) == len(GOEMOTIONS_LABELS)
                    and all(
                        index == position for position, (index, _) in enumerate(labels)
                    )
                ):
                    return GOEMOTIONS_LABELS
                return tuple(label for _, label in labels)
        return GOEMOTIONS_LABELS

    def analyze_texts(self, texts: Sequence[str]) -> Optional[list[EmotionAnalysis]]:
        """Return embeddings and scored labels for the requested texts.

        Args:
            texts: The texts to analyze in one batch.

        Returns:
            Optional[list[EmotionAnalysis]]: Per-text emotion results, or ``None`` when the model is unavailable.
        """
        backend = self._load_backend()
        if backend is None or not texts:
            return None

        try:
            encoded = backend.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = backend.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = cast(
                Optional[tuple[torch.Tensor, ...]], outputs.hidden_states
            )
            if hidden_states is not None and hidden_states != ():
                last_hidden = hidden_states[-1]
            else:
                last_hidden = cast(
                    Optional[torch.Tensor], getattr(outputs, "last_hidden_state", None)
                )
            if last_hidden is None:
                self.last_error = f"{self.model_name} did not expose hidden states or last_hidden_state"
                return None
            attention_mask = cast(torch.Tensor, encoded["attention_mask"])
            attention = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(
                min=1
            )
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings = normalized.cpu().numpy().astype(np.float32)
            logits = cast(Optional[torch.Tensor], getattr(outputs, "logits", None))
            probabilities = (
                torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                if logits is not None
                else None
            )
            self.last_error = ""
        except (RuntimeError, AssertionError, ValueError, OSError, KeyError) as error:
            self._failed = True
            self._backend = None
            self._prototype_cache.clear()
            self.last_error = str(error)
            return None

        analyses: list[EmotionAnalysis] = []
        for index, embedding in enumerate(embeddings):
            row = probabilities[index] if probabilities is not None else None
            predictions = self._row_predictions(row, backend.labels)
            analyses.append(
                EmotionAnalysis(
                    embedding=cast(EmbeddingVector, embedding),
                    predictions=predictions,
                )
            )
        return analyses

    @staticmethod
    def _row_predictions(
        row: Optional[npt.NDArray[np.float32]],
        labels: tuple[str, ...],
    ) -> tuple[EmotionPrediction, ...]:
        """Return sorted label scores for one model row."""
        if row is None:
            return ()
        ranked = sorted(
            (
                EmotionPrediction(label=label, score=float(score))
                for label, score in zip(labels, row)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return tuple(ranked)

    def _prototype_embeddings(self) -> Optional[dict[str, EmbeddingVector]]:
        """Return prototype emotion embeddings keyed by label."""
        backend = self._load_backend()
        if backend is None:
            return None
        if self._prototype_cache:
            return dict(self._prototype_cache)

        prompts = [
            f"This feels like {label.replace('_', ' ')}." for label in backend.labels
        ]
        analyses = self.analyze_texts(prompts)
        if analyses is None or len(analyses) != len(prompts):
            return None

        self._prototype_cache = {
            label: analysis.embedding
            for label, analysis in zip(backend.labels, analyses)
        }
        return dict(self._prototype_cache)

    @staticmethod
    def _cosine_similarity(first: EmbeddingVector, second: EmbeddingVector) -> float:
        """Return cosine similarity between two embedding vectors."""
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom <= 0:
            return -1.0
        return float(np.dot(first, second) / denom)

    @staticmethod
    def _blend_predictions(
        weighted_predictions: Sequence[tuple[float, EmotionAnalysis]],
    ) -> str:
        """Return the strongest aggregate label across weighted predictions."""
        scores: dict[str, float] = {}
        for weight, analysis in weighted_predictions:
            for prediction in analysis.predictions:
                scores[prediction.label] = scores.get(prediction.label, 0.0) + (
                    weight * prediction.score
                )
        if not scores:
            return "neutral"
        return max(scores.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _normalize_embedding(vector: EmbeddingVector) -> EmbeddingVector:
        """Return one L2-normalized embedding vector."""
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector
        return (vector / norm).astype(np.float32)

    def _weighted_role_mean(
        self,
        entries: Sequence[tuple[float, EmotionAnalysis]],
    ) -> Optional[EmbeddingVector]:
        """Return one normalized weighted mean embedding for a message role."""
        if not entries:
            return None
        total_weight = sum(weight for weight, _ in entries)
        if total_weight <= 0:
            return None
        combined = sum(
            analysis.embedding * np.float32(weight) for weight, analysis in entries
        )
        return self._normalize_embedding(
            cast(
                EmbeddingVector,
                (combined / np.float32(total_weight)).astype(np.float32),
            )
        )

    @staticmethod
    def _message_weight(index: int, total: int, decay_power: float) -> float:
        """Return one recency weight for a chronological message index."""
        if total <= 0:
            return 0.0
        return float(((index + 1) / total) ** decay_power)

    def summarize_history(
        self,
        history: Sequence[Mapping[str, JSONSerializable]],
        request: str,
    ) -> Optional[PersonaEmotionState]:
        """Return a softened target emotion from prior chat plus the current request.

        Args:
            history: Prior Persona chat messages in chronological order.
            request: The current user request that should be treated as the latest user emotion sample.

        Returns:
            Optional[PersonaEmotionState]: The blended Persona mood target, or ``None`` when emotion analysis is
            unavailable.
        """
        entries: list[tuple[str, str]] = []
        for message in history:
            role = message.get("role")
            content = message.get("content")
            if (
                role in {"user", "assistant"}
                and isinstance(content, str)
                and content.strip()
            ):
                entries.append((str(role), content.strip()))

        clean_request = request.strip()
        if clean_request:
            entries.append(("user", clean_request))
        if not entries:
            return None

        analyses = self.analyze_texts([content for _, content in entries])
        if analyses is None or len(analyses) != len(entries):
            return None

        weighted_user: list[tuple[float, EmotionAnalysis]] = []
        weighted_assistant: list[tuple[float, EmotionAnalysis]] = []
        total = len(entries)
        for index, ((role, _content), analysis) in enumerate(zip(entries, analyses)):
            weight = self._message_weight(index, total, self.history_decay_power)
            if role == "user":
                weighted_user.append((weight, analysis))
            else:
                weighted_assistant.append((weight, analysis))

        user_embedding = self._weighted_role_mean(weighted_user)
        assistant_embedding = self._weighted_role_mean(weighted_assistant)
        if user_embedding is None and assistant_embedding is None:
            self.last_error = "no user or assistant emotion embeddings could be derived"
            return None

        role_mix: list[tuple[float, EmbeddingVector]] = []
        if user_embedding is not None:
            role_mix.append((self.user_weight, user_embedding))
        if assistant_embedding is not None:
            role_mix.append((self.assistant_weight, assistant_embedding))

        blend_total = sum(weight for weight, _ in role_mix)
        if blend_total <= 0:
            return None
        blended = sum(
            embedding * np.float32(weight) for weight, embedding in role_mix
        ) / np.float32(blend_total)
        target_embedding = self._normalize_embedding(
            cast(EmbeddingVector, blended.astype(np.float32))
        )

        target_label = self._nearest_label(target_embedding)
        user_label = self._blend_predictions(weighted_user)
        assistant_label = self._blend_predictions(weighted_assistant)
        if user_label == "neutral" and user_embedding is not None:
            user_label = self._nearest_label(user_embedding)
        if assistant_label == "neutral" and assistant_embedding is not None:
            assistant_label = self._nearest_label(assistant_embedding)
        return PersonaEmotionState(
            target_label=target_label,
            target_state=self._render_target_state(
                target_label,
                user_label,
                assistant_label,
            ),
            user_label=user_label,
            assistant_label=assistant_label,
        )

    def _nearest_label(self, target_embedding: EmbeddingVector) -> str:
        """Return the nearest emotion label for one blended embedding."""
        prototypes = self._prototype_embeddings()
        if not prototypes:
            return "neutral"
        best_label = "neutral"
        best_score = -1.0
        for label, embedding in prototypes.items():
            score = self._cosine_similarity(target_embedding, embedding)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label

    @staticmethod
    def _render_target_state(
        target_label: str,
        user_label: str,
        assistant_label: str,
    ) -> str:
        """Return the Persona prompt text for one target emotion blend."""
        discard(user_label)
        discard(assistant_label)
        if target_label in _NEGATIVE_EMOTION_TARGETS:
            softened = _NEGATIVE_EMOTION_TARGETS[target_label]
            return f"Target emotion: {softened}."

        if target_label in _POSITIVE_EMOTION_TARGETS:
            softened = _POSITIVE_EMOTION_TARGETS[target_label]
            return f"Target emotion: {softened}."

        return f"Target emotion: softly {target_label.replace('_', ' ')} and grounded."
