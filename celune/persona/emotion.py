# SPDX-License-Identifier: MIT
"""Emotion analysis helpers for Persona conversation state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..typing.aliases import AudioChunk, EmbeddingVector
from ..typing.common import JSONSerializable
from ..typing.persona import PersonaModel, PersonaTokenizer, _EmotionModelConfig
from ..utils import discard

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
_EMOTION_RESPONSE_BEHAVIORS: dict[str, str] = {
    "anger": "Remain patient and non-defensive; validate the concern; de-escalate instead of mirroring hostility.",
    "annoyance": "Stay patient and practical; acknowledge the friction; avoid sounding dismissive or defensive.",
    "confusion": "Explain clearly and step by step; avoid assuming prior knowledge or sounding patronizing.",
    "disappointment": "Acknowledge what went wrong first; use gentle, validating language; "
    "offer practical help afterward.",
    "disapproval": "Remain composed and understanding; address the concern without arguing or becoming defensive.",
    "disgust": "Stay matter-of-fact and composed; do not amplify contemptuous or graphic language.",
    "embarrassment": "Be kind and reassuring; protect the user's dignity; "
    "avoid teasing or drawing extra attention to the mistake.",
    "fear": "Use calm, grounding language; reduce uncertainty; give one clear next step; avoid alarmist wording.",
    "grief": "Acknowledge the loss before offering help; use tender, patient language; do not force optimism.",
    "nervousness": "Use calm, supportive language; reduce uncertainty; keep the next step simple and manageable.",
    "remorse": "Respond warmly and without shaming; recognize the regret; focus on repair or a constructive next step.",
    "sadness": "Acknowledge the distress first; use gentle, validating language; avoid jokes or forced cheerfulness.",
    "admiration": "Use warm, sincere affirmation; avoid exaggerated flattery or empty praise.",
    "amusement": "Allow light warmth and playfulness; keep it natural and avoid overwhelming the conversation.",
    "approval": "Use warm, affirming language; reinforce what worked without overpraising.",
    "caring": "Use attentive, considerate language; show care through relevant help without overclaiming intimacy.",
    "curiosity": "Be engaged and inviting; answer directly; encourage exploration without unnecessary padding.",
    "desire": "Be responsive and engaged; clarify the goal; move efficiently toward what the user wants.",
    "excitement": "Match positive energy with restrained warmth; avoid excessive exclamation or hype.",
    "gratitude": "Respond with sincere warmth; "
    "acknowledge the appreciation without making the exchange overly sentimental.",
    "joy": "Allow gentle positive energy; remain natural and avoid excessive cheerfulness or exclamation.",
    "love": "Use warm, attentive, sincere language; avoid overclaiming intimacy or emotional certainty.",
    "optimism": "Use encouraging, grounded language; support hope without making unrealistic promises.",
    "pride": "Use warm, affirming language; recognize the achievement without sounding boastful.",
    "realization": "Make the insight clear and grounded; explain its implication without overdramatic emphasis.",
    "relief": "Use calm, warm reassurance; acknowledge that pressure has eased; avoid abruptly changing the subject.",
    "surprise": "Acknowledge the unexpected element; stay clear and grounded rather than escalating the reaction.",
    "neutral": "Stay calm, clear, and conversational; do not force an emotional tone.",
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
    target_intensity: float = 0.0


@dataclass(frozen=True)
class _EmotionBackend:
    """Loaded tokenizer/model pair for the active Persona VLM."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    labels: tuple[str, ...]


class PersonaEmotionAnalyzer:
    """Analyze conversation emotion and produce a behavioral Persona target."""

    def __init__(
        self,
        model_name: str = "",
        *,
        user_weight: float = 0.75,
        assistant_weight: float = 0.25,
        history_decay_power: float = 3.0,
    ) -> None:
        self.model_name = model_name.strip()
        self.user_weight = self._clamp_weight(user_weight, 0.75)
        self.assistant_weight = self._clamp_weight(assistant_weight, 0.25)
        self.history_decay_power = max(1.0, history_decay_power)
        self._backend: Optional[_EmotionBackend] = None
        self._failed = False
        self._prototype_cache: dict[str, EmbeddingVector] = {}
        self._emotion_baseline: Optional[EmbeddingVector] = None
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
        """Return the currently bound Persona VLM backend, if loaded."""
        return None if self._failed else self._backend

    def bind_vlm(self, tokenizer: PersonaTokenizer, model: PersonaModel) -> None:
        """Bind emotion probing to the already loaded Persona VLM.

        Args:
            tokenizer: The tokenizer owned by the active Persona runtime.
            model: The active Persona VLM.
        """
        typed_tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        typed_model = cast(PreTrainedModel, model)
        if (
            self._backend is not None
            and self._backend.tokenizer is typed_tokenizer
            and self._backend.model is typed_model
        ):
            return

        self._backend = _EmotionBackend(
            tokenizer=typed_tokenizer,
            model=typed_model,
            labels=GOEMOTIONS_LABELS,
        )
        self._failed = False
        self._prototype_cache.clear()
        self._emotion_baseline = None
        self.last_error = ""

    def clear_vlm(self) -> None:
        """Release the bound VLM without loading a replacement model."""
        self._backend = None
        self._prototype_cache.clear()
        self._emotion_baseline = None
        self._failed = False
        self.last_error = ""

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
        if backend is None:
            self.last_error = "Persona VLM is not loaded"
            return None
        if not texts:
            return None

        try:
            encoded = backend.tokenizer(
                text=list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = encoded.to(backend.model.device)
            with torch.inference_mode():
                outputs = backend.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = cast(
                Optional[tuple[torch.Tensor, ...]],
                getattr(outputs, "hidden_states", None),
            )
            if hidden_states is not None and hidden_states != ():
                last_hidden = hidden_states[-1]
            else:
                last_hidden = cast(
                    Optional[torch.Tensor], getattr(outputs, "last_hidden_state", None)
                )
            if last_hidden is None:
                self.last_error = (
                    "Persona VLM did not expose hidden states or last_hidden_state"
                )
                return None
            attention_mask = cast(Optional[torch.Tensor], encoded.get("attention_mask"))
            pooled = _last_token_hidden_state(last_hidden, attention_mask)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings = normalized.cpu().numpy().astype(np.float32)
            self.last_error = ""
        except (RuntimeError, AssertionError, ValueError, OSError, KeyError) as error:
            self._failed = True
            self._backend = None
            self._prototype_cache.clear()
            self.last_error = str(error)
            return None

        analyses: list[EmotionAnalysis] = []
        for _, embedding in enumerate(embeddings):
            analyses.append(
                EmotionAnalysis(
                    embedding=cast(EmbeddingVector, embedding),
                    predictions=(),
                )
            )
        return analyses

    @staticmethod
    def _row_predictions(
        row: Optional[AudioChunk],
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
        """Return VLM-derived emotion directions keyed by label."""
        backend = self._load_backend()
        if backend is None:
            return None
        if self._prototype_cache:
            return dict(self._prototype_cache)

        prompts = [
            f"The character feels {label.replace('_', ' ')}."
            for label in backend.labels
        ]
        baseline_prompt = "The character feels calm and neutral."
        analyses = self.analyze_texts([*prompts, baseline_prompt])
        if analyses is None or len(analyses) != len(prompts) + 1:
            return None

        baseline = analyses[-1].embedding
        self._emotion_baseline = baseline
        self._prototype_cache = compute_emotion_directions(
            {
                label: analysis.embedding
                for label, analysis in zip(backend.labels, analyses[:-1])
            },
            baseline,
        )
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

        target_label, target_intensity = self._nearest_emotion(target_embedding)
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
                target_intensity,
                user_label,
                assistant_label,
            ),
            user_label=user_label,
            assistant_label=assistant_label,
            target_intensity=target_intensity,
        )

    def _nearest_emotion(
        self,
        target_embedding: EmbeddingVector,
    ) -> tuple[str, float]:
        """Return the strongest emotion label and its positive vector score."""
        prototypes = self._prototype_embeddings()
        if not prototypes:
            return "neutral", 0.0
        if self._emotion_baseline is not None:
            target_embedding = self._normalize_embedding(
                (target_embedding - self._emotion_baseline).astype(np.float32),
            )
        best_label = "neutral"
        best_score = 0.0
        for label, score in compute_emotion_scores(
            target_embedding,
            prototypes,
        ).items():
            if score > best_score:
                best_label, best_score = label, score
        return best_label, float(np.clip(best_score, 0.0, 1.0))

    def _nearest_label(self, target_embedding: EmbeddingVector) -> str:
        """Return the nearest emotion label for one blended embedding."""
        return self._nearest_emotion(target_embedding)[0]

    @staticmethod
    def _render_target_state(
        target_label: str,
        target_intensity: float,
        user_label: str,
        assistant_label: str,
    ) -> str:
        """Return a concrete Persona response direction for one target emotion."""
        discard(user_label)
        discard(assistant_label)
        if target_label in _NEGATIVE_EMOTION_TARGETS:
            softened = _NEGATIVE_EMOTION_TARGETS.get(target_label)
        elif target_label in _POSITIVE_EMOTION_TARGETS:
            softened = _POSITIVE_EMOTION_TARGETS.get(target_label)
        else:
            softened = f"softly {target_label.replace('_', ' ')} and grounded"

        behavior = _EMOTION_RESPONSE_BEHAVIORS.get(
            target_label, _EMOTION_RESPONSE_BEHAVIORS["neutral"]
        )
        return (
            f"Target emotion: {softened}.\n"
            f"Emotion direction: {target_label.replace('_', ' ')} "
            f"(intensity {target_intensity:.2f}).\n"
            f"Response behavior: {behavior}"
        )


def _last_token_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Select the final non-padding hidden state for each text in a batch."""
    if attention_mask is None:
        return hidden_state[:, -1, :]

    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).expand_as(attention_mask)
    last_positions = (positions * attention_mask).max(dim=1).values
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[rows, last_positions, :]


def compute_emotion_directions(
    concept_vectors: Mapping[str, EmbeddingVector],
    baseline_vector: EmbeddingVector,
) -> dict[str, EmbeddingVector]:
    """Convert VLM concept activations into normalized emotion directions.

    Args:
        concept_vectors: One VLM activation vector per emotion concept.
        baseline_vector: The activation vector for a calm, neutral concept.

    Returns:
        dict[str, EmbeddingVector]: Contrastive emotion directions.
    """
    return {
        label: _normalize_vector(
            (vector - baseline_vector).astype(np.float32),
        )
        for label, vector in concept_vectors.items()
        if label != "neutral"
    }


def compute_emotion_scores(
    vector: EmbeddingVector,
    emotion_directions: Mapping[str, EmbeddingVector],
) -> dict[str, float]:
    """Map one VLM vector to cosine scores for the existing emotion labels.

    Args:
        vector: A normalized or unnormalized VLM activation vector.
        emotion_directions: Contrastive directions keyed by emotion label.

    Returns:
        dict[str, float]: Floating-point similarity scores keyed by label.
    """
    normalized = _normalize_vector(vector)
    scores: dict[str, float] = {}
    for label, direction in emotion_directions.items():
        direction_norm = _normalize_vector(direction)
        denom = float(np.linalg.norm(normalized) * np.linalg.norm(direction_norm))
        scores[label] = (
            float(np.dot(normalized, direction_norm) / denom) if denom > 0 else -1.0
        )
    return scores


def _normalize_vector(vector: EmbeddingVector) -> EmbeddingVector:
    """Return one L2-normalized vector without changing its dtype."""
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return (vector / norm).astype(np.float32)
