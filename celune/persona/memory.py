# SPDX-License-Identifier: Apache-2.0
"""Persistent long-term memory helpers for the Persona system."""

from __future__ import annotations

import re
import json
import uuid
import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Union, Optional, cast
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as f
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..paths import persona_data_dir
from .paths import persona_character_slug
from ..typing.common import JSON, JSONSerializable
from ..constants import PERSONA_MEMORY_EMBEDDING_MODEL
from ..typing.aliases import EmbeddingVector, EmbeddingBackend

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "remember",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "you",
}
_EXPLICIT_MEMORY_PATTERNS = (
    r"^(?:please\s+)?remember(?:\s+that)?\s+(.+)$",
    r"^(?:please\s+)?(?:don't|do not)\s+forget(?:\s+that)?\s+(.+)$",
    (
        r"^(?:i\s+want|i'd\s+like|i\s+would\s+like)\s+you\s+to\s+"
        r"(?:remember|know|keep\s+in\s+mind)(?:\s+that)?\s*[:,-]?\s*(.+)$"
    ),
    r"^(?:please\s+)?keep(?:\s+(?:this|that))?\s+in\s+mind(?:\s+that)?\s*[:,-]?\s*(.+)$",
    r"^(?:please\s+)?bear\s+in\s+mind(?:\s+that)?\s*[:,-]?\s*(.+)$",
    (
        r"^(?:please\s+)?(?:make|take)\s+(?:a\s+)?note"
        r"\b(?:\s+(?:that|of)(?:\s+(?:this|that))?|\s+(?:this|that))?"
        r"\s*[:,-]?\s*(.+)$"
    ),
    r"^(?:please\s+)?note\b(?:\s+(?:that|of)(?:\s+(?:this|that))?|\s+(?:this|that))?\s*[:,-]?\s*(.+)$",
    r"^(?:please\s+)?(?:save|store)\s+(?:this|that)(?:\s+(?:in|to)\s+(?:your\s+)?memory)?\s*[:,-]?\s*(.+)$",
    r"^(?:please\s+)?add\s+(?:this|that)\s+to\s+(?:your\s+)?memory\s*[:,-]?\s*(.+)$",
    r"^(?:please\s+)?keep(?:\s+(?:this|that))?\s+on\s+record\s*[:,-]?\s*(.+)$",
    r"^(?:please\s+)?make\s+sure\s+you\s+remember(?:\s+that)?\s*[:,-]?\s*(.+)$",
    (
        r"^(?:for\s+(?:future|our\s+future)\s+conversations|for\s+next\s+time|"
        r"going\s+forward|from\s+now\s+on)\s*[:,-]?\s*(.+)$"
    ),
    r"^(?:you\s+should|you\s+need\s+to)\s+know(?:\s+that)?\s*[:,-]?\s*(.+)$",
    (
        r"^(?:it(?:'s|\s+is)|this\s+is|that\s+is)\s+(?:a\s+)?"
        r"(?:key|important)\s+(?:fact|detail)(?:\s+that)?\s*[:,-]?\s*(.+)$"
    ),
)
_SENSITIVE_MEMORY_PATTERN = re.compile(
    r"\b(?:password|passcode|secret|api[ -]?key|access token|private key|"
    r"credit card|bank account|social security|\bssn\b)\b",
    flags=re.IGNORECASE,
)

_EMBEDDING_BACKENDS: dict[str, EmbeddingBackend] = {}
_FAILED_EMBEDDING_MODELS: set[str] = set()


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _normalize_text(text: str) -> str:
    """Normalize one memory string for stable storage and matching."""
    collapsed = " ".join(text.strip().split())
    return collapsed.strip(" \t\r\n-:;,.")


def _tokenize(text: str) -> set[str]:
    """Return normalized matching tokens for one text string."""
    lowered = text.casefold()
    return {
        token
        for token in _WORD_RE.findall(lowered)
        if token and token not in _STOPWORDS
    }


def _clamp_similarity_threshold(
    value: JSONSerializable, fallback: float = 0.62
) -> float:
    """Normalize one semantic match threshold into the valid cosine range."""
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return max(-1.0, min(1.0, float(value)))
    return fallback


def _clamp_overlap_threshold(value: JSONSerializable, fallback: int = 1) -> int:
    """Normalize the fallback token-overlap threshold."""
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return max(1, int(value))
    return fallback


def _cosine_similarity(first: EmbeddingVector, second: EmbeddingVector) -> float:
    """Return cosine similarity between two embedding vectors."""
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 0:
        raise ValueError("embedding norm is zero")
    return float(np.dot(first, second) / denom)


def _load_transformer_text_embedder(model_name: str) -> Optional[EmbeddingBackend]:
    """Load one lazy text-embedding backend, or return ``None`` when unavailable."""
    if model_name in _FAILED_EMBEDDING_MODELS:
        return None
    if model_name in _EMBEDDING_BACKENDS:
        return _EMBEDDING_BACKENDS[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        if tokenizer is None:
            raise RuntimeError("tokenizer not available")

        model = cast(
            PreTrainedModel,
            AutoModel.from_pretrained(model_name, local_files_only=True),
        )
        model.eval()
        model.to(torch.device("cpu"))
    except (RuntimeError, AssertionError, ValueError, OSError):
        _FAILED_EMBEDDING_MODELS.add(model_name)
        return None

    backend: EmbeddingBackend = (tokenizer, model)
    _EMBEDDING_BACKENDS[model_name] = backend
    return backend


def _compute_text_embeddings(
    texts: Sequence[str],
    model_name: str,
) -> Optional[list[EmbeddingVector]]:
    """Return semantic text embeddings for the requested texts when available."""
    backend = _load_transformer_text_embedder(model_name)
    if backend is None:
        return None

    tokenizer, model = backend
    tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

    try:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = cast(torch.Tensor, encoded["attention_mask"])
        attention = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
        normalized = f.normalize(pooled, p=2, dim=1)
        array = normalized.cpu().numpy().astype(np.float32)
        return [cast(EmbeddingVector, row) for row in array]
    except (RuntimeError, AssertionError, ValueError, OSError):
        _FAILED_EMBEDDING_MODELS.add(model_name)
        _EMBEDDING_BACKENDS.pop(model_name, None)
        return None


def default_memory_dir() -> Path:
    """Return the default on-disk directory for Persona memories.

    Returns:
        Path: The Persona character app-data directory where memories are stored.
    """
    return persona_data_dir()


@dataclass(slots=True, frozen=True)
class MemoryRecord:
    """One stored long-term memory record."""

    id: str
    content: str
    importance: int
    explicit: bool
    created_at: str
    updated_at: str
    last_used_at: str

    def to_json(self) -> JSON:
        """Serialize this memory record for typed agent and API results."""
        return {
            "id": self.id,
            "content": self.content,
            "importance": self.importance,
            "explicit": self.explicit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
        }

    @staticmethod
    def create(content: str, importance: int, explicit: bool) -> MemoryRecord:
        """Construct one new memory record.

        Args:
            content: The memory's content.
            importance: The memory's importance value.
            explicit: Whether this memory was created explicitly or not.

        Returns:
            MemoryRecord: The created memory record.
        """
        now = _utc_now()
        return MemoryRecord(
            id=uuid.uuid4().hex,
            content=_normalize_text(content),
            importance=max(1, min(3, importance)),
            explicit=explicit,
            created_at=now,
            updated_at=now,
            last_used_at=now,
        )


@dataclass(slots=True, frozen=True)
class MemoryCandidate:
    """Candidate memory extracted from a user message."""

    content: str
    importance: int
    explicit: bool


class PersonaMemoryStore:
    """JSON-backed character-specific long-term memory store."""

    def __init__(
        self,
        storage_dir: Optional[Union[Path, str]] = None,
        *,
        semantic_similarity_threshold: float = 0.62,
        fallback_token_overlap_threshold: int = 1,
        embedding_model: str = PERSONA_MEMORY_EMBEDDING_MODEL,
    ) -> None:
        self.storage_dir = (
            Path(storage_dir) if storage_dir is not None else default_memory_dir()
        )
        self.semantic_similarity_threshold = _clamp_similarity_threshold(
            semantic_similarity_threshold
        )
        self.fallback_token_overlap_threshold = _clamp_overlap_threshold(
            fallback_token_overlap_threshold
        )
        self.embedding_model = embedding_model.strip() or PERSONA_MEMORY_EMBEDDING_MODEL
        self._embedding_cache: dict[str, EmbeddingVector] = {}
        self._embedding_cache_max = 2048

    def _path_for_character(self, character_name: str) -> Path:
        """Return the JSON file path for one active character."""
        slug = persona_character_slug(character_name)
        return self.storage_dir / slug / "memory" / "records.json"

    def load_records(self, character_name: str) -> list[MemoryRecord]:
        """Load all memory records for one character.

        Args:
            character_name: The character name to retrieve memory records for.

        Returns:
            list[MemoryRecord]: All memory records for the selected character.
        """
        path = self._path_for_character(character_name)
        if not path.exists():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []

        if not isinstance(raw, list):
            return []

        records: list[MemoryRecord] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not _normalize_text(content):
                continue
            record_id = item.get("id")
            created_at = item.get("created_at")
            updated_at = item.get("updated_at")
            last_used_at = item.get("last_used_at")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (record_id, created_at, updated_at, last_used_at)
            ):
                continue
            record_id = str(record_id)
            created_at = str(created_at)
            updated_at = str(updated_at)
            last_used_at = str(last_used_at)
            importance = item.get("importance", 1)
            explicit = item.get("explicit", False)
            if isinstance(importance, bool):
                importance = 1
            if not isinstance(importance, (int, float)):
                importance = 1
            records.append(
                MemoryRecord(
                    id=record_id.strip(),
                    content=_normalize_text(content),
                    importance=max(1, min(3, int(importance))),
                    explicit=bool(explicit),
                    created_at=created_at.strip(),
                    updated_at=updated_at.strip(),
                    last_used_at=last_used_at.strip(),
                )
            )
        return records

    def save_records(self, character_name: str, records: list[MemoryRecord]) -> None:
        """Persist all memory records for one character atomically.

        Args:
            character_name: The character name to save memory records for.
            records: The memory records to save.
        """
        path = self._path_for_character(character_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in records]
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def remember(
        self,
        character_name: str,
        content: str,
        *,
        importance: int = 1,
        explicit: bool = False,
    ) -> Optional[MemoryRecord]:
        """Store or update one memory for a character.

        Args:
            character_name: The character name to save or update a memory record for.
            content: The memory's content.
            importance: The memory's importance value.
            explicit: Whether the memory is explicit or not.

        Returns:
            Optional[MemoryRecord]: A stored or updated memory record, or ``None`` if memory normalization failed.
        """
        normalized = _normalize_text(content)
        if not normalized:
            return None

        records = self.load_records(character_name)
        now = _utc_now()
        for index, record in enumerate(records):
            if record.content.casefold() != normalized.casefold():
                continue
            updated = MemoryRecord(
                id=record.id,
                content=record.content,
                importance=max(record.importance, 1, min(3, importance)),
                explicit=record.explicit or explicit,
                created_at=record.created_at,
                updated_at=now,
                last_used_at=record.last_used_at,
            )
            records[index] = updated
            self.save_records(character_name, records)
            return updated

        created = MemoryRecord.create(
            normalized,
            importance=max(1, min(3, importance)),
            explicit=explicit,
        )
        records.append(created)
        self.save_records(character_name, records)
        return created

    def collect_candidates(self, user_message: str) -> list[MemoryCandidate]:
        """Extract explicit and automatic memory candidates from one message.

        Args:
            user_message: The user message to get memory candidates from.

        Returns:
            list[MemoryCandidate]: All possible candidates to be stored as memories.
        """
        text = _normalize_text(user_message)
        if not text:
            return []

        explicit = self._explicit_candidate(text)
        if explicit is not None:
            return [explicit]

        automatic = self._automatic_candidates(text)
        unique: list[MemoryCandidate] = []
        seen: set[str] = set()
        for candidate in automatic:
            key = candidate.content.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def remember_from_user_message(
        self, character_name: str, user_message: str
    ) -> list[MemoryRecord]:
        """Store any memories that should be derived from one user message.

        Args:
            character_name: The character name to save memory records for from current user message.
            user_message: The user message containing memory records to save.

        Returns:
            list[MemoryRecord]: A list of memories saved from the current user message.
        """
        saved: list[MemoryRecord] = []
        for candidate in self.collect_candidates(user_message):
            record = self.remember(
                character_name,
                candidate.content,
                importance=candidate.importance,
                explicit=candidate.explicit,
            )
            if record is not None:
                saved.append(record)
        return saved

    def retrieve(
        self, character_name: str, request: str, limit: int = 5
    ) -> list[MemoryRecord]:
        """Return the most relevant memories for the current request.

        Args:
            character_name: The character name to retrieve memory records for.
            request: The user request to retrieve memory records back to.
            limit: How many memory records should be retrieved at a time.

        Returns:
            list[MemoryRecord]: Up to ``limit`` most recent memories stored for the current character.
        """
        records = self.load_records(character_name)
        if not records or limit <= 0:
            return []

        request_text = _normalize_text(request)
        if not request_text:
            return []

        ranked = self._semantic_rank_records(records, request_text)
        if ranked is None or ranked == []:
            ranked = self._fallback_rank_records(records, request_text)

        if not ranked:
            return []

        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[2].importance,
                not item[2].explicit,
                item[1],
            )
        )
        selected = [record for _, _, record in ranked[:limit]]
        if not selected:
            return []

        now = _utc_now()
        selected_ids = {record.id for record in selected}
        updated_records = [
            MemoryRecord(
                id=record.id,
                content=record.content,
                importance=record.importance,
                explicit=record.explicit,
                created_at=record.created_at,
                updated_at=record.updated_at,
                last_used_at=now if record.id in selected_ids else record.last_used_at,
            )
            for record in records
        ]
        self.save_records(character_name, updated_records)
        return [
            record
            if record.id not in selected_ids
            else next(updated for updated in updated_records if updated.id == record.id)
            for record in selected
        ]

    def forget(self, character_name: str, record_id: str) -> bool:
        """Remove one character-scoped memory by its stable record identifier.

        Args:
            character_name: Character whose memory should be changed.
            record_id: Identifier of the memory record to remove.

        Returns:
            bool: Whether a matching record was removed.
        """
        normalized_id = record_id.strip()
        if not normalized_id:
            raise ValueError("memory record_id must not be empty")
        records = self.load_records(character_name)
        remaining = [record for record in records if record.id != normalized_id]
        if len(remaining) == len(records):
            return False
        self.save_records(character_name, remaining)
        return True

    def _embedding_cache_key(self, text: str) -> str:
        """Return the cache key used for one normalized text embedding."""
        return f"{self.embedding_model}\0{text.casefold()}"

    def _embed_texts(self, texts: Sequence[str]) -> Optional[list[EmbeddingVector]]:
        """Return embeddings for normalized texts, using an in-memory cache."""
        results: list[Optional[EmbeddingVector]] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cache_key = self._embedding_cache_key(text)
            if len(self._embedding_cache) >= self._embedding_cache_max:
                self._embedding_cache.pop(next(iter(self._embedding_cache)), None)

            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                results.append(None)
                missing_indices.append(index)
                missing_texts.append(text)
                continue
            results.append(cached)

        if missing_texts:
            fresh = _compute_text_embeddings(missing_texts, self.embedding_model)
            if fresh is None or len(fresh) != len(missing_texts):
                return None
            for index, text, embedding in zip(missing_indices, missing_texts, fresh):
                cache_key = self._embedding_cache_key(text)
                self._embedding_cache[cache_key] = embedding
                results[index] = embedding

        if any(embedding is None for embedding in results):
            return None
        return [cast(EmbeddingVector, embedding) for embedding in results]

    def _semantic_rank_records(
        self,
        records: Sequence[MemoryRecord],
        request_text: str,
    ) -> Optional[list[tuple[float, int, MemoryRecord]]]:
        """Return semantic similarity matches, or ``None`` when embeddings are unavailable."""
        texts = [request_text, *(record.content for record in records)]
        embeddings = self._embed_texts(texts)
        if embeddings is None:
            return None

        request_embedding = embeddings[0]
        ranked: list[tuple[float, int, MemoryRecord]] = []
        for index, (record, embedding) in enumerate(zip(records, embeddings[1:])):
            similarity = _cosine_similarity(request_embedding, embedding)
            if similarity < self.semantic_similarity_threshold:
                continue

            score = similarity * 100.0
            score += record.importance * 3.0
            if record.explicit:
                score += 2.0
            if request_text.casefold() in record.content.casefold():
                score += 2.0
            ranked.append((score, index, record))
        return ranked

    def _fallback_rank_records(
        self,
        records: Sequence[MemoryRecord],
        request_text: str,
    ) -> list[tuple[float, int, MemoryRecord]]:
        """Return the legacy token-overlap ranking when semantic embeddings are unavailable."""
        request_tokens = _tokenize(request_text)
        ranked: list[tuple[float, int, MemoryRecord]] = []
        for index, record in enumerate(records):
            content_tokens = _tokenize(record.content)
            overlap = len(request_tokens & content_tokens)
            if overlap < self.fallback_token_overlap_threshold:
                continue

            score = float(overlap * 10)
            score += record.importance * 3.0
            if record.explicit:
                score += 2.0
            if request_text.casefold() in record.content.casefold():
                score += 2.0
            ranked.append((score, index, record))
        return ranked

    @staticmethod
    def _explicit_candidate(text: str) -> Optional[MemoryCandidate]:
        """Return one explicit memory candidate when the user asks to remember."""
        lowered = text.casefold()
        if lowered.startswith(("do you remember", "what do you remember")):
            return None

        for pattern in _EXPLICIT_MEMORY_PATTERNS:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue
            content = _normalize_text(match.group(1))
            if content:
                return MemoryCandidate(content=content, importance=3, explicit=True)
        return None

    @staticmethod
    def _automatic_candidates(text: str) -> list[MemoryCandidate]:
        """Return conservative automatic memory candidates."""
        candidates: list[MemoryCandidate] = []
        patterns: tuple[tuple[str, str, int], ...] = (
            (
                r"^my name is\s+(.+)$",
                "The user's name is {value}.",
                3,
            ),
            (
                r"^call me\s+(.+)$",
                "The user wants to be called {value}.",
                3,
            ),
            (
                r"^my favorite ([a-z0-9 _-]+?) is\s+(.+)$",
                "The user's favorite {key} is {value}.",
                3,
            ),
            (
                r"^i prefer\s+(.+)$",
                "The user prefers {value}.",
                2,
            ),
            (
                r"^i(?:'m| am) working on\s+(.+)$",
                "The user is working on {value}.",
                2,
            ),
            (
                r"^my project is\s+(.+)$",
                "The user's project is {value}.",
                2,
            ),
            (
                r"^we(?:'re| are) working on\s+(.+)$",
                "The user is working on {value}.",
                2,
            ),
            (
                r"^my goal is\s+(.+)$",
                "The user's goal is {value}.",
                2,
            ),
            (
                r"^i want to build\s+(.+)$",
                "The user wants to build {value}.",
                2,
            ),
        )

        for pattern, template, importance in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue

            groups = match.groups()
            if len(groups) == 1:
                value = _normalize_text(groups[0])
                if value:
                    candidates.append(
                        MemoryCandidate(
                            content=template.format(value=value),
                            importance=importance,
                            explicit=False,
                        )
                    )
                continue

            if len(groups) == 2:
                key = _normalize_text(groups[0])
                value = _normalize_text(groups[1])
                if key and value:
                    candidates.append(
                        MemoryCandidate(
                            content=template.format(key=key, value=value),
                            importance=importance,
                            explicit=False,
                        )
                    )

        return candidates


def classifier_memory_candidates(
    payload: JSONSerializable,
    *,
    minimum_confidence: float = 0.82,
    maximum_candidates: int = 3,
) -> list[MemoryCandidate]:
    """Parse safe long-term memory candidates from classifier JSON output.

    Args:
        payload: Classifier response text or decoded JSON payload.
        minimum_confidence: Minimum classifier confidence required for storage.
        maximum_candidates: Maximum number of candidates accepted from one turn.

    Returns:
        list[MemoryCandidate]: Validated, non-explicit memory candidates.
    """
    if maximum_candidates <= 0:
        return []

    raw_payload: JSONSerializable = payload
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        try:
            raw_payload = cast(JSONSerializable, json.loads(text))
        except (TypeError, ValueError):
            return []

    if isinstance(raw_payload, dict):
        raw_candidates = raw_payload.get("memories")
    elif isinstance(raw_payload, list):
        raw_candidates = raw_payload
    else:
        return []

    if not isinstance(raw_candidates, list):
        return []

    threshold = max(0.0, min(1.0, minimum_confidence))
    candidates: list[MemoryCandidate] = []
    seen: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        confidence = item.get("confidence")
        importance = item.get("importance", 1)
        if not isinstance(content, str) or not isinstance(confidence, (int, float)):
            continue
        if isinstance(confidence, bool) or float(confidence) < threshold:
            continue

        normalized = _normalize_text(content)
        key = normalized.casefold()
        if (
            not normalized
            or key in seen
            or _SENSITIVE_MEMORY_PATTERN.search(normalized)
        ):
            continue

        if isinstance(importance, bool) or not isinstance(importance, (int, float)):
            importance = 1
        candidates.append(
            MemoryCandidate(
                content=normalized,
                importance=max(1, min(3, int(importance))),
                explicit=False,
            )
        )
        seen.add(key)
        if len(candidates) >= maximum_candidates:
            break

    return candidates
