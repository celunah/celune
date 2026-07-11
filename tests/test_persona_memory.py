# SPDX-License-Identifier: MIT
"""Tests for Persona long-term memory persistence and retrieval."""

import tempfile
from pathlib import Path
from unittest import TestCase, mock
from typing import Union, Optional
from collections.abc import Sequence

import numpy as np

from celune.persona.memory import PersonaMemoryStore


class StubEmbeddingMemoryStore(PersonaMemoryStore):
    """Memory store with deterministic test embeddings."""

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        *,
        semantic_similarity_threshold: float = 0.62,
        fallback_token_overlap_threshold: int = 1,
        embedding_model: str = "stub",
        embedding_map: Optional[dict[str, tuple[float, ...]]] = None,
    ) -> None:
        super().__init__(
            storage_dir=storage_dir,
            semantic_similarity_threshold=semantic_similarity_threshold,
            fallback_token_overlap_threshold=fallback_token_overlap_threshold,
            embedding_model=embedding_model,
        )
        self.embedding_map = embedding_map or {}
        self.return_none = False

    def _embed_texts(self, texts: Sequence[str]) -> Optional[list[np.ndarray]]:
        if self.return_none:
            return None
        return [np.array(self.embedding_map[text], dtype=np.float32) for text in texts]


class PersonaMemoryTests(TestCase):
    """Verify Persona long-term memory behavior stays conservative and stable."""

    def test_explicit_memory_persists_across_store_instances(self) -> None:
        """Verify explicit memories survive reloads from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            first = PersonaMemoryStore(storage_dir=temp_dir)
            saved = first.remember_from_user_message(
                "Celune",
                "remember that my test word is moonlight",
            )

            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].content, "my test word is moonlight")
            self.assertEqual(saved[0].explicit, True)

            second = PersonaMemoryStore(storage_dir=temp_dir)
            records = second.load_records("Celune")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].content, "my test word is moonlight")
            self.assertEqual(records[0].explicit, True)

    def test_automatic_memory_extracts_persistent_user_context(self) -> None:
        """Verify obvious user preferences are stored automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PersonaMemoryStore(storage_dir=temp_dir)
            saved = store.remember_from_user_message(
                "Celune",
                "my favorite color is blue",
            )

            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].content, "The user's favorite color is blue")
            self.assertEqual(saved[0].explicit, False)
            self.assertEqual(saved[0].importance, 3)

    def test_memory_records_use_persona_character_directory_by_default(self) -> None:
        """Verify memory records use the character-specific app-data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PersonaMemoryStore(storage_dir=temp_dir)
            store.remember("Celune", "my test word is moonlight", explicit=True)

            self.assertTrue(
                (Path(temp_dir) / "celune" / "memory" / "records.json").is_file()
            )

    def test_automatic_memory_extracts_project_context(self) -> None:
        """Verify recurring project information can be stored automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PersonaMemoryStore(storage_dir=temp_dir)
            saved = store.remember_from_user_message(
                "Celune",
                "my project is the lighthouse refactor",
            )

            self.assertEqual(len(saved), 1)
            self.assertEqual(
                saved[0].content,
                "The user's project is the lighthouse refactor",
            )
            self.assertEqual(saved[0].explicit, False)

    def test_memory_retrieval_is_character_specific(self) -> None:
        """Verify one character cannot read another character's memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            store.remember("Celune", "my test word is moonlight", explicit=True)
            store.remember("Mirelle", "my test word is starlight", explicit=True)

            celune = store.retrieve("Celune", "what is my test word?")
            mirelle = store.retrieve("Mirelle", "what is my test word?")

            self.assertEqual(
                [record.content for record in celune],
                ["my test word is moonlight"],
            )
            self.assertEqual(
                [record.content for record in mirelle],
                ["my test word is starlight"],
            )

    def test_low_value_filler_is_not_saved_automatically(self) -> None:
        """Verify greetings do not become long-term memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PersonaMemoryStore(storage_dir=temp_dir)
            saved = store.remember_from_user_message("Celune", "hello there")

            self.assertEqual(saved, [])
            self.assertEqual(store.load_records("Celune"), [])

    def test_retrieval_updates_last_used_timestamp(self) -> None:
        """Verify successful retrieval refreshes long-term memory usage time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            first = store.remember("Celune", "my project is the lighthouse refactor")
            assert first is not None

            records_before = store.load_records("Celune")
            self.assertEqual(records_before[0].last_used_at, first.last_used_at)

            retrieved = store.retrieve("Celune", "tell me about my project")

            self.assertEqual(len(retrieved), 1)
            records_after = store.load_records("Celune")
            self.assertNotEqual(
                records_after[0].last_used_at,
                records_before[0].last_used_at,
            )

    def test_semantic_retrieval_matches_rephrased_request(self) -> None:
        """Verify semantic retrieval can match a memory when wording differs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = "The user's favorite color is blue"
            request = "What shade do I like the most?"
            store = StubEmbeddingMemoryStore(
                storage_dir=temp_dir,
                embedding_map={
                    request: (1.0, 0.0, 0.0),
                    memory: (0.98, 0.02, 0.0),
                },
            )
            store.remember("Celune", memory, explicit=True)

            retrieved = store.retrieve("Celune", request)

            self.assertEqual([record.content for record in retrieved], [memory])

    def test_semantic_retrieval_rejects_unrelated_memory(self) -> None:
        """Verify unrelated memories are ignored when semantic similarity is low."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = "The user's favorite color is blue"
            request = "Do I have a dentist appointment tomorrow?"
            store = StubEmbeddingMemoryStore(
                storage_dir=temp_dir,
                embedding_map={
                    request: (1.0, 0.0, 0.0),
                    memory: (0.0, 1.0, 0.0),
                },
            )
            store.remember("Celune", memory, explicit=True)

            self.assertEqual(store.retrieve("Celune", request), [])

    def test_fallback_retrieval_still_works_when_embeddings_are_unavailable(
        self,
    ) -> None:
        """Verify the legacy overlap matcher remains available as a safe fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            store.remember("Celune", "my project is the lighthouse refactor")

            retrieved = store.retrieve("Celune", "tell me about my project")

            self.assertEqual(len(retrieved), 1)
            self.assertEqual(
                retrieved[0].content,
                "my project is the lighthouse refactor",
            )

    def test_fallback_retrieval_survives_missing_offline_embedding_cache(
        self,
    ) -> None:
        """Verify Hugging Face offline cache misses fall back to token overlap."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PersonaMemoryStore(storage_dir=temp_dir)
            store.remember("Celune", "my project is the lighthouse refactor")

            with mock.patch(
                "celune.persona.memory.AutoTokenizer.from_pretrained",
                side_effect=OSError("offline cache missing"),
            ):
                retrieved = store.retrieve("Celune", "tell me about my project")

            self.assertEqual(len(retrieved), 1)
            self.assertEqual(
                retrieved[0].content,
                "my project is the lighthouse refactor",
            )

    def test_semantic_similarity_threshold_controls_retrieval(self) -> None:
        """Verify the configured semantic threshold gates borderline matches."""
        with (
            tempfile.TemporaryDirectory() as strict_dir,
            tempfile.TemporaryDirectory() as relaxed_dir,
        ):
            memory = "The user prefers tabs for indentation"
            request = "Should I keep that formatting habit?"
            embedding_map: dict[str, tuple[float, ...]] = {
                request: (1.0, 0.0, 0.0),
                memory: (0.65, np.sqrt(1 - (0.65**2)), 0.0),
            }
            strict_store = StubEmbeddingMemoryStore(
                storage_dir=strict_dir,
                semantic_similarity_threshold=0.7,
                embedding_map=embedding_map,
            )
            strict_store.remember("Celune", memory)
            self.assertEqual(strict_store.retrieve("Celune", request), [])

            relaxed_store = StubEmbeddingMemoryStore(
                storage_dir=relaxed_dir,
                semantic_similarity_threshold=0.6,
                embedding_map=embedding_map,
            )
            relaxed_store.remember("Celune", memory)
            retrieved = relaxed_store.retrieve("Celune", request)
            self.assertEqual([record.content for record in retrieved], [memory])
