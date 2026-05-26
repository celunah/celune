# SPDX-License-Identifier: MIT
"""Tests for Persona long-term memory persistence and retrieval."""

import tempfile
from unittest import TestCase

from celune.persona.memory import PersonaMemoryStore


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
            store = PersonaMemoryStore(storage_dir=temp_dir)
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
            store = PersonaMemoryStore(storage_dir=temp_dir)
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
