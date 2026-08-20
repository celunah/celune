# SPDX-License-Identifier: MIT
"""Tests for Persona character app-data paths and Markdown overrides."""

import tempfile
from pathlib import Path
from unittest import TestCase, mock

from celune.persona.paths import (
    persona_memory_dir,
    persona_character_dir,
    persona_character_slug,
    persona_override_files,
)


class PersonaPathTests(TestCase):
    """Verify safe Persona character paths and supported override loading."""

    def test_character_names_use_stable_safe_slugs(self) -> None:
        """Verify character names become safe, stable directory identifiers."""
        self.assertEqual(persona_character_slug("Celune"), "celune")
        self.assertEqual(persona_character_slug("A/B: Test"), "a-b-test")
        self.assertEqual(persona_character_slug("..."), "unknown")

    def test_persona_character_directories_use_app_data_root(self) -> None:
        """Verify character and memory directories use the Persona app-data root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch(
                "celune.persona.paths.persona_data_dir",
                return_value=root,
            ):
                self.assertEqual(
                    persona_character_dir("Celune"),
                    root / "celune",
                )
                self.assertEqual(
                    persona_memory_dir("Celune"),
                    root / "celune" / "memory",
                )

    def test_persona_override_files_only_reads_supported_non_empty_markdown(
        self,
    ) -> None:
        """Verify debug overrides ignore unsupported, missing, and empty files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "celune"
            root.mkdir(parents=True)
            (root / "personality.md").write_text(
                "Override personality.",
                encoding="utf-8",
            )
            (root / "examples.md").write_text("   ", encoding="utf-8")
            (root / "notes.md").write_text("Ignore this.", encoding="utf-8")

            with mock.patch(
                "celune.persona.paths.persona_data_dir",
                return_value=Path(temp_dir),
            ):
                self.assertEqual(
                    persona_override_files("Celune"),
                    {"personality.md": "Override personality."},
                )
