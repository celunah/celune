# SPDX-License-Identifier: MIT
"""Tests for the Pocket TTS backend cleanup behavior."""

import tempfile
from typing import cast
from pathlib import Path
from unittest import mock, TestCase
from types import SimpleNamespace

from pocket_tts import TTSModel

from celune.backends.tts.mini import Mini


class MiniBackendTests(TestCase):
    """Verify Mini backend lifecycle cleanup stays tidy."""

    def test_language_mapping_normalizes_aliases_and_falls_back_to_english(
        self,
    ) -> None:
        """Verify Pocket TTS language selection uses supported variants only."""
        backend = Mini(log=lambda *_args, **_kwargs: None)
        self.assertEqual(backend.resolve_generation_language("fr"), "fr")
        self.assertEqual(backend.resolve_generation_language("pt-BR"), "pt")
        self.assertEqual(backend.resolve_generation_language("english"), "en")
        self.assertEqual(backend.resolve_generation_language("pl"), "en")
        self.assertEqual(backend.resolve_language_name("pl"), "english")

    def test_unload_model_removes_generated_config_file(self) -> None:
        """Verify Pocket TTS generated configs are deleted on unload."""
        backend = Mini(log=lambda *_args, **_kwargs: None)

        with tempfile.TemporaryDirectory() as temp_dir:
            pocket_dir = Path(temp_dir) / "pocket-tts"
            pocket_dir.mkdir(parents=True, exist_ok=True)
            generated = pocket_dir / "english-fixture.yaml"
            generated.write_text("weights_path: demo\n", encoding="utf-8")
            backend._generated_config_path = generated
            backend.model = cast(TTSModel, SimpleNamespace())

            backend.unload_model()

            self.assertFalse(generated.exists())
            self.assertFalse(pocket_dir.exists())

    def test_snapshot_language_dir_accepts_variant_suffixes(self) -> None:
        """Verify Pocket TTS reloads can use suffixed snapshot language folders."""
        backend = Mini(log=lambda *_args, **_kwargs: None)

        with tempfile.TemporaryDirectory() as temp_dir:
            languages_dir = Path(temp_dir) / "languages"
            (languages_dir / "french_24l").mkdir(parents=True)

            resolved = backend.resolve_snapshot_language_dir(temp_dir, "fr")

        self.assertEqual(resolved.name, "french_24l")

    def test_model_availability_accepts_variant_language_folders(self) -> None:
        """Verify cached French 24-layer snapshots are not downloaded again."""
        backend = Mini(log=lambda *_args, **_kwargs: None)

        with mock.patch(
            "celune.backends.tts.mini.cached_hf_snapshot_path",
            return_value=(True, "cached-snapshot"),
        ) as cached_snapshot:
            available = backend.model_is_available_locally("example/model", "fr")

        self.assertEqual(available, (True, "cached-snapshot"))
        cached_snapshot.assert_called_once_with(
            "example/model",
            [
                "languages/french*/model.safetensors",
                "languages/french*/tokenizer.model",
            ],
        )

    def test_template_config_accepts_code_and_suffix_variants(self) -> None:
        """Verify Pocket TTS reloads can use non-plain template config names."""
        backend = Mini(log=lambda *_args, **_kwargs: None)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            template_path = config_dir / "fr_24l.yaml"
            template_path.write_text("weights_path: demo\n", encoding="utf-8")

            with mock.patch.dict(
                "sys.modules",
                {
                    "pocket_tts.utils.config": SimpleNamespace(CONFIGS_DIR=config_dir),
                },
            ):
                resolved = backend.resolve_template_config_path("fr")

        self.assertEqual(resolved, template_path)
