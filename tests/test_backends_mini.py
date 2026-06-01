# SPDX-License-Identifier: MIT
"""Tests for the Pocket TTS backend cleanup behavior."""

import tempfile
from pathlib import Path
from unittest import TestCase

from celune.backends.mini import Mini


class MiniBackendTests(TestCase):
    """Verify Mini backend lifecycle cleanup stays tidy."""

    def test_unload_model_removes_generated_config_file(self) -> None:
        """Verify Pocket TTS generated configs are deleted on unload."""
        backend = Mini(log=lambda *_args, **_kwargs: None)

        with tempfile.TemporaryDirectory() as temp_dir:
            pocket_dir = Path(temp_dir) / "pocket-tts"
            pocket_dir.mkdir(parents=True, exist_ok=True)
            generated = pocket_dir / "english-fixture.yaml"
            generated.write_text("weights_path: demo\n", encoding="utf-8")
            backend._generated_config_path = generated
            backend.model = object()

            backend.unload_model()

            self.assertFalse(generated.exists())
            self.assertFalse(pocket_dir.exists())
