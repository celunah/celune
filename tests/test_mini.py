# SPDX-License-Identifier: Apache-2.0
"""Tests for the Pocket TTS backend adapter."""

import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest import mock

from .support import mock_mini_backend


def test_mini_generated_config_matches_snapshot_language_variant(
    tmp_path: Path,
) -> None:
    """Use the 24-layer French config when the snapshot contains French 24L weights."""
    with mock_mini_backend() as mini_cls:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "french.yaml").write_text(
            "flow_lm:\n  lookup_table: {}\n  transformer:\n    num_layers: 6\n",
            encoding="utf-8",
        )
        (config_dir / "french_24l.yaml").write_text(
            "flow_lm:\n  lookup_table: {}\n  transformer:\n    num_layers: 24\n",
            encoding="utf-8",
        )
        config_module = ModuleType("pocket_tts.utils.config")
        config_module.__dict__["CONFIGS_DIR"] = config_dir
        utils_module = ModuleType("pocket_tts.utils")
        utils_module.__dict__["__path__"] = []

        language_dir = tmp_path / "snapshot" / "languages" / "french_24l"
        language_dir.mkdir(parents=True)
        (language_dir / "model.safetensors").touch()
        (language_dir / "tokenizer.json").touch()
        runtime_model = mock.Mock()
        generated_configs: list[str] = []

        def load_fake_model(
            *, config: Path, temp: float, lsd_decode_steps: int
        ) -> mock.Mock:
            del temp, lsd_decode_steps
            generated_configs.append(config.read_text(encoding="utf-8"))
            return runtime_model

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "pocket_tts.utils": utils_module,
                    "pocket_tts.utils.config": config_module,
                },
            ),
            mock.patch("celune.backends.tts.mini.temp_data_dir", return_value=tmp_path),
            mock.patch(
                "celune.backends.tts.mini.require_model_consent",
                return_value=SimpleNamespace(revision="revision"),
            ),
            mock.patch("celune.backends.tts.mini.Mini._validate_refs"),
            mock.patch.object(
                mini_cls,
                "model_is_available_locally",
                return_value=(True, str(tmp_path / "snapshot")),
            ),
            mock.patch(
                "celune.backends.tts.mini.TTSModel.load_model",
                new=mock.Mock(side_effect=load_fake_model),
                create=True,
            ),
        ):
            backend = mini_cls(
                log=lambda _message, _severity="info": None,
                quantize=False,
            )
            assert backend.load_model("fake/model", lang="fr") is runtime_model
            assert not backend.should_reload_for_language("fr")
            backend.unload_model(release_cuda_cache=False)
            assert backend.should_reload_for_language("fr")

        generated_config = generated_configs[0]
        assert "num_layers: 24" in generated_config
        assert "num_layers: 6" not in generated_config
        assert str(language_dir / "model.safetensors") in generated_config
