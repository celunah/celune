# SPDX-License-Identifier: MIT
"""Tests for model loading helpers."""

from types import SimpleNamespace
from typing import cast
from unittest import mock

import torch

from celune import modeling
from celune.backends.tts import CeluneBackend

from .support import CeluneTestCase


class TestModeling(CeluneTestCase):
    """Tests for lightweight modeling helpers."""

    def test_normalizer_device_follows_vram_preset(self) -> None:
        """Verify CeluneNorm device selection follows the VRAM tier."""
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert modeling.normalizer_device(None) == "cpu"
            assert modeling.normalizer_device({"vram": "high"}) == "cpu"
            assert modeling.normalizer_device({"vram": "xhigh"}) == "cuda"

    def test_load_normalizer_components_uses_v4_tokenizer_compatibility(self) -> None:
        """Verify v5 tokenizer metadata is bypassed for Transformers v4.

        Raises:
            AssertionError: Normalizer loading compatibility changes unexpectedly.
        """
        tokenizer = mock.Mock()
        llm = mock.Mock()
        backend = SimpleNamespace(
            model_is_available_locally=mock.Mock(return_value=(True, "local-model"))
        )
        log = mock.Mock()

        with (
            mock.patch("celune.vram.torch.cuda.is_available", return_value=False),
            mock.patch.object(
                modeling.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_loader,
            mock.patch.object(
                modeling.AutoModelForCausalLM,
                "from_pretrained",
                return_value=llm,
            ) as model_loader,
        ):
            loaded_tokenizer, loaded_llm = modeling.load_normalizer_components(
                log, cast(CeluneBackend, backend), {"vram": "xhigh"}
            )

        assert loaded_tokenizer is tokenizer
        assert loaded_llm is llm
        tokenizer_loader.assert_called_once_with(
            "local-model", extra_special_tokens=list(modeling.NORMALIZER_SPECIAL_TOKENS)
        )
        model_loader.assert_called_once_with(
            "local-model",
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
