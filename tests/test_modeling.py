# SPDX-License-Identifier: MIT
"""Tests for model loading helpers."""

from types import SimpleNamespace
from typing import cast
from unittest import TestCase, mock

import torch

from celune import modeling
from celune.backends import CeluneBackend


class ModelingTests(TestCase):
    """Tests for lightweight modeling helpers."""

    def test_load_normalizer_components_uses_v4_tokenizer_compatibility(self) -> None:
        """Verify v5 tokenizer metadata is bypassed for Transformers v4.

        Returns:
            None: Assertions verify tokenizer loading arguments.

        Raises:
            AssertionError: Normalizer loading compatibility changes unexpectedly.
        """
        tokenizer = mock.Mock()
        llm = object()
        backend = SimpleNamespace(
            model_is_available_locally=mock.Mock(return_value=(True, "local-model"))
        )
        log = mock.Mock()

        with (
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
                log, cast(CeluneBackend, backend)
            )

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_llm, llm)
        tokenizer_loader.assert_called_once_with("local-model", extra_special_tokens={})
        tokenizer.add_special_tokens.assert_called_once_with(
            {"additional_special_tokens": list(modeling.NORMALIZER_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        model_loader.assert_called_once_with(
            "local-model",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
