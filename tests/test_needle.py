# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's inference-only PyTorch Needle implementation."""

from typing import cast
from pathlib import Path

from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from celune.typing.agent import AgentTool
from celune.agent.needle_model import NeedleModel, NeedleConfig
from celune.agent.needle import (
    NeedleHandler,
    _parse_selection,
    convert_needle_safetensors,
)


class TestNeedleModel:  # pylint: disable=attribute-defined-outside-init
    """Verify the model's inference and cache semantics."""

    def setup_method(self) -> None:
        """Create a small deterministic model for fast structural checks."""
        torch.manual_seed(7)
        self.config = NeedleConfig(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_kv_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            max_seq_len=16,
        )
        self.model = NeedleModel(self.config).eval()

    def test_incremental_decoder_matches_full_decoder(self) -> None:
        """Verify KV caching preserves the full decoder's next-token logits."""
        source = torch.tensor([[2, 5, 9, 11]], dtype=torch.long)
        decoder_tokens = torch.tensor([[1, 4, 8]], dtype=torch.long)
        encoder_output = self.model.encode(source)

        full_logits, _ = self.model.decode_step(
            decoder_tokens,
            encoder_output,
        )
        cached_logits: list[torch.Tensor] = []
        past = None
        for index in range(decoder_tokens.shape[1]):
            logits, past = self.model.decode_step(
                decoder_tokens[:, index : index + 1],
                encoder_output,
                past,
            )
            cached_logits.append(logits[:, -1:])

        torch.testing.assert_close(
            full_logits,
            torch.cat(cached_logits, dim=1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_generation_respects_decoder_limit(self) -> None:
        """Verify generation cannot request positions beyond the RoPE table."""
        source = torch.tensor([[2, 5, 9]], dtype=torch.long)

        generated = self.model.generate(source, max_new_tokens=100)

        assert generated.shape[1] <= self.config.max_seq_len


class TestNeedleHandler:
    """Verify checkpoint conversion and tool-call boundary behavior."""

    def test_checkpoint_conversion_removes_hugging_face_prefix(self) -> None:
        """Verify a safetensors checkpoint becomes a loadable PyTorch state dict."""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.safetensors"
            destination = Path(directory) / "converted.pt"
            save_file(
                {"model.embed_tokens.weight": torch.ones((2, 2))},
                str(source),
            )
            convert_needle_safetensors(source, destination)
            state = torch.load(destination, map_location="cpu", weights_only=True)
            assert set(state) == {"embed_tokens.weight"}
            torch.testing.assert_close(
                state["embed_tokens.weight"],
                torch.ones((2, 2)),
            )

    def test_registered_tools_and_json_calls_are_normalized(self) -> None:
        """Verify Celune tool names survive Needle's snake-case convention."""
        tool = cast(
            AgentTool,
            SimpleNamespace(name="SetTimer", description="Set a timer."),
        )

        catalog = NeedleHandler.catalog_for_tools([tool])
        selection = _parse_selection(
            '<tool_call>[{"name":"set_timer","arguments":{"minutes":5}}]',
            {"set_timer": "SetTimer"},
        )

        assert catalog[0]["name"] == "SetTimer"
        assert selection == [{"name": "SetTimer", "arguments": {"minutes": 5}}]
