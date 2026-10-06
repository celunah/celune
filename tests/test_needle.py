# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's inference-only PyTorch Needle implementation."""

from typing import cast
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch
import pytest
from safetensors.torch import save_file
from celune.agent.needle.impl import (
    NeedleHandler,
    _parse_selection,
    convert_needle_safetensors,
)
from celune.typing.agent import AgentTool
from celune.agent.needle.models import NeedleModel, NeedleConfig, NeedleRoPE


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
        """Verify generation cannot request positions beyond the RoPE limit."""
        source = torch.tensor([[2, 5, 9]], dtype=torch.long)

        generated = self.model.generate(source, max_new_tokens=100)

        assert generated.shape[1] <= self.config.max_seq_len

    def test_rope_calculates_only_requested_positions(self) -> None:
        """Keep positional state independent of the configured context length."""
        rope = NeedleRoPE(self.config)
        value = torch.randn(1, self.config.num_heads, 3, self.config.head_dim)

        actual = rope(value, start=2)

        positions = torch.arange(2, 5, dtype=torch.float32)
        indices = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
        frequencies = 1.0 / (self.config.rope_theta ** (indices / self.config.head_dim))
        angles = torch.outer(positions, frequencies)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        half_dim = self.config.head_dim // 2
        first, second = value[..., :half_dim], value[..., half_dim:]
        expected = torch.cat(
            [first * cos - second * sin, second * cos + first * sin], dim=-1
        )

        torch.testing.assert_close(actual, expected)
        buffers = dict(rope.named_buffers())
        assert set(buffers) == {"_frequency_indices"}
        assert buffers["_frequency_indices"].numel() == self.config.head_dim // 2

    def test_rope_rejects_positions_beyond_configured_limit(self) -> None:
        """Keep max_seq_len as the hard position limit."""
        rope = NeedleRoPE(self.config)
        value = torch.randn(1, self.config.num_heads, 2, self.config.head_dim)

        with pytest.raises(ValueError, match="max_seq_len"):
            rope(value, start=self.config.max_seq_len - 1)

    def test_bfloat16_model_can_generate(self) -> None:
        """Keep decoder logits compatible with BF16 checkpoint weights."""
        model = NeedleModel(self.config).to(dtype=torch.bfloat16).eval()
        source = torch.tensor([[2, 5, 9]], dtype=torch.long)

        generated = model.generate(source, max_new_tokens=3)

        assert generated.shape[0] == source.shape[0]


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
