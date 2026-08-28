# SPDX-License-Identifier: Apache-2.0
"""Inference-only PyTorch implementation of the Cactus Needle model.

This implementation follows the published Needle encoder-decoder architecture
and keeps the module names aligned with the Hugging Face ``model.safetensors``
checkpoint.  It intentionally contains no JAX, Flax, training, or quantization
code.
"""

from __future__ import annotations

import math
from typing import Optional, cast
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...typing.common import JSONSerializable


@dataclass(frozen=True)
class NeedleConfig:
    """Architecture settings read from the published Needle configuration."""

    vocab_size: int = 8192
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_encoder_layers: int = 12
    num_decoder_layers: int = 8
    max_seq_len: int = 1024
    pad_token_id: int = 0
    decoder_start_token_id: int = 1
    eos_token_id: int = 1
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        """Validate dimensions required by grouped-query attention and RoPE."""
        if self.d_model % self.num_heads:
            raise ValueError("Needle d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Needle num_heads must be divisible by num_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("Needle max_seq_len must be positive")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, JSONSerializable],
    ) -> NeedleConfig:
        """Build configuration from a Hugging Face JSON mapping."""
        max_seq_len = _config_int(values, "max_seq_len", cls.max_seq_len)
        if "max_seq_len" not in values:
            max_seq_len = _config_int(
                values,
                "max_position_embeddings",
                max_seq_len,
            )
        num_heads = _config_int(
            values,
            "num_heads",
            _config_int(values, "num_attention_heads", cls.num_heads),
        )
        num_kv_heads = _config_int(
            values,
            "num_kv_heads",
            _config_int(values, "num_key_value_heads", cls.num_kv_heads),
        )
        return cls(
            vocab_size=_config_int(values, "vocab_size", cls.vocab_size),
            d_model=_config_int(values, "d_model", cls.d_model),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_encoder_layers=_config_int(
                values,
                "num_encoder_layers",
                cls.num_encoder_layers,
            ),
            num_decoder_layers=_config_int(
                values,
                "num_decoder_layers",
                cls.num_decoder_layers,
            ),
            max_seq_len=max_seq_len,
            pad_token_id=_config_int(values, "pad_token_id", cls.pad_token_id),
            decoder_start_token_id=_config_int(
                values,
                "decoder_start_token_id",
                cls.decoder_start_token_id,
            ),
            eos_token_id=_config_int(values, "eos_token_id", cls.eos_token_id),
            rope_theta=_config_float(values, "rope_theta", cls.rope_theta),
        )

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden size."""
        return self.d_model // self.num_heads

    @property
    def kv_repeats(self) -> int:
        """Return the number of grouped-query attention repetitions."""
        return self.num_heads // self.num_kv_heads


def _config_int(
    values: Mapping[str, JSONSerializable],
    name: str,
    default: int,
) -> int:
    """Read an integer architecture setting with a safe default."""
    value = values.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _config_float(
    values: Mapping[str, JSONSerializable],
    name: str,
    default: float,
) -> float:
    """Read a floating-point architecture setting with a safe default."""
    value = values.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


class NeedleRMSNorm(nn.Module):
    """Zero-centred RMS normalization used by Needle."""

    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.zeros(dimension))

    def forward(self, value: Tensor) -> Tensor:
        """Normalize one hidden-state tensor."""
        input_dtype = value.dtype
        value_float = value.float()
        rms = torch.sqrt(value_float.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return ((1.0 + self.weight) * value_float / rms).to(input_dtype)


class NeedleRoPE(nn.Module):
    """Rotary position embeddings with explicit position slicing."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        half = config.head_dim // 2
        frequencies = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        positions = torch.arange(config.max_seq_len).float()
        angles = torch.outer(positions, frequencies)
        self.register_buffer("_cos", torch.cos(angles), persistent=False)
        self.register_buffer("_sin", torch.sin(angles), persistent=False)
        self._half_dim = half

    def forward(self, value: Tensor, start: int = 0) -> Tensor:
        """Apply rotary embeddings to ``(batch, heads, time, head_dim)``."""
        length = value.shape[2]
        cos = cast(Tensor, self._cos)[start : start + length].to(value.device)
        sin = cast(Tensor, self._sin)[start : start + length].to(value.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        first, second = value[..., : self._half_dim], value[..., self._half_dim :]
        return torch.cat(
            [first * cos - second * sin, second * cos + first * sin], dim=-1
        )


def _causal_mask(
    length: int,
    past_length: int,
    device: torch.device,
) -> Tensor:
    """Build a boolean causal mask for one decoder step or sequence."""
    rows = torch.arange(past_length, past_length + length, device=device)
    columns = torch.arange(past_length + length, device=device)
    return (rows.unsqueeze(1) >= columns.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


class NeedleAttention(nn.Module):
    """Grouped-query attention with optional causal KV caching."""

    def __init__(
        self,
        config: NeedleConfig,
        causal: bool,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.repeats = config.kv_repeats
        self.causal = causal
        kv_dimension = config.num_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dimension, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dimension, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = NeedleRMSNorm(self.head_dim)
        self.k_norm = NeedleRMSNorm(self.head_dim)

    def forward(
        self,
        query: Tensor,
        source: Tensor,
        mask: Optional[Tensor] = None,
        rope: Optional[NeedleRoPE] = None,
        rope_start: int = 0,
        past: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run attention and return the output plus the current KV cache."""
        batch, query_length, _ = query.shape
        key_length = source.shape[1]
        q = self.q_proj(query).reshape(
            batch, query_length, self.num_heads, self.head_dim
        )
        k = self.k_proj(source).reshape(
            batch, key_length, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(source).reshape(
            batch, key_length, self.num_kv_heads, self.head_dim
        )
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)

        if rope is not None:
            q = rope(q, rope_start)
            k = rope(k, rope_start)

        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)

        present = (k, v)
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
        output = torch.matmul(probabilities, v)
        output = output.transpose(1, 2).reshape(batch, query_length, -1)
        return self.out_proj(output), present


class NeedleEncoderLayer(nn.Module):
    """One gated encoder self-attention block."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.input_layernorm = NeedleRMSNorm(config.d_model)
        self.self_attn = NeedleAttention(config, causal=False)

    def forward(self, value: Tensor, rope: NeedleRoPE) -> Tensor:
        """Run the encoder block."""
        residual = value
        normalized = self.input_layernorm(value)
        attention, _ = self.self_attn(normalized, normalized, rope=rope)
        return residual + torch.sigmoid(self.attn_gate) * attention


class NeedleEncoder(nn.Module):
    """Needle's stacked self-attention encoder."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [NeedleEncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.final_norm = NeedleRMSNorm(config.d_model)
        self.rope = NeedleRoPE(config)
        self.embed_scale = math.sqrt(config.d_model)

    def forward(self, embeddings: nn.Embedding, input_ids: Tensor) -> Tensor:
        """Encode query and tool-definition token IDs."""
        value = embeddings(input_ids) * self.embed_scale
        for layer in self.layers:
            value = layer(value, self.rope)
        return self.final_norm(value)


class NeedleDecoderLayer(nn.Module):
    """One gated decoder self-attention and encoder cross-attention block."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        self.input_layernorm = NeedleRMSNorm(config.d_model)
        self.encoder_attn_layer_norm = NeedleRMSNorm(config.d_model)
        self.self_attn = NeedleAttention(config, causal=True)
        self.encoder_attn = NeedleAttention(config, causal=False)

    def forward(
        self,
        value: Tensor,
        encoder_output: Tensor,
        rope: NeedleRoPE,
        past: Optional[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run one decoder layer and return its updated KV cache."""
        past_length = 0 if past is None else past[0].shape[2]
        residual = value
        normalized = self.input_layernorm(value)
        self_output, present = self.self_attn(
            normalized,
            normalized,
            mask=_causal_mask(value.shape[1], past_length, value.device),
            rope=rope,
            rope_start=past_length,
            past=past,
        )
        value = residual + torch.sigmoid(self.self_attn_gate) * self_output

        residual = value
        normalized = self.encoder_attn_layer_norm(value)
        cross_output, _ = self.encoder_attn(normalized, encoder_output)
        value = residual + torch.sigmoid(self.cross_attn_gate) * cross_output
        return value, present


class NeedleDecoder(nn.Module):
    """Needle's autoregressive decoder with explicit KV caches."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [NeedleDecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.norm = NeedleRMSNorm(config.d_model)
        self.rope = NeedleRoPE(config)
        self.embed_scale = math.sqrt(config.d_model)

    def forward(
        self,
        embeddings: nn.Embedding,
        lm_head: nn.Linear,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
    ) -> Tensor:
        """Decode a complete teacher-forced sequence without a KV cache."""
        value = embeddings(decoder_input_ids) * self.embed_scale
        for layer in self.layers:
            value, _ = layer(
                value,
                encoder_output,
                self.rope,
                None,
            )
        return lm_head(self.norm(value).float())

    def step(
        self,
        embeddings: nn.Embedding,
        lm_head: nn.Linear,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
        past: Optional[Sequence[Optional[tuple[Tensor, Tensor]]]],
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """Decode one token and return logits plus updated layer caches."""
        value = embeddings(decoder_input_ids) * self.embed_scale
        updated: list[tuple[Tensor, Tensor]] = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past is None else past[index]
            value, present = layer(value, encoder_output, self.rope, layer_past)
            updated.append(present)
        logits = lm_head(self.norm(value).float())
        return logits, tuple(updated)


class NeedleModel(nn.Module):
    """PyTorch Needle model used only for inference."""

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = NeedleEncoder(config)
        self.decoder = NeedleDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def encode(self, input_ids: Tensor) -> Tensor:
        """Encode one or more query/tool prompts."""
        return self.encoder(self.embed_tokens, input_ids)

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        """Run one complete inference pass for a decoder token sequence."""
        encoder_output = self.encode(input_ids)
        return self.decoder(
            self.embed_tokens,
            self.lm_head,
            decoder_input_ids,
            encoder_output,
        )

    def decode_step(
        self,
        decoder_input_ids: Tensor,
        encoder_output: Tensor,
        past: Optional[Sequence[Optional[tuple[Tensor, Tensor]]]] = None,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """Decode one autoregressive step."""
        return self.decoder.step(
            self.embed_tokens,
            self.lm_head,
            decoder_input_ids,
            encoder_output,
            past,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 96,
    ) -> Tensor:
        """Greedily generate one Needle output sequence."""
        encoder_output = self.encode(input_ids)
        batch = input_ids.shape[0]
        next_input = torch.full(
            (batch, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        past: Optional[tuple[Optional[tuple[Tensor, Tensor]], ...]] = None
        generated: list[Tensor] = []
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        for _ in range(min(max_new_tokens, self.config.max_seq_len)):
            logits, past = self.decode_step(next_input, encoder_output, past)
            token = logits[:, -1].argmax(dim=-1)
            token = torch.where(
                finished,
                torch.full_like(token, self.config.eos_token_id),
                token,
            )
            generated.append(token)
            finished |= token == self.config.eos_token_id
            next_input = token.unsqueeze(1)
            if bool(finished.all()):
                break
        if not generated:
            return torch.empty((batch, 0), dtype=torch.long, device=input_ids.device)
        return torch.stack(generated, dim=1)
