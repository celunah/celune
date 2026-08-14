# SPDX-License-Identifier: MIT
"""PyTorch Needle handler for Celune's local tool-selection runtime.

The handler owns only model inference and structured tool-call parsing. Celune
retains tool registration, permission checks, execution, and final Persona
response generation in the main runtime.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor

from ..paths import huggingface_hub_cache_dir
from ..typing.agent import (
    AgentTool,
    NeedleToolCall,
    NeedleToolCatalog,
    NeedleToolDefinition,
    NeedleToolSelection,
)
from ..typing.common import JSONSerializable
from .needle_model import NeedleConfig, NeedleModel

NEEDLE_MODEL_ID = "Cactus-Compute/needle"
NEEDLE_CONFIG_FILE = "config.json"
NEEDLE_WEIGHTS_FILE = "model.safetensors"
NEEDLE_TOKENIZER_FILE = "tokenizer.model"
NEEDLE_TOOL_CALL_TOKEN_ID = 4
NEEDLE_TOOLS_TOKEN_ID = 5


class NeedleTokenizer:
    """Small SentencePiece adapter for Needle's fixed vocabulary."""

    def __init__(self, model_path: Path) -> None:
        self._processor = SentencePieceProcessor(model_file=str(model_path))

    def encode(self, value: str) -> list[int]:
        """Encode text without injecting BOS or EOS tokens."""
        return cast(list[int], self._processor.Encode(value, out_type=int))

    def decode(self, values: Sequence[int]) -> str:
        """Decode Needle token IDs into text."""
        return str(self._processor.Decode(list(values)))


def _snake_case(value: str) -> str:
    """Normalize one tool name to the spelling used during Needle training."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").casefold()


def _normalized_tools(
    tools: NeedleToolCatalog,
) -> tuple[NeedleToolCatalog, dict[str, str]]:
    """Normalize tool names and return a mapping back to Celune names."""
    normalized: NeedleToolCatalog = []
    original_names: dict[str, str] = {}
    for tool in tools:
        name = tool.get("name", "")
        normalized_name = _snake_case(name)
        if not normalized_name:
            continue
        copy = dict(tool)
        copy["name"] = normalized_name
        normalized.append(cast(NeedleToolDefinition, copy))
        original_names[normalized_name] = name
    return normalized, original_names


def _json_candidates(text: str) -> list[object]:
    """Extract JSON arrays or objects from a decoded Needle response."""
    decoder = json.JSONDecoder()
    candidates: list[object] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)
    return candidates


def _parse_selection(
    text: str,
    original_names: Mapping[str, str],
) -> NeedleToolSelection:
    """Parse and normalize Needle's JSON tool-call response."""
    for candidate in _json_candidates(text.replace("<tool_call>", "")):
        values = candidate if isinstance(candidate, list) else [candidate]
        selection: NeedleToolSelection = []
        valid = True
        for value in values:
            if not isinstance(value, dict):
                valid = False
                break
            name = value.get("name")
            arguments = value.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                valid = False
                break
            selection.append(
                NeedleToolCall(
                    name=original_names.get(name, name),
                    arguments=cast(dict[str, JSONSerializable], arguments),
                )
            )
        if valid and selection:
            return selection
    return []


def convert_needle_safetensors(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Convert Needle's safe tensor checkpoint into a PyTorch state dict.

    The checkpoint is already stored in PyTorch-compatible tensor layouts. The
    conversion removes the Hugging Face ``model.`` prefix and writes a safe
    ``weights_only`` PyTorch state dictionary for fast subsequent startup.
    """
    source_state = load_file(str(source_path), device="cpu")
    converted_state = {
        name.removeprefix("model."): value for name, value in source_state.items()
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_state, destination_path)


class NeedleHandler:
    """Load and run the PyTorch Needle tool selector."""

    def __init__(
        self,
        model: NeedleModel,
        tokenizer: NeedleTokenizer,
        device: torch.device,
    ) -> None:
        self.model: Optional[NeedleModel] = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = NEEDLE_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Path] = None,
    ) -> "NeedleHandler":
        """Download, convert, and load one Needle checkpoint."""
        cache_root = cache_dir or huggingface_hub_cache_dir(create=True)
        cache = str(cache_root)
        config_path = Path(
            hf_hub_download(
                model_id,
                NEEDLE_CONFIG_FILE,
                cache_dir=cache,
                repo_type="model",
            )
        )
        weights_path = Path(
            hf_hub_download(
                model_id,
                NEEDLE_WEIGHTS_FILE,
                cache_dir=cache,
                repo_type="model",
            )
        )
        tokenizer_path = Path(
            hf_hub_download(
                model_id,
                NEEDLE_TOKENIZER_FILE,
                cache_dir=cache,
                repo_type="model",
            )
        )
        converted_path = (
            cache_root
            / "celune-needle"
            / model_id.replace("/", "--")
            / "needle-pytorch.pt"
        )
        if (
            not converted_path.exists()
            or converted_path.stat().st_mtime < weights_path.stat().st_mtime
        ):
            convert_needle_safetensors(weights_path, converted_path)

        config_data = cast(
            dict[str, JSONSerializable],
            json.loads(config_path.read_text(encoding="utf-8")),
        )
        config = NeedleConfig.from_mapping(config_data)
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = NeedleModel(config)
        normalized_state = cast(
            dict[str, torch.Tensor],
            torch.load(converted_path, map_location="cpu", weights_only=True),
        )
        model.load_state_dict(normalized_state, strict=True)
        model.to(selected_device).eval()
        return cls(model, NeedleTokenizer(tokenizer_path), selected_device)

    @staticmethod
    def catalog_for_tools(tools: Sequence[AgentTool]) -> NeedleToolCatalog:
        """Convert Celune's registered tools into Needle's catalog format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {},
            }
            for tool in tools
        ]

    def _require_model(self) -> NeedleModel:
        """Return the loaded model or report that this handler was closed."""
        if self.model is None:
            raise RuntimeError("NeedleHandler is closed")
        return self.model

    def build_encoder_input(
        self,
        query: str,
        tools: NeedleToolCatalog,
    ) -> tuple[torch.Tensor, dict[str, str]]:
        """Build the documented query-separator-tools encoder sequence."""
        model = self._require_model()
        normalized_tools, original_names = _normalized_tools(tools)
        tools_json = json.dumps(
            normalized_tools,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        query_tokens = self.tokenizer.encode(query)
        query_tokens = query_tokens[: model.config.max_seq_len - 2]
        remaining = model.config.max_seq_len - len(query_tokens) - 1
        tool_tokens = self.tokenizer.encode(tools_json)[: max(0, remaining)]
        values = query_tokens + [NEEDLE_TOOLS_TOKEN_ID] + tool_tokens
        return (
            torch.tensor([values], dtype=torch.long, device=self.device),
            original_names,
        )

    @torch.inference_mode()
    def select_tools(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = 96,
    ) -> NeedleToolSelection:
        """Return validated-looking tool calls selected by Needle."""
        input_ids, original_names = self.build_encoder_input(query, tools)
        generated = self._require_model().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        text = self.tokenizer.decode(generated[0].tolist())
        return _parse_selection(text, original_names)

    def select_registered_tools(
        self,
        query: str,
        tools: Sequence[AgentTool],
        max_new_tokens: int = 96,
    ) -> NeedleToolSelection:
        """Select calls from Celune's registered local tools."""
        return self.select_tools(
            query,
            self.catalog_for_tools(tools),
            max_new_tokens=max_new_tokens,
        )

    def close(self) -> None:
        """Release model references and best-effort accelerator memory."""
        self.model = None
        if self.device.type == "cuda":
            with torch.no_grad():
                torch.cuda.empty_cache()
