# SPDX-License-Identifier: MIT
"""PyTorch Needle handler for Celune's local tool-selection runtime.

The handler owns only model inference and structured tool-call parsing. Celune
retains tool registration, permission checks, execution, and final Persona
response generation in the main runtime.
"""

from __future__ import annotations

import os
import re
import json
from uuid import uuid4
from pathlib import Path
from typing import Union, Optional, cast
from collections.abc import Mapping, Sequence

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

from ..typing.common import JSONSerializable
from ..paths import huggingface_hub_cache_dir
from ..exceptions import NeedleSelectionError
from .needle_model import NeedleModel, NeedleConfig
from .needle_checkpoint import (
    NEEDLE_MODEL_ID,
    NEEDLE_CONFIG_FILE,
    NEEDLE_PICKLE_FILE,
    NEEDLE_WEIGHTS_FILE,
    NEEDLE_MODEL_REVISION,
    NEEDLE_TOKENIZER_FILE,
    NeedlePickleConverter,
    NeedlePreparedCheckpoint,
    prepare_needle_checkpoint,
)
from ..typing.agent import (
    ToolCall,
    AgentTool,
    AgentOutput,
    AgentContext,
    NeedleToolCall,
    AgentToolSchema,
    NeedleToolCatalog,
    ValidatedToolCall,
    AgentToolValueType,
    NeedleToolParameter,
    NeedleToolSelection,
    NeedleToolDefinition,
    AgentToolArgumentSchema,
    NeedleToolParameterSpec,
)

NEEDLE_TOOL_CALL_TOKEN_ID = 4
NEEDLE_TOOLS_TOKEN_ID = 5

__all__ = [
    "NEEDLE_CONFIG_FILE",
    "NEEDLE_MODEL_ID",
    "NEEDLE_MODEL_REVISION",
    "NEEDLE_PICKLE_FILE",
    "NEEDLE_TOKENIZER_FILE",
    "NEEDLE_WEIGHTS_FILE",
    "NeedleHandler",
    "NeedleTokenizer",
    "NeedleToolSelector",
    "convert_needle_safetensors",
]


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
        previous_name = original_names.get(normalized_name)
        if previous_name is not None and previous_name != name:
            raise NeedleSelectionError(
                "Needle tool names collide after normalization: "
                f"{previous_name!r} and {name!r}"
            )
        copy = dict(tool)
        copy["name"] = normalized_name
        normalized.append(cast(NeedleToolDefinition, copy))
        original_names[normalized_name] = name
    return normalized, original_names


def _parse_single_selection(
    text: str,
    original_names: Mapping[str, str],
) -> NeedleToolCall:
    """Parse exactly one JSON tool call from a Needle response."""
    candidates = _json_candidates(text.replace("<tool_call>", ""))
    if not candidates:
        raise NeedleSelectionError("Needle returned malformed or empty JSON")
    candidate = candidates[0]
    values = candidate if isinstance(candidate, list) else [candidate]
    if len(values) != 1:
        raise NeedleSelectionError("Needle returned multiple tool calls")
    value = values[0]
    if not isinstance(value, dict):
        raise NeedleSelectionError("Needle returned a malformed tool call")
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise NeedleSelectionError("Needle returned malformed tool arguments")
    return NeedleToolCall(
        name=original_names.get(name, name),
        arguments=cast(dict[str, JSONSerializable], arguments),
    )


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
    """Parse the legacy list-shaped Needle response without agent enforcement."""
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
    temporary = destination_path.with_name(
        f".{destination_path.name}.{uuid4().hex}.tmp"
    )
    try:
        torch.save(converted_state, temporary)
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()


class NeedleHandler:
    """Load and run the PyTorch Needle tool selector."""

    def __init__(
        self,
        model: NeedleModel,
        tokenizer: NeedleTokenizer,
        device: torch.device,
        prepared_checkpoint: Optional[NeedlePreparedCheckpoint] = None,
    ) -> None:
        self.model: Optional[NeedleModel] = model
        self.tokenizer = tokenizer
        self.device = device
        self.prepared_checkpoint = prepared_checkpoint

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = NEEDLE_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Path] = None,
        *,
        revision: str = NEEDLE_MODEL_REVISION,
        source_filename: str = NEEDLE_WEIGHTS_FILE,
        pickle_converter: Optional[NeedlePickleConverter] = None,
    ) -> NeedleHandler:
        """Prepare, strictly load, and initialize one Needle checkpoint."""
        cache_root = cache_dir or huggingface_hub_cache_dir(create=True)
        cache = str(cache_root)
        prepared: NeedlePreparedCheckpoint = prepare_needle_checkpoint(
            model_id,
            revision,
            cache_root,
            source_filename=source_filename,
            pickle_converter=pickle_converter,
        )
        tokenizer_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=NEEDLE_TOKENIZER_FILE,
                revision=revision,
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
            or converted_path.stat().st_mtime < prepared.path.stat().st_mtime
        ):
            convert_needle_safetensors(prepared.path, converted_path)

        config_data = cast(
            dict[str, JSONSerializable],
            json.loads(prepared.config_path.read_text(encoding="utf-8")),
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
        return cls(
            model,
            NeedleTokenizer(tokenizer_path),
            selected_device,
            prepared,
        )

    @staticmethod
    def catalog_for_tools(
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        available_only: bool = False,
    ) -> NeedleToolCatalog:
        """Convert registered tools and their typed schemas for Needle."""
        catalog: NeedleToolCatalog = []
        for tool in tools:
            schema = None if schemas is None else schemas.get(tool.name)
            if available_only and schema is not None and not schema.available:
                continue
            parameters: dict[str, NeedleToolParameter] = {}
            if schema is not None:
                for argument in schema.arguments:
                    parameter: NeedleToolParameterSpec = {
                        "type": argument.value_type.value,
                        "description": argument.description,
                        "required": argument.required,
                    }
                    if argument.item_type is not None:
                        parameter["item_type"] = argument.item_type.value
                    parameters[argument.name] = parameter
            catalog.append(
                {
                    "name": tool.name,
                    "description": (
                        schema.description
                        if schema is not None and schema.description
                        else tool.description
                    ),
                    "parameters": parameters,
                }
            )
        return catalog

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

    @torch.inference_mode()
    def select_one_tool(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = 96,
    ) -> NeedleToolCall:
        """Return exactly one strictly parsed Needle tool call."""
        input_ids, original_names = self.build_encoder_input(query, tools)
        generated = self._require_model().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        text = self.tokenizer.decode(generated[0].tolist())
        return _parse_single_selection(text, original_names)

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


class NeedleToolSelector:
    """Adapt one verified Needle handler to the existing agent selector boundary."""

    def __init__(
        self,
        handler: NeedleHandler,
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        max_new_tokens: int = 96,
    ) -> None:
        """Create a selector that validates Needle output against registered tools."""
        if max_new_tokens <= 0:
            raise ValueError("Needle max_new_tokens must be positive")
        self.handler = handler
        self.tools = tuple(tools)
        provided_schemas = tuple((schemas or {}).values())
        resolved_schemas: dict[str, AgentToolSchema] = {}
        for tool in self.tools:
            schema = self._schema_for_tool(tool, schemas, provided_schemas)
            if schema is not None:
                resolved_schemas[tool.name] = schema
        self.schemas = resolved_schemas
        self.max_new_tokens = max_new_tokens
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        if len(self._tools_by_name) != len(self.tools):
            raise ValueError("Needle tool names must be unique")

    @staticmethod
    def _schema_for_tool(
        tool: AgentTool,
        schemas: Optional[Mapping[str, AgentToolSchema]],
        provided_schemas: tuple[AgentToolSchema, ...],
    ) -> Optional[AgentToolSchema]:
        """Resolve a schema by registered name or its normalized tool ID."""
        if schemas is not None:
            schema = schemas.get(tool.name)
            if schema is not None:
                return schema
        normalized_name = _snake_case(tool.name)
        return next(
            (
                schema
                for schema in provided_schemas
                if schema.tool_id == normalized_name
            ),
            None,
        )

    @classmethod
    def from_pretrained(
        cls,
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        model_id: str = NEEDLE_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Path] = None,
        revision: str = NEEDLE_MODEL_REVISION,
        source_filename: str = NEEDLE_WEIGHTS_FILE,
        pickle_converter: Optional[NeedlePickleConverter] = None,
        max_new_tokens: int = 96,
    ) -> NeedleToolSelector:
        """Load the verified Needle handler and build one typed selector adapter."""
        handler = NeedleHandler.from_pretrained(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir,
            revision=revision,
            source_filename=source_filename,
            pickle_converter=pickle_converter,
        )
        return cls(
            handler,
            tools,
            schemas=schemas,
            max_new_tokens=max_new_tokens,
        )

    def __call__(
        self,
        context: AgentContext,
        output: AgentOutput,
        /,
    ) -> Optional[ToolCall]:
        """Select and schema-validate one tool without executing it."""
        intent = output.get("response")
        if not isinstance(intent, str) or not intent.strip():
            raise NeedleSelectionError(
                "Needle requires a non-empty natural-language action intent"
            )
        catalog = self.handler.catalog_for_tools(
            self.tools,
            schemas=self.schemas,
            available_only=True,
        )
        selection = self.handler.select_one_tool(
            intent,
            catalog,
            max_new_tokens=self.max_new_tokens,
        )
        if context.task is not None and context.task.is_terminal:
            return None
        return self._validate_selection(selection)

    def _validate_selection(self, selection: NeedleToolCall) -> ValidatedToolCall:
        """Validate one restored canonical call against registered schemas."""
        tool = self._tools_by_name.get(selection["name"])
        if tool is None:
            raise NeedleSelectionError(
                f"Needle selected an unknown tool: {selection['name']}"
            )
        schema = self.schemas.get(tool.name)
        if schema is None:
            schema = AgentToolSchema(
                tool_id=tool.name,
                display_name=tool.name,
                description=tool.description,
            )
        if not schema.available:
            raise NeedleSelectionError(
                f"Needle selected an unavailable tool: {selection['name']}"
            )
        self._validate_arguments(selection["arguments"], schema)
        return {
            "id": f"needle-{uuid4().hex}",
            "name": tool.name,
            "arguments": selection["arguments"],
            "tool_id": schema.tool_id,
            "behavior": schema.behavior,
            "danger": schema.danger,
            "approval_required": schema.approval_required,
        }

    @staticmethod
    def _validate_arguments(
        arguments: Mapping[str, JSONSerializable],
        schema: AgentToolSchema,
    ) -> None:
        """Reject unknown, missing, or type-invalid registered arguments."""
        declared = {argument.name: argument for argument in schema.arguments}
        unknown = sorted(set(arguments) - set(declared))
        if unknown:
            raise NeedleSelectionError(
                "Needle returned unknown arguments: " + ", ".join(unknown)
            )
        missing = sorted(
            argument.name
            for argument in schema.arguments
            if argument.required and argument.name not in arguments
        )
        if missing:
            raise NeedleSelectionError(
                "Needle omitted required arguments: " + ", ".join(missing)
            )
        for name, value in arguments.items():
            argument = declared[name]
            if not NeedleToolSelector._matches_type(value, argument):
                raise NeedleSelectionError(
                    f"Needle argument '{name}' does not match its registered schema"
                )

    @staticmethod
    def _matches_type(
        value: JSONSerializable,
        argument: AgentToolArgumentSchema,
    ) -> bool:
        """Return whether one JSON value matches one typed argument schema."""
        value_type = argument.value_type
        if value_type == AgentToolValueType.STRING:
            return isinstance(value, str)
        if value_type == AgentToolValueType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == AgentToolValueType.NUMBER:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if value_type == AgentToolValueType.BOOLEAN:
            return isinstance(value, bool)
        if value_type == AgentToolValueType.OBJECT:
            return isinstance(value, dict)
        if value_type == AgentToolValueType.ARRAY:
            if not isinstance(value, list):
                return False
            item_type = argument.item_type
            if item_type is None:
                return False
            return all(
                NeedleToolSelector._matches_value_type(item, item_type)
                for item in value
            )
        return False

    @staticmethod
    def _matches_value_type(
        value: JSONSerializable,
        value_type: AgentToolValueType,
    ) -> bool:
        """Return whether one array item matches its declared primitive type."""
        if value_type == AgentToolValueType.STRING:
            return isinstance(value, str)
        if value_type == AgentToolValueType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == AgentToolValueType.NUMBER:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if value_type == AgentToolValueType.BOOLEAN:
            return isinstance(value, bool)
        if value_type == AgentToolValueType.OBJECT:
            return isinstance(value, dict)
        if value_type == AgentToolValueType.ARRAY:
            return isinstance(value, list)
        return False
