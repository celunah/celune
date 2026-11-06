# SPDX-License-Identifier: Apache-2.0
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
from typing import Union, Optional, cast
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch
from sentencepiece import SentencePieceProcessor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models import NeedleModel, NeedleConfig
from ...paths import huggingface_hub_cache_dir
from .checkpoints import (
    NEEDLE_MODEL_ID,
    NEEDLE_CONFIG_FILE,
    NEEDLE_PICKLE_FILE,
    NEEDLE_WEIGHTS_FILE,
    NEEDLE_MODEL_REVISION,
    NEEDLE_TOKENIZER_FILE,
    NeedlePickleConverter,
    NeedlePreparedCheckpoint,
    resolve_expected_dtype,
    prepare_needle_checkpoint,
)
from ...exceptions import NeedleSelectionError
from ...typing.agent import (
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
from ...typing.common import JSONSerializable

NEEDLE_TOOL_CALL_TOKEN_ID = 4
NEEDLE_TOOLS_TOKEN_ID = 5
_NEEDLE_MAX_NEW_TOKENS = 256
_NEEDLE_CANDIDATE_TOOL_LIMIT = 8
_NEEDLE_CLEAR_MATCH_SCORE = 8
_NEEDLE_CLEAR_MATCH_MARGIN = 4
_NEEDLE_WORD_RE = re.compile(r"[a-z0-9]+")
_EXPLICIT_TOOL_REQUEST_RE = re.compile(
    r"^\s*(?:please\s+)?(?:use|call|run|invoke)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_NEEDLE_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "can",
        "could",
        "for",
        "give",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "the",
        "to",
        "want",
        "what",
        "would",
    }
)

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
    "parse_selection",
    "parse_single_selection",
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


def _tool_relevance_score(query: str, tool: NeedleToolDefinition) -> int:
    """Score catalog overlap with the intent, weighting names most heavily."""
    query_words = {
        word
        for word in _NEEDLE_WORD_RE.findall(query.casefold())
        if len(word) > 1 and word not in _NEEDLE_QUERY_STOP_WORDS
    }
    if not query_words:
        return 0

    name_words = set(_NEEDLE_WORD_RE.findall(tool["name"].casefold()))
    description = tool.get("description", "")
    description_words = set(_NEEDLE_WORD_RE.findall(description.casefold()))
    parameter_words: set[str] = set()
    for name, parameter in tool["parameters"].items():
        parameter_words.update(_NEEDLE_WORD_RE.findall(name.casefold()))
        if isinstance(parameter, str):
            parameter_words.update(_NEEDLE_WORD_RE.findall(parameter.casefold()))
        else:
            parameter_description = parameter.get("description", "")
            parameter_words.update(
                _NEEDLE_WORD_RE.findall(parameter_description.casefold())
            )

    return (
        3 * len(query_words & name_words)
        + 2 * len(query_words & description_words)
        + len(query_words & parameter_words)
    )


def _rank_tool_indexes(
    query: str,
    tools: NeedleToolCatalog,
) -> list[int]:
    """Order tools by intent overlap while keeping registration order for ties."""
    return sorted(
        range(len(tools)),
        key=lambda index: (-_tool_relevance_score(query, tools[index]), index),
    )


def _compact_tool_catalog(
    tools: NeedleToolCatalog,
    *,
    tool_description_indexes: set[int],
    argument_description_indexes: set[int],
) -> NeedleToolCatalog:
    """Drop lower-priority descriptions while preserving every tool schema."""
    compacted: NeedleToolCatalog = []
    for index, tool in enumerate(tools):
        compact_tool: NeedleToolDefinition = {
            "name": tool["name"],
            "parameters": {},
        }
        description = tool.get("description")
        if description and index in tool_description_indexes:
            compact_tool["description"] = description

        parameters: dict[str, NeedleToolParameter] = {}
        for name, parameter in tool["parameters"].items():
            if isinstance(parameter, str):
                parameters[name] = parameter
                continue
            parameter_spec = dict(parameter)
            if index not in argument_description_indexes:
                parameter_spec.pop("description", None)
            parameters[name] = cast(NeedleToolParameter, parameter_spec)
        compact_tool["parameters"] = parameters
        compacted.append(compact_tool)
    return compacted


def _serialize_tool_catalog(tools: NeedleToolCatalog) -> str:
    """Serialize a stable compact catalog payload for Needle's encoder."""
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def parse_single_selection(
    text: str,
    original_names: Mapping[str, str],
) -> NeedleToolCall:
    """Parse the first JSON tool call from a Needle response."""
    candidates = _json_candidates(text.replace("<tool_call>", ""))
    if not candidates:
        raise NeedleSelectionError("Needle returned malformed or empty JSON")
    candidate = candidates[0]
    if isinstance(candidate, list):
        if not candidate:
            raise NeedleSelectionError("Needle returned malformed or empty JSON")
        candidate = candidate[0]
    value = candidate
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


def parse_selection(
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
        model = NeedleModel(config).to(dtype=resolve_expected_dtype(config_data))
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
        query_tokens = self.tokenizer.encode(query)
        max_seq_len = model.config.max_seq_len
        query_budget = max(0, min(max_seq_len - 2, max_seq_len // 2))
        query_tokens = query_tokens[:query_budget]
        remaining = max(0, max_seq_len - len(query_tokens) - 1)
        tool_tokens = self.tokenizer.encode(_serialize_tool_catalog(normalized_tools))
        if len(tool_tokens) > remaining:
            ranked_indexes = _rank_tool_indexes(query, normalized_tools)
            description_limits = (
                (len(normalized_tools), min(8, len(normalized_tools))),
                (min(8, len(normalized_tools)), min(8, len(normalized_tools))),
                (min(4, len(normalized_tools)), min(4, len(normalized_tools))),
                (0, 0),
            )
            tool_tokens = []
            fitted_catalog = False
            for (
                tool_description_count,
                argument_description_count,
            ) in description_limits:
                compacted_tools = _compact_tool_catalog(
                    normalized_tools,
                    tool_description_indexes=set(
                        ranked_indexes[:tool_description_count]
                    ),
                    argument_description_indexes=set(
                        ranked_indexes[:argument_description_count]
                    ),
                )
                candidate_tokens = self.tokenizer.encode(
                    _serialize_tool_catalog(compacted_tools)
                )
                if len(candidate_tokens) <= remaining:
                    tool_tokens = candidate_tokens
                    fitted_catalog = True
                    break

            if not fitted_catalog:
                compacted_tools = _compact_tool_catalog(
                    normalized_tools,
                    tool_description_indexes=set(),
                    argument_description_indexes=set(),
                )
                selected_tools: NeedleToolCatalog = []
                for index in ranked_indexes:
                    candidate_tools = [*selected_tools, compacted_tools[index]]
                    candidate_tokens = self.tokenizer.encode(
                        _serialize_tool_catalog(candidate_tools)
                    )
                    if len(candidate_tokens) > remaining:
                        continue
                    selected_tools = candidate_tools
                    tool_tokens = candidate_tokens
                if not selected_tools:
                    raise NeedleSelectionError(
                        "Needle cannot fit any registered tool into its input limit"
                    )
                included_names = {tool["name"] for tool in selected_tools}
                original_names = {
                    name: original
                    for name, original in original_names.items()
                    if name in included_names
                }
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
        max_new_tokens: int = _NEEDLE_MAX_NEW_TOKENS,
    ) -> NeedleToolSelection:
        """Return validated-looking tool calls selected by Needle."""
        input_ids, original_names = self.build_encoder_input(query, tools)
        generated = self._require_model().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        text = self.tokenizer.decode(generated[0].tolist())
        return parse_selection(text, original_names)

    @torch.inference_mode()
    def select_one_tool(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = _NEEDLE_MAX_NEW_TOKENS,
    ) -> NeedleToolCall:
        """Return exactly one strictly parsed Needle tool call."""
        input_ids, original_names = self.build_encoder_input(query, tools)
        generated = self._require_model().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        text = self.tokenizer.decode(generated[0].tolist())
        return parse_single_selection(text, original_names)

    def select_registered_tools(
        self,
        query: str,
        tools: Sequence[AgentTool],
        max_new_tokens: int = _NEEDLE_MAX_NEW_TOKENS,
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
        max_new_tokens: int = _NEEDLE_MAX_NEW_TOKENS,
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
        max_new_tokens: int = _NEEDLE_MAX_NEW_TOKENS,
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
    ) -> Optional[Union[ValidatedToolCall, Sequence[ValidatedToolCall]]]:
        """Select and validate tool candidates without executing them."""
        intent = output.get("response")
        if not isinstance(intent, str) or not intent.strip():
            raise NeedleSelectionError(
                "Needle requires a non-empty natural-language action intent"
            )
        task_request = context.request.request.strip()
        request = (context.request.original_request or task_request).strip()
        intent = intent.strip()
        query = intent
        if request and request.casefold() not in intent.casefold():
            query = f"{request}\n{intent}"
        catalog = self.handler.catalog_for_tools(
            self.tools,
            schemas=self.schemas,
            available_only=True,
        )
        if not catalog:
            raise NeedleSelectionError("Needle has no available registered tools")
        explicit_match = _EXPLICIT_TOOL_REQUEST_RE.match(
            request
        ) or _EXPLICIT_TOOL_REQUEST_RE.match(task_request)
        explicit_name = explicit_match.group(1).casefold() if explicit_match else None
        explicit_catalog = (
            [tool for tool in catalog if tool["name"].casefold() == explicit_name]
            if explicit_name is not None
            else []
        )
        candidate_catalog = explicit_catalog or self._candidate_catalog(
            catalog,
            f"{request}\n{task_request}",
            intent,
        )
        queries = [query]
        if request and request.casefold() != query.casefold():
            queries.append(request)
        selection: NeedleToolSelection = []
        catalogs = [candidate_catalog]
        if candidate_catalog != catalog and not explicit_catalog:
            catalogs.append(catalog)
        for candidate_tools in catalogs:
            for candidate_query in queries:
                selection = self.handler.select_tools(
                    candidate_query,
                    candidate_tools,
                    max_new_tokens=self.max_new_tokens,
                )
                if context.task is not None and context.task.is_terminal:
                    return None
                if selection:
                    break
            if selection:
                break
        if not selection:
            selection = self._retry_empty_tool_selection(
                request=request,
                intent=intent,
                catalog=candidate_catalog,
            )
            if context.task is not None and context.task.is_terminal:
                return None
        if not selection and candidate_catalog != catalog and not explicit_catalog:
            selection = self._retry_empty_tool_selection(
                request=request,
                intent=intent,
                catalog=catalog,
            )
            if context.task is not None and context.task.is_terminal:
                return None
        if not selection:
            raise NeedleSelectionError(
                "Needle returned no valid tool calls for the action intent or original request"
            )
        validated: list[ValidatedToolCall] = []
        for call in selection:
            if call["name"] not in self._tools_by_name and explicit_catalog:
                call = NeedleToolCall(
                    name=candidate_catalog[0]["name"],
                    arguments=call["arguments"],
                )
            try:
                validated.append(self._validate_selection(call))
            except NeedleSelectionError as exc:
                missing = self._missing_required_arguments(call)
                if not missing or not str(exc).startswith(
                    "Needle omitted required arguments for "
                ):
                    raise
                repaired = self._retry_tool_selection(
                    context,
                    request=request,
                    intent=intent,
                    missing=missing,
                    catalog=candidate_catalog,
                )
                if context.task is not None and context.task.is_terminal:
                    return None
                if not repaired and candidate_catalog != catalog:
                    repaired = self._retry_tool_selection(
                        context,
                        request=request,
                        intent=intent,
                        missing=missing,
                        catalog=catalog,
                    )
                    if context.task is not None and context.task.is_terminal:
                        return None
                if not repaired:
                    raise NeedleSelectionError(
                        "Needle returned no tool calls after reconsidering "
                        + call["name"]
                        + " with missing required arguments: "
                        + ", ".join(missing)
                    ) from exc
                repaired_validated = [
                    self._validate_selection(repaired_call)
                    for repaired_call in repaired
                ]
                return (
                    repaired_validated[0]
                    if len(repaired_validated) == 1
                    else repaired_validated
                )
        unique_calls: list[ValidatedToolCall] = []
        seen_calls: set[tuple[str, str]] = set()
        for call in validated:
            key = (
                call["name"],
                json.dumps(
                    call["arguments"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            if key not in seen_calls:
                seen_calls.add(key)
                unique_calls.append(call)
        return unique_calls[0] if len(unique_calls) == 1 else unique_calls

    @staticmethod
    def _candidate_catalog(
        catalog: NeedleToolCatalog,
        request: str,
        intent: str,
    ) -> NeedleToolCatalog:
        """Reduce model choice to request-relevant tools when wording is clear."""
        if len(catalog) <= 1:
            return catalog

        request_scores = [_tool_relevance_score(request, tool) for tool in catalog]
        ranked_request = sorted(
            range(len(catalog)),
            key=lambda index: (-request_scores[index], index),
        )
        best_request_score = request_scores[ranked_request[0]]
        next_request_score = request_scores[ranked_request[1]]
        if (
            best_request_score >= _NEEDLE_CLEAR_MATCH_SCORE
            and best_request_score - next_request_score >= _NEEDLE_CLEAR_MATCH_MARGIN
        ):
            return [catalog[ranked_request[0]]]

        ranking_query = f"{request}\n{intent}".strip()
        ranked = _rank_tool_indexes(ranking_query, catalog)
        if not ranked or _tool_relevance_score(ranking_query, catalog[ranked[0]]) == 0:
            return catalog
        return [catalog[index] for index in ranked[:_NEEDLE_CANDIDATE_TOOL_LIMIT]]

    def _retry_tool_selection(
        self,
        context: AgentContext,
        *,
        request: str,
        intent: str,
        missing: Sequence[str],
        catalog: NeedleToolCatalog,
    ) -> NeedleToolSelection:
        """Reconsider an incomplete call against a relevant candidate catalog."""
        if not catalog:
            return []

        sections = [
            (
                "Reconsider tool selection because the previous call omitted required arguments. "
                "Choose the registered tool that best matches the original request; the previous "
                "tool may be wrong. Do not repeat an incomplete call. Use only values supported "
                "by the request, action intent, or previous tool result; never invent arguments."
            ),
            f"Original user request: {request}",
            f"Persona action intent: {intent}",
            "The previous call omitted required arguments: " + ", ".join(missing),
        ]
        if context.last_tool_result is not None:
            result_output = context.last_tool_result.get("output")
            if result_output is not None:
                serialized_result = json.dumps(
                    result_output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(serialized_result) > 1024:
                    serialized_result = serialized_result[:1024] + "…"
                sections.append("Previous tool result: " + serialized_result)
        return self.handler.select_tools(
            "\n".join(sections),
            catalog,
            max_new_tokens=self.max_new_tokens,
        )

    def _retry_empty_tool_selection(
        self,
        *,
        request: str,
        intent: str,
        catalog: NeedleToolCatalog,
    ) -> NeedleToolSelection:
        """Retry once with a compact instruction when Needle emits no calls."""
        if not catalog:
            return []
        query = "\n".join(
            (
                "Return exactly one valid call from the registered tools.",
                "Include every required argument and use only supplied facts.",
                f"User request: {request}",
                f"Action intent: {intent}",
            )
        )
        return self.handler.select_tools(
            query,
            catalog,
            max_new_tokens=self.max_new_tokens,
        )

    def _missing_required_arguments(
        self,
        selection: NeedleToolCall,
    ) -> list[str]:
        """Return required schema fields absent from one selected call."""
        schema = self.schemas.get(selection["name"])
        if schema is None:
            return []
        arguments = selection["arguments"]
        return sorted(
            argument.name
            for argument in schema.arguments
            if argument.required and argument.name not in arguments
        )

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
        arguments = self._validate_arguments(selection["arguments"], schema)
        return {
            "id": f"needle-{uuid4().hex}",
            "name": tool.name,
            "arguments": arguments,
            "tool_id": schema.tool_id,
            "behavior": schema.behavior,
            "danger": schema.danger,
            "approval_required": schema.approval_required,
        }

    @staticmethod
    def _validate_arguments(
        arguments: Mapping[str, JSONSerializable],
        schema: AgentToolSchema,
    ) -> dict[str, JSONSerializable]:
        """Validate arguments and normalize lossless integer representations."""
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
                f"Needle omitted required arguments for {schema.tool_id}: "
                + ", ".join(missing)
            )
        normalized: dict[str, JSONSerializable] = {}
        for name, value in arguments.items():
            argument = declared[name]
            if value is None and not argument.required:
                continue
            if argument.value_type == AgentToolValueType.INTEGER:
                integer_value = NeedleToolSelector._normalize_integer(value)
                if integer_value is not None:
                    value = integer_value
            if not NeedleToolSelector._matches_type(value, argument):
                if not argument.required:
                    continue
                raise NeedleSelectionError(
                    f"Needle argument '{name}' does not match its registered schema"
                )
            normalized[name] = value
        return normalized

    @staticmethod
    def _normalize_integer(value: JSONSerializable) -> Optional[int]:
        """Convert an exact integer representation to an integer value."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[+-]?\d+", stripped):
                try:
                    return int(stripped)
                except ValueError:
                    return None
        return None

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
