# SPDX-License-Identifier: Apache-2.0
"""Small, explicitly allowlisted tools owned by the Celune agent runtime."""

from __future__ import annotations

import os
import sys
import time
import shutil
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Callable
from typing import TYPE_CHECKING, Optional, cast

import psutil

from ..typing.modes import OperationMode
from ..typing.common import JSON, JSONSerializable
from ..typing.agent import (
    ToolCall,
    AgentTool,
    AgentContext,
    AgentToolSchema,
    AgentToolBehavior,
    AgentToolValueType,
    ToolExecutionResult,
    AgentToolDangerLevel,
    AgentToolArgumentSchema,
    AgentToolExecutionStatus,
)
from ..utils import format_error_message

if TYPE_CHECKING:
    from ..celune import Celune


class AgentStatusTool:
    """Read the current task state without changing local application state."""

    name = "read_agent_status"
    description = "Read the current Celune agent task status."

    def execute(self, call: ToolCall, context: AgentContext) -> ToolExecutionResult:
        """Return typed state for the task that owns this tool call."""
        task = context.task
        if call["arguments"]:
            return {
                "tool_call_id": call["id"],
                "output": None,
                "error": "read_agent_status does not accept arguments",
                "tool_id": self.name,
                "status": AgentToolExecutionStatus.FAILED,
            }
        if task is None:
            return {
                "tool_call_id": call["id"],
                "output": None,
                "error": "agent task context is unavailable",
                "tool_id": self.name,
                "status": AgentToolExecutionStatus.FAILED,
            }
        output: JSON = {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "state": task.state.value,
            "iterations": task.iterations,
            "generated_tokens": task.generated_tokens,
        }
        return {
            "tool_call_id": call["id"],
            "output": output,
            "error": None,
            "tool_id": self.name,
            "status": AgentToolExecutionStatus.SUCCEEDED,
        }


def agent_test_tools(engine: Optional[Celune] = None) -> tuple[AgentTool, ...]:
    """Return the narrowly allowlisted tools permitted by agent test mode."""
    return tuple(OfflineAgentTool(engine, spec) for spec in _LOCAL_TEST_SPECS)


def agent_test_tool_schemas() -> Mapping[str, AgentToolSchema]:
    """Return schemas for the read-only agent test tool allowlist."""
    return _schemas_for_specs(_LOCAL_TEST_SPECS)


OfflineToolHandler = Callable[["Celune", ToolCall, AgentContext], JSONSerializable]


@dataclass(frozen=True)
class OfflineToolSpec:
    """Definition shared by one production tool and its typed schema."""

    tool_id: str
    display_name: str
    description: str
    arguments: tuple[AgentToolArgumentSchema, ...]
    behavior: AgentToolBehavior
    danger: AgentToolDangerLevel
    handler: OfflineToolHandler
    available: bool = True
    end_task_on_success: bool = False


class LocalManagementError(RuntimeError):
    """Describe a typed local-management failure and its exact target."""

    def __init__(
        self, status: str, message: str, target: Optional[Path] = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.target = target

    def to_json(self) -> JSON:
        """Serialize the failure without exposing an ambiguous target."""
        return {
            "result": self.status,
            "message": str(self),
            "target": str(self.target) if self.target is not None else None,
        }


class OfflineAgentTool:
    """Bind one allowlisted operation to the active Celune engine."""

    def __init__(self, engine: Optional[Celune], spec: OfflineToolSpec) -> None:
        """Create one tool instance for a specific engine."""
        self._engine = engine
        self._spec = spec
        self.name = spec.tool_id
        self.description = spec.description

    def execute(self, call: ToolCall, context: AgentContext) -> ToolExecutionResult:
        """Execute the tool and normalize exceptions into typed results."""
        if self._engine is None:
            return _failure(call, self.name, "Celune engine is unavailable")
        if not self._spec.available:
            return _failure(call, self.name, "tool is currently unavailable")
        try:
            output = self._spec.handler(self._engine, call, context)
        except LocalManagementError as exc:
            log = getattr(self._engine, "log", None)
            if callable(log):
                log(
                    format_error_message(
                        f"[AGENT] tool_failed tool={self.name}",
                        exc,
                        getattr(self._engine, "log_level", "info"),
                    ),
                    "error",
                    loglevel="verbose",
                )
            return _failure(
                call,
                self.name,
                str(exc),
                output=exc.to_json(),
            )
        except Exception as exc:
            log = getattr(self._engine, "log", None)
            if callable(log):
                log(
                    format_error_message(
                        f"[AGENT] tool_failed tool={self.name}",
                        exc,
                        getattr(self._engine, "log_level", "info"),
                    ),
                    "error",
                    loglevel="verbose",
                )
            return _failure(call, self.name, str(exc))
        return {
            "tool_call_id": call["id"],
            "output": output,
            "error": None,
            "tool_id": self.name,
            "status": AgentToolExecutionStatus.SUCCEEDED,
            **({"end_task": True} if self._spec.end_task_on_success else {}),
        }


def _failure(
    call: ToolCall,
    tool_id: str,
    error: str,
    *,
    output: Optional[JSON] = None,
) -> ToolExecutionResult:
    """Build one typed tool failure."""
    return {
        "tool_call_id": call["id"],
        "output": output,
        "error": error,
        "tool_id": tool_id,
        "status": AgentToolExecutionStatus.FAILED,
    }


def _required(call: ToolCall, name: str) -> JSONSerializable:
    """Return one required tool argument."""
    value = call["arguments"].get(name)
    if value is None:
        raise ValueError(f"missing required argument '{name}'")
    return value


def _string(call: ToolCall, name: str) -> str:
    """Return one required non-empty string argument."""
    value = _required(call, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"argument '{name}' must be a non-empty string")
    return value.strip()


def _optional_string(call: ToolCall, name: str) -> Optional[str]:
    """Return one optional string argument."""
    value = call["arguments"].get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"argument '{name}' must be a string")
    return value.strip()


def _text(call: ToolCall, name: str) -> str:
    """Return one required string argument, allowing an empty payload."""
    value = _required(call, name)
    if not isinstance(value, str):
        raise TypeError(f"argument '{name}' must be a string")
    return value


def _integer(call: ToolCall, name: str, default: int) -> int:
    """Return one integer argument or its default."""
    value = call["arguments"].get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"argument '{name}' must be an integer")
    return value


def _number(call: ToolCall, name: str) -> float:
    """Return one numeric argument as a float."""
    value = _required(call, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"argument '{name}' must be numeric")
    return float(value)


def _character(engine: Celune) -> str:
    """Return the active character required by memory operations."""
    character = engine.current_character
    if not isinstance(character, str) or not character.strip():
        raise ValueError("no active character is loaded")
    return character


def _memory_store(engine: Celune):
    """Return the existing Persona memory store, creating the normal default if needed."""
    store = engine.persona_memory_store
    if store is None:
        from ..persona.memory import PersonaMemoryStore

        store = PersonaMemoryStore()
        engine.persona_memory_store = store
    return store


def _task(context: AgentContext):
    """Return the current task or reject a task-independent invocation."""
    if context.task is None:
        raise ValueError("agent task context is unavailable")
    return context.task


def _status(engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Read engine and current-task status."""
    task = context.task
    return {
        "engine_state": engine.cur_state,
        "loaded": engine.loaded,
        "sleeping": engine.sleeping,
        "mode": engine.mode,
        "input_mode": engine.input_mode,
        "backend_mode": engine.backend_mode,
        "voice": engine.current_voice,
        "character": engine.current_character,
        "task_state": task.state.value if task is not None else None,
    }


def _legacy_status(_engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Preserve the original production status-tool response shape."""
    task = _task(context)
    return {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "state": task.state.value,
        "iterations": task.iterations,
        "generated_tokens": task.generated_tokens,
    }


def _capabilities(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read capabilities of the active local runtime."""
    return {
        "persona_ready": engine.persona_ready,
        "persona_loading": engine.persona_loading,
        "voice_prompt": engine.voice_prompt_supported(),
        "needle_ready": engine.agent_needle_ready,
        "tools": cast(JSONSerializable, sorted(engine._agent_tool_schemas)),
    }


def _models(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read configured model identifiers without exposing model objects."""
    from ..persona.impl import persona_model_id

    return {
        "tts_backend": engine.tts_backend,
        "tts_model": engine.model_name,
        "persona_model": persona_model_id(engine.config),
        "normalizer_loaded": engine.llm is not None and engine.tokenizer is not None,
    }


def _locks(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read component lock ownership diagnostics."""
    snapshot = engine.component_locks.snapshot()
    return {
        component.value: owner.to_json() if owner is not None else None
        for component, owner in snapshot.items()
    }


def _audio_state(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read speech queue and playback state."""
    return {
        "state": engine.cur_state,
        "locked": engine.locked,
        "text_queue_items": engine.text_queue.qsize(),
        "audio_queue_items": engine.audio_queue.qsize(),
        "speed": engine.speed,
        "reverb": engine.reverb.strength,
        "audio_unavailable": engine.audio_unavailable,
    }


def _agent_task(_engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Read the current task contract."""
    return _task(context).to_json()


def _health(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Check prerequisites for local agent operation."""
    checks = {
        "engine_loaded": engine.loaded,
        "model_ready": engine.model_ready.is_set(),
        "persona_ready": engine.persona_ready,
        "agent_tools_registered": bool(engine._agent_tools),
        "agent_mode": engine.mode == "agent",
    }
    return {
        "healthy": all(checks.values()),
        "checks": cast(JSONSerializable, checks),
    }


def _speak(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Queue speech through the normal Celune pipeline."""
    queued = engine.say(_string(call, "text"))
    if not queued:
        raise RuntimeError("speech could not be queued")
    return {"queued": True}


def _stop_speech(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Stop speech through the normal interruption path."""
    return {"stopped": engine.force_stop_speech()}


def _set_voice(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Switch voice through the synchronized reload path."""
    voice = _string(call, "voice")
    return {"voice": voice, "changed": engine.set_voice_and_wait(voice)}


def _set_voice_prompt(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Set or clear the active voice prompt."""
    if not engine.voice_prompt_supported():
        raise ValueError("voice prompts are unavailable for the active model")
    prompt = _optional_string(call, "prompt")
    engine.voice_prompt = prompt
    return {"voice_prompt": prompt}


def _set_speed(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Set speech playback speed."""
    speed = _number(call, "speed")
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    engine.speed = speed
    return {"speed": speed}


def _set_reverb(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Set speech reverb strength."""
    strength = _number(call, "strength")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("reverb strength must be between 0 and 1")
    engine.reverb.strength = strength
    return {"reverb": strength}


def _clear_speech_queue(
    engine: Celune, _call: ToolCall, _context: AgentContext
) -> JSON:
    """Clear pending speech and audio queue items."""
    engine._clear_queue(engine.text_queue)
    engine._clear_queue(engine.audio_queue)
    return {"cleared": True}


def _set_character(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Load a named or path-based CEVOICE pack."""
    bundle = _string(call, "bundle")
    return {"bundle": bundle, "loaded": engine.set_cevoice_and_wait(bundle)}


def _character_query(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read active character and voice-pack identity."""
    return {
        "character": engine.current_character,
        "voice_bundle_is_default": engine.voice_bundle_is_default,
        "voice": engine.current_voice,
        "voice_prompt": engine.voice_prompt,
    }


def _set_mode(
    engine: Celune,
    call: ToolCall,
    _context: AgentContext,
    expected: OperationMode,
) -> JSON:
    """Change the in-memory operation mode when it matches the tool contract."""
    mode = _string(call, "mode").casefold()
    if mode != expected:
        raise ValueError(f"mode must be '{expected}'")
    engine.mode = cast(OperationMode, mode)
    return {"mode": engine.mode}


def _set_conversation_mode(
    engine: Celune, call: ToolCall, context: AgentContext
) -> JSON:
    """Select conversation routing."""
    return _set_mode(engine, call, context, "converse")


def _set_agent_mode(engine: Celune, call: ToolCall, context: AgentContext) -> JSON:
    """Select agent routing."""
    return _set_mode(engine, call, context, "agent")


def _sleep(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Enter configured Celune sleep mode."""
    return {"sleeping": engine.enter_sleep_mode()}


def _wake(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Wake Celune through its existing model restoration path."""
    return {"awake": engine.wake_from_sleep()}


def _remember(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Persist one explicit character-scoped memory."""
    importance = _integer(call, "importance", 1)
    if not 1 <= importance <= 3:
        raise ValueError("importance must be between 1 and 3")
    record = _memory_store(engine).remember(
        _character(engine),
        _string(call, "content"),
        importance=importance,
        explicit=True,
    )
    if record is None:
        raise ValueError("memory content was empty after normalization")
    return record.to_json()


def _recall(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Retrieve semantically relevant character-scoped memories."""
    limit = _integer(call, "limit", 5)
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    records = _memory_store(engine).retrieve(
        _character(engine), _string(call, "request"), limit
    )
    return {"memories": [record.to_json() for record in records]}


def _forget(engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Forget one character-scoped memory by record ID."""
    record_id = _string(call, "record_id")
    removed = _memory_store(engine).forget(_character(engine), record_id)
    return {"forgotten": removed, "record_id": record_id}


def _clear_context(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Clear recent Persona context while retaining long-term memory."""
    engine._reset_persona_conversation()
    return {"cleared": True}


def _summarize(engine: Celune, _call: ToolCall, _context: AgentContext) -> JSON:
    """Read the current Persona context summary."""
    return {
        "summary": engine.persona_session_summary,
        "message_count": len(engine.persona_history),
    }


def _pause_task(engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Pause the current task through AgentRuntime."""
    task = _task(context)
    engine.agent_runtime.pause_task(task.task_id)
    return {"task": task.to_json()}


def _resume_task(engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Resume the current task through AgentRuntime."""
    task = _task(context)
    engine.agent_runtime.resume(task.session_id)
    return {"task": task.to_json()}


def _cancel_task(engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Cancel the current task through AgentRuntime."""
    task = _task(context)
    engine.agent_runtime.cancel_task(task.task_id)
    return {"task": task.to_json()}


def _query_task(_engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Read the current task."""
    return _task(context).to_json()


def _query_history(_engine: Celune, _call: ToolCall, context: AgentContext) -> JSON:
    """Read the current task's preserved request history."""
    task = _task(context)
    return {
        "task_id": task.task_id,
        "history": cast(JSONSerializable, list(task.request.history)),
    }


_MAX_LOCAL_READ_BYTES = 1_048_576
_MAX_LOCAL_ITEMS = 1_000


def _local_path(call: ToolCall, name: str) -> Path:
    """Validate and normalize one absolute local-management path."""
    raw = _string(call, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise LocalManagementError(
            "invalid_path",
            "local-management paths must be absolute",
        )
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise LocalManagementError("invalid_path", str(exc), path) from exc


def _local_limit(call: ToolCall, name: str, default: int, maximum: int) -> int:
    """Read one bounded integer option for a local-management operation."""
    value = _integer(call, name, default)
    if not 1 <= value <= maximum:
        raise ValueError(f"argument '{name}' must be between 1 and {maximum}")
    return value


def _local_missing(path: Path) -> LocalManagementError:
    """Build a consistent missing-target error."""
    return LocalManagementError("missing", "target does not exist", path)


def _local_access(path: Path, error: OSError) -> LocalManagementError:
    """Build a consistent access-denied error."""
    return LocalManagementError("access_denied", str(error), path)


def _local_list_directory(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """List one exact directory without recursive traversal."""
    path = _local_path(call, "path")
    limit = _local_limit(call, "limit", 100, _MAX_LOCAL_ITEMS)
    if not path.exists():
        raise _local_missing(path)
    if not path.is_dir():
        raise LocalManagementError("invalid_target", "target is not a directory", path)
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        raise _local_access(path, exc) from exc
    payload = [
        {
            "name": entry.name,
            "path": str(entry),
            "kind": "directory" if entry.is_dir() else "file",
        }
        for entry in entries[:limit]
    ]
    return {
        "result": "success",
        "path": str(path),
        "items": cast(JSONSerializable, payload),
        "truncated": len(entries) > limit,
    }


def _local_file_metadata(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Return metadata for one exact local filesystem target."""
    path = _local_path(call, "path")
    if not path.exists():
        raise _local_missing(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise _local_access(path, exc) from exc
    return {
        "result": "success",
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def _local_read_text(_engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Read a bounded UTF-8 text file from one exact path."""
    path = _local_path(call, "path")
    limit = _local_limit(
        call, "max_bytes", _MAX_LOCAL_READ_BYTES, _MAX_LOCAL_READ_BYTES
    )
    if not path.exists():
        raise _local_missing(path)
    if not path.is_file():
        raise LocalManagementError("invalid_target", "target is not a file", path)
    try:
        if path.stat().st_size > limit:
            raise LocalManagementError("too_large", "file exceeds max_bytes", path)
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _local_access(path, exc) from exc
    return {"result": "success", "path": str(path), "text": text}


def _local_write_text(_engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Write bounded UTF-8 text to one exact existing-parent path."""
    path = _local_path(call, "path")
    text = _text(call, "text")
    if len(text.encode("utf-8")) > _MAX_LOCAL_READ_BYTES:
        raise LocalManagementError(
            "too_large", "text exceeds the local size limit", path
        )
    if path.exists() and path.is_dir():
        raise LocalManagementError("invalid_target", "target is a directory", path)
    if not path.parent.exists():
        raise LocalManagementError(
            "missing", "parent directory does not exist", path.parent
        )
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise _local_access(path, exc) from exc
    return {"result": "success", "path": str(path), "bytes": len(text.encode("utf-8"))}


def _local_make_directory(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Create one exact directory path without implicit parent creation."""
    path = _local_path(call, "path")
    if path.exists():
        if path.is_dir():
            return {"result": "success", "path": str(path), "created": False}
        raise LocalManagementError(
            "invalid_target", "target already exists as a file", path
        )
    if not path.parent.exists():
        raise LocalManagementError(
            "missing", "parent directory does not exist", path.parent
        )
    try:
        path.mkdir()
    except OSError as exc:
        raise _local_access(path, exc) from exc
    return {"result": "success", "path": str(path), "created": True}


def _local_copy(_engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Copy one exact file to one exact non-existing destination."""
    source = _local_path(call, "source")
    destination = _local_path(call, "destination")
    if not source.exists():
        raise _local_missing(source)
    if not source.is_file():
        raise LocalManagementError("invalid_target", "source is not a file", source)
    if destination.exists():
        raise LocalManagementError(
            "invalid_target", "destination already exists", destination
        )
    if not destination.parent.exists():
        raise LocalManagementError(
            "missing", "destination parent does not exist", destination.parent
        )
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise _local_access(destination, exc) from exc
    return {"result": "success", "source": str(source), "destination": str(destination)}


def _local_move(_engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Move one exact file to one exact non-existing destination."""
    source = _local_path(call, "source")
    destination = _local_path(call, "destination")
    if not source.exists():
        raise _local_missing(source)
    if destination.exists():
        raise LocalManagementError(
            "invalid_target", "destination already exists", destination
        )
    if not destination.parent.exists():
        raise LocalManagementError(
            "missing", "destination parent does not exist", destination.parent
        )
    try:
        shutil.move(str(source), str(destination))
    except OSError as exc:
        raise _local_access(destination, exc) from exc
    return {"result": "success", "source": str(source), "destination": str(destination)}


def _local_delete(_engine: Celune, call: ToolCall, _context: AgentContext) -> JSON:
    """Delete one exact file or empty directory, never recursively by default."""
    path = _local_path(call, "path")
    recursive = call["arguments"].get("recursive", False)
    if not isinstance(recursive, bool):
        raise TypeError("argument 'recursive' must be a boolean")
    if not path.exists():
        raise _local_missing(path)
    try:
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink()
    except OSError as exc:
        raise _local_access(path, exc) from exc
    return {"result": "success", "path": str(path), "deleted": True}


def _process_info(pid: int) -> psutil.Process:
    """Resolve a process and preserve missing/access errors for the caller."""
    try:
        process = psutil.Process(pid)
        process.status()
        return process
    except psutil.NoSuchProcess as exc:
        raise LocalManagementError("missing", "process does not exist") from exc
    except psutil.AccessDenied as exc:
        raise LocalManagementError(
            "access_denied", "process information is unavailable"
        ) from exc


def _process_json(process: psutil.Process) -> JSON:
    """Return bounded process identity and state metadata."""
    try:
        name = process.name()
        executable = process.exe()
        status = process.status()
    except psutil.NoSuchProcess as exc:
        raise LocalManagementError(
            "missing", "process exited during inspection"
        ) from exc
    except psutil.AccessDenied as exc:
        raise LocalManagementError(
            "access_denied", "process information is unavailable"
        ) from exc
    return {
        "pid": process.pid,
        "name": name,
        "executable": executable,
        "status": status,
    }


def _local_list_processes(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """List bounded process identity records using psutil."""
    limit = _local_limit(call, "limit", 100, _MAX_LOCAL_ITEMS)
    records: list[JSONSerializable] = []
    for process in psutil.process_iter(["pid", "name", "exe", "status"]):
        try:
            records.append(_process_json(process))
        except LocalManagementError:
            continue
        if len(records) >= limit:
            break
    return {
        "result": "success",
        "processes": records,
        "truncated": len(records) >= limit,
    }


def _local_inspect_process(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Inspect one process by PID and optionally verify its identity."""
    pid = _integer(call, "pid", 0)
    if pid <= 0:
        raise ValueError("argument 'pid' must be positive")
    process = _process_info(pid)
    output = _process_json(process)
    expected_name = _optional_string(call, "expected_name")
    expected_executable = _optional_string(call, "expected_executable")
    if expected_name and str(output["name"]).casefold() != expected_name.casefold():
        raise LocalManagementError("identity_mismatch", "process name did not match")
    if expected_executable:
        actual = str(output["executable"])
        if Path(actual).resolve(strict=False) != Path(
            expected_executable
        ).expanduser().resolve(strict=False):
            raise LocalManagementError(
                "identity_mismatch", "process executable did not match"
            )
    return {"result": "success", **output}


def _argument_list(call: ToolCall, name: str) -> list[str]:
    """Read a structured list of process arguments without shell syntax."""
    value = call["arguments"].get(name, [])
    if not isinstance(value, list):
        raise TypeError(f"argument '{name}' must be an array of strings")
    arguments: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"argument '{name}' must be an array of strings")
        if item:
            arguments.append(item)
    return arguments


def _local_launch_process(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Launch one explicit executable with structured arguments and no shell."""
    executable = _string(call, "executable")
    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable).expanduser()
        if candidate.is_absolute() and candidate.exists():
            resolved = str(candidate.resolve(strict=True))
    if resolved is None:
        raise LocalManagementError("missing", "executable was not found")
    cwd_value = _optional_string(call, "cwd")
    cwd = Path(cwd_value).expanduser().resolve(strict=False) if cwd_value else None
    if cwd is not None and not cwd.is_dir():
        raise LocalManagementError("invalid_target", "cwd is not a directory", cwd)
    try:
        # Popen must remain alive so the caller can inspect or terminate it later.
        # pylint: disable=consider-using-with
        process = subprocess.Popen(
            [resolved, *_argument_list(call, "arguments")],
            cwd=str(cwd) if cwd is not None else None,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise LocalManagementError("access_denied", str(exc)) from exc
    return {"result": "success", "pid": process.pid, "executable": resolved}


def _local_terminate_process(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Terminate one process only after verifying its requested identity."""
    pid = _integer(call, "pid", 0)
    if pid <= 0:
        raise ValueError("argument 'pid' must be positive")
    process = _process_info(pid)
    expected_name = _string(call, "expected_name")
    details = _process_json(process)
    if str(details["name"]).casefold() != expected_name.casefold():
        raise LocalManagementError("identity_mismatch", "process name did not match")
    try:
        process.terminate()
        process.wait(timeout=3)
    except psutil.TimeoutExpired as exc:
        raise LocalManagementError(
            "timeout", "process did not terminate in time"
        ) from exc
    except psutil.Error as exc:
        raise LocalManagementError("access_denied", str(exc)) from exc
    return {"result": "success", "pid": pid, "terminated": True}


def _local_system_info(
    _engine: Celune, _call: ToolCall, _context: AgentContext
) -> JSON:
    """Read bounded local OS, resource, disk, and uptime diagnostics."""
    root = Path.cwd().anchor or str(Path.cwd())
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(root)
    return {
        "result": "success",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory": {"total": memory.total, "available": memory.available},
        "disk": {"path": root, "total": disk.total, "free": disk.free},
        "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "cuda": {"path": os.environ.get("CUDA_PATH")},
        "uptime_seconds": max(0.0, time.time() - psutil.boot_time()),
        "local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _local_current_working_directory(
    _engine: Celune, _call: ToolCall, _context: AgentContext
) -> JSON:
    """Return the exact resolved working directory of the tool process."""
    try:
        path = Path.cwd().resolve(strict=True)
    except OSError as exc:
        raise LocalManagementError("access_denied", str(exc)) from exc
    if not path.is_dir():
        raise LocalManagementError(
            "invalid_target", "working directory is not a directory", path
        )
    return {"result": "success", "path": str(path), "kind": "directory"}


def _local_discover_application(
    _engine: Celune, call: ToolCall, _context: AgentContext
) -> JSON:
    """Resolve one application executable without launching it."""
    name = _string(call, "name")
    executable = shutil.which(name)
    if executable is None:
        raise LocalManagementError("missing", "application executable was not found")
    return {"result": "success", "name": name, "executable": executable}


def _local_running_applications(
    engine: Celune, call: ToolCall, context: AgentContext
) -> JSON:
    """Expose running applications through the same bounded process view."""
    return _local_list_processes(engine, call, context)


def _local_launch_application(
    engine: Celune, call: ToolCall, context: AgentContext
) -> JSON:
    """Launch an explicitly named application through the process executor."""
    return _local_launch_process(engine, call, context)


def _local_close_application(
    engine: Celune, call: ToolCall, context: AgentContext
) -> JSON:
    """Close an application only after process-name verification."""
    return _local_terminate_process(engine, call, context)


def _arg(
    name: str,
    value_type: AgentToolValueType,
    description: str,
    *,
    required: bool = True,
    item_type: Optional[AgentToolValueType] = None,
) -> AgentToolArgumentSchema:
    """Create one typed argument declaration."""
    return AgentToolArgumentSchema(
        name,
        value_type,
        description,
        required,
        item_type,
    )


def _spec(
    tool_id: str,
    display_name: str,
    description: str,
    handler: OfflineToolHandler,
    *,
    arguments: tuple[AgentToolArgumentSchema, ...] = (),
    behavior: AgentToolBehavior = AgentToolBehavior.READ_ONLY,
    danger: AgentToolDangerLevel = AgentToolDangerLevel.LOW,
    available: bool = True,
    end_task_on_success: bool = False,
) -> OfflineToolSpec:
    """Create one catalog entry."""
    return OfflineToolSpec(
        tool_id,
        display_name,
        description,
        arguments,
        behavior,
        danger,
        handler,
        available,
        end_task_on_success,
    )


_SPECS = (
    _spec(
        "read_agent_status",
        "Read agent status",
        "Read the current Celune agent task status.",
        _legacy_status,
    ),
    _spec("query_status", "Query status", "Read Celune runtime status.", _status),
    _spec(
        "query_capabilities",
        "Query capabilities",
        "Read Celune capabilities.",
        _capabilities,
    ),
    _spec("query_models", "Query models", "Read loaded model identifiers.", _models),
    _spec("query_locks", "Query locks", "Read component lock ownership.", _locks),
    _spec(
        "query_audio_state",
        "Query audio state",
        "Read speech queue state.",
        _audio_state,
    ),
    _spec(
        "query_agent_task",
        "Query agent task",
        "Read the current agent task.",
        _agent_task,
    ),
    _spec(
        "run_health_check",
        "Run health check",
        "Check local runtime prerequisites.",
        _health,
    ),
    _spec(
        "speak",
        "Speak",
        "Speak text through Celune.",
        _speak,
        arguments=(_arg("text", AgentToolValueType.STRING, "Text to speak."),),
        behavior=AgentToolBehavior.MUTATING,
        end_task_on_success=True,
    ),
    _spec(
        "stop_speech",
        "Stop speech",
        "Stop active speech playback.",
        _stop_speech,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "pause_speech",
        "Pause speech",
        "Pause speech when supported.",
        _stop_speech,
        behavior=AgentToolBehavior.MUTATING,
        available=False,
    ),
    _spec(
        "resume_speech",
        "Resume speech",
        "Resume paused speech when supported.",
        _speak,
        arguments=(_arg("text", AgentToolValueType.STRING, "Text to resume."),),
        behavior=AgentToolBehavior.MUTATING,
        available=False,
    ),
    _spec(
        "set_voice",
        "Set voice",
        "Switch the active Celune voice.",
        _set_voice,
        arguments=(_arg("voice", AgentToolValueType.STRING, "Voice name."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "set_voice_prompt",
        "Set voice prompt",
        "Set or clear the active voice prompt.",
        _set_voice_prompt,
        arguments=(
            _arg("prompt", AgentToolValueType.STRING, "Prompt text.", required=False),
        ),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "set_playback_speed",
        "Set playback speed",
        "Set speech speed from 0.25 to 4.0.",
        _set_speed,
        arguments=(_arg("speed", AgentToolValueType.NUMBER, "Playback speed."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "set_reverb",
        "Set reverb",
        "Set reverb strength from 0 to 1.",
        _set_reverb,
        arguments=(_arg("strength", AgentToolValueType.NUMBER, "Reverb strength."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "clear_speech_queue",
        "Clear speech queue",
        "Clear pending speech and audio.",
        _clear_speech_queue,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "set_character",
        "Set character",
        "Load a CEVOICE character pack.",
        _set_character,
        arguments=(_arg("bundle", AgentToolValueType.STRING, "Pack name or path."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "query_character",
        "Query character",
        "Read character and voice-pack identity.",
        _character_query,
    ),
    _spec(
        "set_conversation_mode",
        "Set conversation mode",
        "Use conversation routing.",
        _set_conversation_mode,
        arguments=(_arg("mode", AgentToolValueType.STRING, "Must be converse."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "set_agent_mode",
        "Set agent mode",
        "Use agent routing.",
        _set_agent_mode,
        arguments=(_arg("mode", AgentToolValueType.STRING, "Must be agent."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "sleep",
        "Sleep",
        "Put Celune into configured sleep mode.",
        _sleep,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "wake",
        "Wake",
        "Wake Celune and restore models.",
        _wake,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "remember",
        "Remember",
        "Persist a character-scoped memory.",
        _remember,
        arguments=(
            _arg("content", AgentToolValueType.STRING, "Memory content."),
            _arg(
                "importance",
                AgentToolValueType.INTEGER,
                "Importance from 1 to 3.",
                required=False,
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "recall",
        "Recall",
        "Retrieve relevant character memories.",
        _recall,
        arguments=(
            _arg("request", AgentToolValueType.STRING, "Recall request."),
            _arg(
                "limit", AgentToolValueType.INTEGER, "Maximum records.", required=False
            ),
        ),
    ),
    _spec(
        "forget",
        "Forget",
        "Forget a character memory by ID.",
        _forget,
        arguments=(_arg("record_id", AgentToolValueType.STRING, "Memory record ID."),),
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "clear_recent_context",
        "Clear recent context",
        "Clear recent Persona context.",
        _clear_context,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "summarize_context",
        "Summarize context",
        "Read the current context summary.",
        _summarize,
    ),
    _spec(
        "pause_task",
        "Pause task",
        "Pause the current agent task.",
        _pause_task,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "resume_task",
        "Resume task",
        "Resume the current agent task.",
        _resume_task,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec(
        "cancel_task",
        "Cancel task",
        "Cancel the current agent task.",
        _cancel_task,
        behavior=AgentToolBehavior.MUTATING,
    ),
    _spec("query_task", "Query task", "Read the current agent task.", _query_task),
    _spec(
        "query_task_history",
        "Query task history",
        "Read preserved task request history.",
        _query_history,
    ),
)


_LOCAL_SPECS = (
    _spec(
        "local_current_working_directory",
        "Current working directory",
        "Read the exact working directory of the Celune tool process.",
        _local_current_working_directory,
    ),
    _spec(
        "local_list_directory",
        "List directory",
        "List entries in one exact local directory.",
        _local_list_directory,
        arguments=(
            _arg("path", AgentToolValueType.STRING, "Absolute directory path."),
            _arg(
                "limit", AgentToolValueType.INTEGER, "Maximum entries.", required=False
            ),
        ),
    ),
    _spec(
        "local_file_metadata",
        "File metadata",
        "Read metadata for one exact local path.",
        _local_file_metadata,
        arguments=(_arg("path", AgentToolValueType.STRING, "Absolute file path."),),
    ),
    _spec(
        "local_read_text",
        "Read text file",
        "Read a bounded UTF-8 text file from one exact local path.",
        _local_read_text,
        arguments=(
            _arg("path", AgentToolValueType.STRING, "Absolute file path."),
            _arg(
                "max_bytes",
                AgentToolValueType.INTEGER,
                "Maximum bytes.",
                required=False,
            ),
        ),
    ),
    _spec(
        "local_write_text",
        "Write text file",
        "Write bounded UTF-8 text to one exact local path.",
        _local_write_text,
        arguments=(
            _arg("path", AgentToolValueType.STRING, "Absolute file path."),
            _arg("text", AgentToolValueType.STRING, "Text to write."),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.MEDIUM,
    ),
    _spec(
        "local_make_directory",
        "Make directory",
        "Create one exact local directory without creating parents.",
        _local_make_directory,
        arguments=(
            _arg("path", AgentToolValueType.STRING, "Absolute directory path."),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.MEDIUM,
    ),
    _spec(
        "local_copy",
        "Copy file",
        "Copy one exact local file to a new destination.",
        _local_copy,
        arguments=(
            _arg("source", AgentToolValueType.STRING, "Absolute source path."),
            _arg(
                "destination", AgentToolValueType.STRING, "Absolute destination path."
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.MEDIUM,
    ),
    _spec(
        "local_move",
        "Move file",
        "Move one exact local file to a new destination.",
        _local_move,
        arguments=(
            _arg("source", AgentToolValueType.STRING, "Absolute source path."),
            _arg(
                "destination", AgentToolValueType.STRING, "Absolute destination path."
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
    _spec(
        "local_delete",
        "Delete local path",
        "Delete one exact local file or directory.",
        _local_delete,
        arguments=(
            _arg("path", AgentToolValueType.STRING, "Absolute target path."),
            _arg(
                "recursive",
                AgentToolValueType.BOOLEAN,
                "Allow recursive deletion.",
                required=False,
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
    _spec(
        "local_list_processes",
        "List processes",
        "List bounded local process identity and status records.",
        _local_list_processes,
        arguments=(
            _arg(
                "limit",
                AgentToolValueType.INTEGER,
                "Maximum processes.",
                required=False,
            ),
        ),
    ),
    _spec(
        "local_inspect_process",
        "Inspect process",
        "Inspect one process by PID and optional identity checks.",
        _local_inspect_process,
        arguments=(
            _arg("pid", AgentToolValueType.INTEGER, "Process ID."),
            _arg(
                "expected_name",
                AgentToolValueType.STRING,
                "Expected process name.",
                required=False,
            ),
            _arg(
                "expected_executable",
                AgentToolValueType.STRING,
                "Expected executable path.",
                required=False,
            ),
        ),
    ),
    _spec(
        "local_launch_process",
        "Launch process",
        "Launch one explicit executable with structured arguments and no shell.",
        _local_launch_process,
        arguments=(
            _arg(
                "executable",
                AgentToolValueType.STRING,
                "Executable name or absolute path.",
            ),
            _arg(
                "arguments",
                AgentToolValueType.ARRAY,
                "Structured process arguments.",
                required=False,
                item_type=AgentToolValueType.STRING,
            ),
            _arg(
                "cwd",
                AgentToolValueType.STRING,
                "Absolute working directory.",
                required=False,
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
    _spec(
        "local_terminate_process",
        "Terminate process",
        "Terminate a process only after verifying its name.",
        _local_terminate_process,
        arguments=(
            _arg("pid", AgentToolValueType.INTEGER, "Process ID."),
            _arg("expected_name", AgentToolValueType.STRING, "Expected process name."),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
    _spec(
        "local_system_info",
        "System diagnostics",
        "Read local OS, CPU, memory, GPU, CUDA, disk, uptime, and time diagnostics.",
        _local_system_info,
    ),
    _spec(
        "local_discover_application",
        "Discover application",
        "Resolve one application executable without launching it.",
        _local_discover_application,
        arguments=(
            _arg("name", AgentToolValueType.STRING, "Application executable name."),
        ),
    ),
    _spec(
        "local_running_applications",
        "Running applications",
        "List running applications through bounded process diagnostics.",
        _local_running_applications,
        arguments=(
            _arg(
                "limit",
                AgentToolValueType.INTEGER,
                "Maximum processes.",
                required=False,
            ),
        ),
    ),
    _spec(
        "local_launch_application",
        "Launch application",
        "Launch one explicit application without GUI automation or a shell.",
        _local_launch_application,
        arguments=(
            _arg(
                "executable",
                AgentToolValueType.STRING,
                "Executable name or absolute path.",
            ),
            _arg(
                "arguments",
                AgentToolValueType.ARRAY,
                "Structured application arguments.",
                required=False,
                item_type=AgentToolValueType.STRING,
            ),
            _arg(
                "cwd",
                AgentToolValueType.STRING,
                "Absolute working directory.",
                required=False,
            ),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
    _spec(
        "local_close_application",
        "Close application",
        "Close one application only after process-name verification.",
        _local_close_application,
        arguments=(
            _arg("pid", AgentToolValueType.INTEGER, "Process ID."),
            _arg("expected_name", AgentToolValueType.STRING, "Expected process name."),
        ),
        behavior=AgentToolBehavior.MUTATING,
        danger=AgentToolDangerLevel.HIGH,
    ),
)


_LOCAL_TEST_SPECS = tuple(
    spec for spec in _LOCAL_SPECS if spec.tool_id == "local_current_working_directory"
)


def _schemas_for_specs(
    specs: tuple[OfflineToolSpec, ...],
) -> dict[str, AgentToolSchema]:
    """Build typed schemas for one internal registry."""
    return {
        spec.tool_id: AgentToolSchema(
            tool_id=spec.tool_id,
            display_name=spec.display_name,
            description=spec.description,
            arguments=spec.arguments,
            behavior=spec.behavior,
            danger=spec.danger,
            approval_required=spec.behavior == AgentToolBehavior.MUTATING,
            available=spec.available,
        )
        for spec in specs
    }


def local_management_tools(engine: Optional[Celune] = None) -> tuple[AgentTool, ...]:
    """Return the explicitly enabled local-management registry tools."""
    return tuple(OfflineAgentTool(engine, spec) for spec in _LOCAL_SPECS)


def local_management_tool_schemas() -> Mapping[str, AgentToolSchema]:
    """Return schemas for the local-management registry."""
    return _schemas_for_specs(_LOCAL_SPECS)


def production_agent_tools(
    engine: Optional[Celune] = None,
    *,
    include_local_management: bool = False,
) -> tuple[AgentTool, ...]:
    """Return the allowlisted offline tools exposed to production agent mode."""
    specs = _SPECS + (_LOCAL_SPECS if include_local_management else ())
    return (AgentStatusTool(),) + tuple(
        OfflineAgentTool(engine, spec)
        for spec in specs
        if spec.tool_id != AgentStatusTool.name
    )


def production_agent_tool_schemas(
    *,
    include_local_management: bool = False,
) -> Mapping[str, AgentToolSchema]:
    """Return typed schemas for every production offline tool."""
    specs = _SPECS + (_LOCAL_SPECS if include_local_management else ())
    return _schemas_for_specs(specs)
