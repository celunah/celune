# SPDX-License-Identifier: Apache-2.0
"""Safe, isolated diagnostics for the production agent feature catalog."""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, cast
from pathlib import Path
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import Mock, patch
from collections.abc import Mapping, Callable

from ..i18n import string
from .tools import production_agent_tools, production_agent_tool_schemas
from .runtime import AgentRuntime, DefaultAgentPermissionPolicy
from ..exceptions import NeedleSelectionError
from ..typing.agent import (
    ToolCall,
    AgentTask,
    AgentTool,
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentPlanner,
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentTaskConfig,
    AgentToolSchema,
    AgentAbortReason,
    AgentInterruption,
    AgentToolBehavior,
    AgentToolExecutor,
    AgentToolSelector,
    AgentFailureReason,
    AgentChoiceResponse,
    ToolExecutionResult,
    AgentTerminalOutcome,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentInterruptionKind,
    AgentPermissionPolicy,
    AgentPermissionReason,
    AgentToolResultHandler,
    AgentCancellationReason,
    AgentPermissionDecision,
)
from ..typing.common import JSON
from ..extensions.events import EventDispatcher

if TYPE_CHECKING:
    from ..celune import Celune


_AGENT_TOOL_ARGUMENTS: dict[str, JSON] = {
    "read_agent_status": {},
    "query_status": {},
    "query_capabilities": {},
    "query_models": {},
    "query_locks": {},
    "query_audio_state": {},
    "query_agent_task": {},
    "run_health_check": {},
    "speak": {"text": "Agent diagnostic speech"},
    "stop_speech": {},
    "pause_speech": {},
    "resume_speech": {"text": "Agent diagnostic speech"},
    "set_voice": {"voice": "diagnostic"},
    "set_voice_prompt": {"prompt": "Agent diagnostic"},
    "set_playback_speed": {"speed": 1.0},
    "set_reverb": {"strength": 0.0},
    "clear_speech_queue": {},
    "set_character": {"bundle": "agent-diagnostic"},
    "query_character": {},
    "set_conversation_mode": {"mode": "converse"},
    "set_agent_mode": {"mode": "agent"},
    "sleep": {},
    "wake": {},
    "remember": {"content": "Temporary agent diagnostic memory"},
    "recall": {"request": "Temporary agent diagnostic memory"},
    "forget": {"record_id": "agent-diagnostic-memory"},
    "clear_recent_context": {},
    "summarize_context": {},
    "pause_task": {},
    "resume_task": {},
    "cancel_task": {},
    "query_task": {},
    "query_task_history": {},
    "local_current_working_directory": {},
    "local_list_directory": {},
    "local_file_metadata": {},
    "local_read_text": {},
    "local_write_text": {},
    "local_make_directory": {},
    "local_copy": {},
    "local_move": {},
    "local_delete": {},
    "local_list_processes": {"limit": 1},
    "local_inspect_process": {"pid": 1},
    "local_launch_process": {"executable": "python"},
    "local_terminate_process": {"pid": 1, "expected_name": "diagnostic"},
    "local_system_info": {},
    "local_discover_application": {"name": "python"},
    "local_running_applications": {"limit": 1},
    "local_launch_application": {"executable": "python"},
    "local_close_application": {"pid": 1, "expected_name": "diagnostic"},
}

_LOCAL_TOOL_NAMES = {
    name for name in _AGENT_TOOL_ARGUMENTS if name.startswith("local_")
}


class _DiagnosticMemoryRecord:
    """Small in-memory record used to keep diagnostics away from user memory."""

    def to_json(self) -> JSON:
        """Return the fixed temporary record in the normal JSON shape."""
        return {
            "record_id": "agent-diagnostic-memory",
            "content": "Temporary agent diagnostic memory",
            "importance": 1,
        }


class _DiagnosticMemoryStore:
    """In-memory substitute for persistent Persona memory operations."""

    def remember(
        self,
        _character: str,
        _content: str,
        *,
        importance: int,
        explicit: bool,
    ) -> Optional[_DiagnosticMemoryRecord]:
        """Return a record without touching the configured memory store."""
        if importance != 1 or not explicit:
            return None
        return _DiagnosticMemoryRecord()

    def retrieve(
        self,
        _character: str,
        _request: str,
        _limit: int,
    ) -> list[_DiagnosticMemoryRecord]:
        """Return the one temporary record expected by the diagnostic."""
        return [_DiagnosticMemoryRecord()]

    def forget(self, _character: str, _record_id: str) -> bool:
        """Accept removal of the temporary record."""
        return True


class _DiagnosticProcess:
    """Pretend process that records requests without starting or stopping one."""

    def __init__(self) -> None:
        self.pid = 1
        self.terminated = False

    @staticmethod
    def name() -> str:
        """Return the expected temporary process name."""
        return "diagnostic"

    @staticmethod
    def exe() -> str:
        """Return the active Python executable as a stable identity."""
        return sys.executable

    @staticmethod
    def status() -> str:
        """Return a live-looking state for the mocked process."""
        return "running"

    def terminate(self) -> None:
        """Record a no-op termination request."""
        self.terminated = True

    @staticmethod
    def wait(timeout: float) -> None:
        """Accept a no-op wait request."""
        del timeout


@dataclass
class _ExternalEffects:
    """Handles that prove external process operations stayed mocked."""

    launcher: Optional[Mock] = None
    process: Optional[_DiagnosticProcess] = None


def run_agent_feature_checks(engine: Celune) -> list[JSON]:
    """Run isolated checks for the active production agent catalog.

    Args:
        engine: Loaded Celune core used by the agent test mode.

    Returns:
        list[JSON]: Ordered per-check status records.
    """
    include_local_management = _local_management_enabled(engine)
    tools = production_agent_tools(
        engine,
        include_local_management=include_local_management,
    )
    schemas = production_agent_tool_schemas(
        include_local_management=include_local_management,
    )
    checks = [
        _run_check(
            "catalog.coverage",
            lambda: _check_catalog(tools, schemas, include_local_management),
        )
    ]
    checks.extend(
        _run_runtime_checks(
            engine,
            tools,
            schemas,
            include_local_management=include_local_management,
        )
    )
    checks.extend(_run_tool_checks(engine, tools, schemas))
    if not include_local_management:
        checks.append(
            _record(
                "local_management",
                "skipped",
                string("test.agent_local_management_disabled"),
            )
        )
    return checks


def _run_check(name: str, check: Callable[[], str]) -> JSON:
    """Capture one isolated diagnostic outcome without stopping later checks."""
    try:
        return _record(name, "passed", check())
    except Exception as exc:
        return _record(name, "failed", str(exc) or type(exc).__name__)


def _record(name: str, status: str, detail: str) -> JSON:
    """Build one JSON-compatible diagnostic result."""
    return {"name": name, "status": status, "detail": detail}


def _check_catalog(
    tools: tuple[AgentTool, ...],
    schemas: Mapping[str, AgentToolSchema],
    include_local_management: bool,
) -> str:
    """Require a diagnostic invocation and typed schema for every tool."""
    tool_names = {tool.name for tool in tools}
    schema_names = set(schemas)
    covered_names = set(_AGENT_TOOL_ARGUMENTS)
    if tool_names != schema_names:
        raise ValueError(
            f"tool/schema mismatch: tools={sorted(tool_names)} schemas={sorted(schema_names)}"
        )
    missing = tool_names - covered_names
    extra = covered_names - tool_names
    if not include_local_management:
        extra -= _LOCAL_TOOL_NAMES
    if missing or extra:
        raise ValueError(
            f"tool diagnostic mapping mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for name, schema in schemas.items():
        if name != schema.tool_id:
            raise ValueError(f"schema key does not match tool ID: {name}")
        if (
            schema.behavior == AgentToolBehavior.MUTATING
            and not schema.approval_required
        ):
            raise ValueError(f"mutating tool does not require approval: {name}")
    return string("test.agent_catalog_ok", count=len(tool_names))


def _run_runtime_checks(
    engine: Celune,
    tools: tuple[AgentTool, ...],
    schemas: Mapping[str, AgentToolSchema],
    *,
    include_local_management: bool,
) -> list[JSON]:
    """Exercise independent validation, permission, lifecycle, and limit paths."""
    return [
        _run_check(
            "runtime.validation",
            lambda: _check_validation(engine, tools, schemas),
        ),
        _run_check(
            "runtime.permissions",
            lambda: _check_permission(engine, schemas, "read"),
        ),
        _run_check(
            "runtime.permission_denied",
            lambda: _check_permission(engine, schemas, "denied"),
        ),
        _run_check(
            "runtime.permission_approval_unavailable",
            lambda: _check_permission(engine, schemas, "approval_unavailable"),
        ),
        _run_check(
            "runtime.process_launch_gate",
            lambda: _check_mutation_gate(
                engine,
                schemas,
                "local_launch_process",
            ),
        )
        if include_local_management
        else _record(
            "runtime.process_launch_gate",
            "skipped",
            string("test.agent_local_management_disabled"),
        ),
        _run_check(
            "runtime.process_terminate_gate",
            lambda: _check_mutation_gate(
                engine,
                schemas,
                "local_terminate_process",
            ),
        )
        if include_local_management
        else _record(
            "runtime.process_terminate_gate",
            "skipped",
            string("test.agent_local_management_disabled"),
        ),
        _run_check(
            "runtime.application_launch_gate",
            lambda: _check_mutation_gate(
                engine,
                schemas,
                "local_launch_application",
            ),
        )
        if include_local_management
        else _record(
            "runtime.application_launch_gate",
            "skipped",
            string("test.agent_local_management_disabled"),
        ),
        _run_check(
            "runtime.application_close_gate",
            lambda: _check_mutation_gate(
                engine,
                schemas,
                "local_close_application",
            ),
        )
        if include_local_management
        else _record(
            "runtime.application_close_gate",
            "skipped",
            string("test.agent_local_management_disabled"),
        ),
        _run_check(
            "runtime.lifecycle.pause_resume",
            lambda: _check_lifecycle_pause_resume(engine),
        ),
        _run_check(
            "runtime.lifecycle.interruption",
            lambda: _check_lifecycle_interruption(engine),
        ),
        _run_check(
            "runtime.lifecycle.steering",
            lambda: _check_lifecycle_steering(engine),
        ),
        _run_check(
            "runtime.lifecycle.cancellation",
            lambda: _check_lifecycle_cancellation(engine),
        ),
        _run_check(
            "runtime.lifecycle.events",
            lambda: _check_lifecycle_events(engine),
        ),
        _run_check(
            "runtime.approval.approved",
            lambda: _check_approval(engine, schemas, AgentApprovalDecision.APPROVED),
        ),
        _run_check(
            "runtime.approval.denied",
            lambda: _check_approval(engine, schemas, AgentApprovalDecision.DENIED),
        ),
        _run_check("runtime.choice", lambda: _check_tool_choice(engine, schemas)),
        _run_check("runtime.context.compaction", lambda: _check_compaction(engine)),
        _run_check("runtime.context.limit", lambda: _check_context_limit(engine)),
        _run_check("runtime.tokens.limit", lambda: _check_token_limit(engine)),
        *[
            _run_check(
                f"runtime.outcome.{state.value}",
                lambda state=state: _check_outcome(engine, state),
            )
            for state in (
                AgentTaskState.COMPLETED,
                AgentTaskState.FAILED,
                AgentTaskState.ABORTED,
                AgentTaskState.CANCELLED,
            )
        ],
    ]


def _dispatcher(engine: Celune) -> EventDispatcher:
    """Create an event dispatcher detached from the user's extension listeners."""
    del engine
    return EventDispatcher(
        log_warning=lambda _message, _severity: None,
        log_level="debug",
        log_debug=lambda _message: None,
    )


def _runtime(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
    *,
    tools: tuple[AgentTool, ...] = (),
    planner: Optional[AgentPlanner] = None,
    selector: Optional[AgentToolSelector] = None,
    executor: Optional[AgentToolExecutor] = None,
    result_handler: Optional[AgentToolResultHandler] = None,
    permission_policy: Optional[AgentPermissionPolicy] = None,
) -> AgentRuntime:
    """Create a task runtime with an isolated event dispatcher."""
    return AgentRuntime(
        event_dispatcher=_dispatcher(engine),
        celune=engine,
        tools=tools,
        tool_schemas=schemas,
        planner=planner,
        tool_selector=selector,
        tool_executor=executor,
        tool_result_handler=result_handler,
        permission_policy=permission_policy,
    )


def _working_task(runtime: AgentRuntime, name: str) -> AgentTask:
    """Create one working task with an isolated session identifier."""
    request = AgentRequest(
        "Run the isolated agent diagnostic.",
        session=AgentSession(session_id=f"agent-diagnostic-{name}"),
    )
    task = runtime.create_task(request)
    runtime.start_task(task.task_id)
    runtime.classify_task(task.task_id)
    return task


def _check_validation(
    engine: Celune,
    tools: tuple[AgentTool, ...],
    schemas: Mapping[str, AgentToolSchema],
) -> str:
    """Verify the production selector rejects schema-invalid model arguments."""
    from .needle.impl import NeedleHandler, NeedleToolSelector

    runtime = _runtime(engine, schemas)
    task = _working_task(runtime, "validation")
    handler = cast(
        NeedleHandler,
        SimpleNamespace(
            catalog_for_tools=NeedleHandler.catalog_for_tools,
            select_tools=lambda *_args, **_kwargs: [
                {"name": "set_voice", "arguments": {"voice": 1}}
            ],
        ),
    )
    selector = NeedleToolSelector(handler, tools, schemas=schemas)
    output: AgentOutput = {
        "tool_call": None,
        "response": "Set the voice to the invalid diagnostic value.",
        "end": False,
        "paused": False,
    }
    try:
        selector(runtime.get_context(task.task_id), output)
    except NeedleSelectionError:
        return string("test.agent_validation_ok")
    raise RuntimeError("schema-invalid arguments were accepted")


def _check_permission(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
    case: str,
) -> str:
    """Verify one allow, deny-list, or unavailable-approval decision."""
    if "query_status" not in schemas:
        raise RuntimeError("permission diagnostic schemas are unavailable")
    read_call: ToolCall = {
        "id": "permission-read",
        "name": "query_status",
        "arguments": {},
    }
    if case == "read":
        task, executed = _run_permission_case(
            engine,
            schemas,
            "permission-read",
            read_call,
        )
        if (
            task.state != AgentTaskState.COMPLETED
            or task.permission_decision is None
            or task.permission_decision.decision != AgentPermissionDecision.ALLOW
            or executed != ["query_status"]
        ):
            raise RuntimeError("safe read-only permission was not allowed")
    elif case == "denied":
        task, executed = _run_permission_case(
            engine,
            schemas,
            "permission-denied",
            read_call,
            DefaultAgentPermissionPolicy(disallowed_tool_ids=("query_status",)),
        )
        if (
            task.state != AgentTaskState.FAILED
            or task.permission_decision is None
            or task.permission_decision.reason != AgentPermissionReason.TOOL_DISALLOWED
            or executed
        ):
            raise RuntimeError("deny-listed tool crossed the permission boundary")
    elif case == "approval_unavailable":
        if "set_voice" not in schemas:
            raise RuntimeError("mutating permission schema is unavailable")
        mutation_call: ToolCall = {
            "id": "permission-mutation",
            "name": "set_voice",
            "arguments": {"voice": "diagnostic"},
        }
        task, executed = _run_permission_case(
            engine,
            schemas,
            "permission-unavailable",
            mutation_call,
            DefaultAgentPermissionPolicy(approval_available=False),
        )
        if (
            task.state != AgentTaskState.FAILED
            or task.permission_decision is None
            or task.permission_decision.reason
            != AgentPermissionReason.APPROVAL_UNAVAILABLE
            or executed
        ):
            raise RuntimeError("mutating tool was not denied without approval")
    else:
        raise ValueError(f"unknown permission diagnostic case: {case}")
    return string("test.agent_permission_ok")


def _run_permission_case(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
    name: str,
    call: ToolCall,
    permission_policy: Optional[AgentPermissionPolicy] = None,
) -> tuple[AgentTask, list[str]]:
    """Run one tool through runtime authorization with an inert executor."""
    executed: list[str] = []

    def execute(_context: AgentContext, selected: ToolCall) -> ToolResult:
        executed.append(selected["name"])
        return {
            "tool_call_id": selected["id"],
            "output": {"ok": True},
            "error": None,
        }

    runtime = _runtime(
        engine,
        schemas,
        planner=lambda _context: {
            "tool_call": call,
            "response": None,
            "end": False,
            "paused": False,
        },
        selector=lambda _context, output: output["tool_call"],
        executor=execute,
        result_handler=lambda _context, _result: {
            "tool_call": None,
            "response": "Diagnostic complete.",
            "end": True,
            "paused": False,
        },
        permission_policy=permission_policy,
    )
    task = _working_task(runtime, name)
    runtime.run(task.request)
    return task, executed


def _check_mutation_gate(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
    name: str,
) -> str:
    """Ensure process or application mutations fail closed without approval."""
    schema = schemas.get(name)
    if (
        schema is None
        or schema.behavior != AgentToolBehavior.MUTATING
        or schema.danger != AgentToolDangerLevel.HIGH
        or not schema.approval_required
    ):
        raise RuntimeError(f"{name} is not protected by the expected high-risk gate")
    call: ToolCall = {
        "id": f"mutation-gate-{name}",
        "name": name,
        "arguments": _AGENT_TOOL_ARGUMENTS[name],
    }
    task, executed = _run_permission_case(
        engine,
        schemas,
        f"mutation-gate-{name}",
        call,
        DefaultAgentPermissionPolicy(approval_available=False),
    )
    if (
        task.state != AgentTaskState.FAILED
        or task.permission_decision is None
        or task.permission_decision.reason != AgentPermissionReason.APPROVAL_UNAVAILABLE
        or executed
    ):
        raise RuntimeError(f"{name} crossed its approval gate")
    return string("test.agent_mutation_gate_ok")


def _lifecycle_task(engine: Celune, name: str) -> tuple[AgentRuntime, AgentTask]:
    """Create one working lifecycle fixture with a detached event bus."""
    runtime = AgentRuntime(event_dispatcher=_dispatcher(engine), celune=engine)
    return runtime, _working_task(runtime, name)


def _check_lifecycle_pause_resume(engine: Celune) -> str:
    """Check pause and resume preserve the active task."""
    runtime, task = _lifecycle_task(engine, "pause-resume")
    runtime.pause_task(task.task_id)
    if task.state != AgentTaskState.PAUSED:
        raise RuntimeError("pause did not suspend the task")
    runtime.resume(task.session_id)
    if task.state != AgentTaskState.WORKING:
        raise RuntimeError("resume did not restore the task")
    return string("test.agent_lifecycle_ok")


def _check_lifecycle_interruption(engine: Celune) -> str:
    """Check an interruption can be resumed at the prior lifecycle boundary."""
    runtime, task = _lifecycle_task(engine, "interruption")
    interruption = AgentInterruption(AgentInterruptionKind.USER_INTERRUPT)
    runtime.interrupt_task(task.task_id, interruption)
    if task.state != AgentTaskState.INTERRUPTED or task.interruption != interruption:
        raise RuntimeError("user interruption was not retained")
    runtime.resume(task.session_id)
    if task.state != AgentTaskState.WORKING:
        raise RuntimeError("interrupted task did not resume")
    return string("test.agent_lifecycle_ok")


def _check_lifecycle_steering(engine: Celune) -> str:
    """Check steering updates the request and returns the task to planning."""
    runtime, task = _lifecycle_task(engine, "steering")
    runtime.steer_task(
        task.task_id,
        AgentInterruption(
            AgentInterruptionKind.USER_STEERING,
            "Continue the isolated diagnostic.",
        ),
    )
    if (
        task.state != AgentTaskState.PLANNING
        or task.request.request != "Continue the isolated diagnostic."
    ):
        raise RuntimeError("steering did not update the active task")
    return string("test.agent_lifecycle_ok")


def _check_lifecycle_cancellation(engine: Celune) -> str:
    """Check cancellation reaches a typed cancelled terminal state."""
    runtime, task = _lifecycle_task(engine, "cancellation")
    runtime.cancel_task(task.task_id, AgentCancellationReason.USER_REQUEST)
    if (
        task.state != AgentTaskState.CANCELLED
        or task.cancellation_reason != AgentCancellationReason.USER_REQUEST
    ):
        raise RuntimeError("task cancellation did not preserve its reason")
    return string("test.agent_lifecycle_ok")


def _check_lifecycle_events(engine: Celune) -> str:
    """Check state-change events report pause, interruption, and cancellation."""
    events: list[str] = []
    dispatcher = _dispatcher(engine)
    dispatcher.subscribe(
        "agent_task_state_changed",
        lambda event: events.append(event.new_state),
        owner_name="agent-diagnostic",
    )
    runtime = AgentRuntime(event_dispatcher=dispatcher, celune=engine)
    task = _working_task(runtime, "lifecycle-events")
    runtime.pause_task(task.task_id)
    runtime.resume(task.session_id)
    runtime.interrupt_task(
        task.task_id,
        AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
    )
    runtime.resume(task.session_id)
    runtime.cancel_task(task.task_id)
    if not {"paused", "interrupted", "cancelled"}.issubset(set(events)):
        raise RuntimeError("task lifecycle events were not delivered")
    return string("test.agent_lifecycle_ok")


def _check_approval(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
    decision: AgentApprovalDecision,
) -> str:
    """Exercise one approved or denied mutation against an inert executor."""
    schema = schemas.get("set_voice")
    if schema is None or schema.behavior != AgentToolBehavior.MUTATING:
        raise RuntimeError("set_voice mutation schema is unavailable")
    executions: list[str] = []

    def planner(_context) -> AgentOutput:
        return {
            "tool_call": {
                "id": "agent-diagnostic-call",
                "name": "set_voice",
                "arguments": {"voice": "diagnostic"},
            },
            "response": None,
            "end": False,
            "paused": False,
        }

    def executor(_context: AgentContext, call: ToolCall) -> ToolResult:
        executions.append(call["name"])
        return {
            "tool_call_id": call["id"],
            "output": {"ok": True},
            "error": None,
        }

    runtime = _runtime(
        engine,
        schemas,
        planner=planner,
        selector=lambda _context, output: output["tool_call"],
        executor=executor,
        result_handler=lambda _context, _result: {
            "tool_call": None,
            "response": "Diagnostic complete.",
            "end": True,
            "paused": False,
        },
    )
    task = _working_task(runtime, f"approval-{decision.value}")
    paused = runtime.run(task.request)
    approval = runtime.get_pending_approval(task.task_id)
    if not paused["paused"] or approval is None:
        raise RuntimeError("mutating call did not request approval")
    runtime.respond_to_approval(
        task.task_id,
        AgentApprovalResponse(approval.request_id, decision),
    )
    if decision == AgentApprovalDecision.APPROVED:
        runtime.run(task.request)
        if task.state != AgentTaskState.COMPLETED or executions != ["set_voice"]:
            raise RuntimeError("approved call did not execute exactly once")
        return string("test.agent_approval_approved")
    if task.state != AgentTaskState.FAILED or executions:
        raise RuntimeError("denied call crossed the tool executor boundary")
    return string("test.agent_approval_denied")


def _check_tool_choice(
    engine: Celune,
    schemas: Mapping[str, AgentToolSchema],
) -> str:
    """Verify a multi-call selection pauses and resumes only the chosen call."""
    names = ("query_status", "query_models")
    if any(name not in schemas for name in names):
        raise RuntimeError("read-only choice candidates are unavailable")
    calls = [
        {"id": f"choice-{index}", "name": name, "arguments": {}}
        for index, name in enumerate(names, start=1)
    ]
    executed: list[str] = []
    selector = cast(AgentToolSelector, lambda _context, _output: calls)
    runtime = _runtime(
        engine,
        schemas,
        planner=lambda _context: {
            "tool_call": calls[0],
            "response": None,
            "end": False,
            "paused": False,
        },
        selector=selector,
        executor=lambda _context, call: (
            executed.append(call["name"])
            or {
                "tool_call_id": call["id"],
                "output": {"ok": True},
                "error": None,
            }
        ),
        result_handler=lambda _context, _result: {
            "tool_call": None,
            "response": "Diagnostic complete.",
            "end": True,
            "paused": False,
        },
    )
    task = _working_task(runtime, "choice")
    result = runtime.run(task.request)
    request = runtime.get_pending_choice(task.task_id)
    if not result["paused"] or request is None or task.iterations:
        raise RuntimeError("multiple calls did not pause for a choice")
    runtime.respond_to_choice(
        task.task_id,
        AgentChoiceResponse(request.request_id, choice_id="tool-2"),
    )
    runtime.run(task.request)
    if task.state != AgentTaskState.COMPLETED or executed != ["query_models"]:
        raise RuntimeError("choice did not execute exactly the selected tool")
    return string("test.agent_choice_ok")


def _check_compaction(engine: Celune) -> str:
    """Verify context pressure calls the production compaction boundary."""
    compacted: list[str] = []

    def compact(context):
        compacted.append(context.task.task_id)
        context.task.update_context_tokens(0)
        return context

    runtime = AgentRuntime(
        celune=engine,
        compactor=compact,
        planner=lambda _context: {
            "tool_call": None,
            "response": "Done.",
            "end": True,
            "paused": False,
        },
    )
    request = AgentRequest(
        "Check context compaction.",
        session=AgentSession(session_id="agent-diagnostic-limits"),
    )
    task = runtime.create_task(
        request,
        AgentTaskConfig(context_size=4, compact_at=100),
    )
    task.update_context_tokens(4)
    runtime.run(request)
    if task.state != AgentTaskState.COMPLETED or compacted != [task.task_id]:
        raise RuntimeError("context compaction did not run at its threshold")
    return string("test.agent_context_compaction_ok")


def _check_token_limit(engine: Celune) -> str:
    """Verify generated-token overflow aborts one bounded task."""
    capped = AgentRuntime(
        celune=engine,
        planner=lambda _context: {
            "tool_call": None,
            "response": "too many tokens",
            "end": True,
            "paused": False,
        },
        token_counter=lambda _text: 3,
    )
    capped_request = AgentRequest(
        "Check token limits.",
        session=AgentSession(session_id="agent-diagnostic-token-limit"),
    )
    capped_task = capped.create_task(
        capped_request,
        AgentTaskConfig(max_tokens=2),
    )
    capped.run(capped_request)
    if (
        capped_task.state != AgentTaskState.ABORTED
        or capped_task.abort_reason != AgentAbortReason.MAX_TOKENS
    ):
        raise RuntimeError("generated-token limit did not abort the task")
    return string("test.agent_token_limit_ok")


def _check_context_limit(engine: Celune) -> str:
    """Verify an uncompacted context larger than its budget aborts the task."""
    context_limited = AgentRuntime(celune=engine)
    limited_request = AgentRequest(
        "Check context limits.",
        session=AgentSession(session_id="agent-diagnostic-context-limit"),
    )
    limited_task = context_limited.create_task(
        limited_request,
        AgentTaskConfig(context_size=2),
    )
    context_limited.start_task(limited_task.task_id)
    context_limited.classify_task(limited_task.task_id)
    limited_task.update_context_tokens(3)
    context_limited.run(limited_request)
    if (
        limited_task.state != AgentTaskState.ABORTED
        or limited_task.abort_reason != AgentAbortReason.CONTEXT_LIMIT
    ):
        raise RuntimeError("context-size limit did not abort the task")
    return string("test.agent_context_limit_ok")


def _check_outcome(engine: Celune, expected: AgentTaskState) -> str:
    """Verify one typed terminal output and its matching finish event."""
    events: list[AgentTaskState] = []
    dispatcher = _dispatcher(engine)
    dispatcher.subscribe(
        "agent_task_finished",
        lambda event: events.append(event.state),
        owner_name="agent-diagnostic",
    )
    runtime = AgentRuntime(
        event_dispatcher=dispatcher,
        celune=engine,
        planner=lambda _context: {
            "tool_call": None,
            "response": "Diagnostic complete.",
            "end": True,
            "paused": False,
        },
    )

    task = _working_task(runtime, f"outcome-{expected.value}")
    if expected == AgentTaskState.COMPLETED:
        runtime.run(task.request)
    elif expected == AgentTaskState.FAILED:
        runtime.fail_task(task.task_id, AgentFailureReason.MODEL_ERROR)
    elif expected == AgentTaskState.ABORTED:
        runtime.abort_task(task.task_id, AgentAbortReason.STUCK_TASK)
    elif expected == AgentTaskState.CANCELLED:
        runtime.cancel_task(
            task.task_id,
            AgentCancellationReason.USER_REQUEST,
        )
    else:
        raise ValueError(f"not a supported terminal state: {expected.value}")
    outcome = runtime.run(task.request).get("terminal")
    if not isinstance(outcome, AgentTerminalOutcome) or outcome.state != expected:
        raise RuntimeError(f"missing typed {expected.value} terminal output")
    if events != [expected]:
        raise RuntimeError(
            f"{expected.value} terminal event was not emitted exactly once"
        )
    return string("test.agent_terminal_ok", state=expected.value)


def _run_tool_checks(
    engine: Celune,
    tools: tuple[AgentTool, ...],
    schemas: Mapping[str, AgentToolSchema],
) -> list[JSON]:
    """Exercise every mapped handler with disposable state and safe adapters."""
    checks: list[JSON] = []
    original_runtime = engine.agent_runtime
    original_tools = engine._agent_tools
    original_schemas = engine._agent_tool_schemas
    original_mode = engine.mode
    original_speed = engine.speed
    original_reverb = engine.reverb.strength
    original_voice_prompt = engine.voice_prompt
    old_needle = engine._agent_needle_selector
    runtime = _runtime(engine, schemas)
    memory = _DiagnosticMemoryStore()

    try:
        engine.agent_runtime = runtime
        engine._agent_tools = tools
        engine._agent_tool_schemas = schemas
        engine.mode = "agent"
        with tempfile.TemporaryDirectory(prefix="celune-agent-test-") as directory:
            root = Path(directory).resolve()
            for tool in tools:
                name = tool.name
                try:
                    schema = schemas.get(name)
                    if schema is None:
                        raise RuntimeError("tool has no production schema")
                    tool_arguments = _AGENT_TOOL_ARGUMENTS.get(name)
                    if tool_arguments is None:
                        raise RuntimeError("tool has no diagnostic argument mapping")
                    if (
                        name == "set_voice_prompt"
                        and not engine.voice_prompt_supported()
                    ):
                        checks.append(
                            _record(
                                name,
                                "skipped",
                                string("test.agent_voice_prompt_skipped"),
                            )
                        )
                        continue

                    tool_root = root / name
                    tool_root.mkdir()
                    (tool_root / "source.txt").write_text(
                        "Celune agent diagnostic",
                        encoding="utf-8",
                    )
                    if name == "local_move":
                        (tool_root / "move-source.txt").write_text(
                            "Celune agent diagnostic",
                            encoding="utf-8",
                        )
                    if name == "local_delete":
                        (tool_root / "delete.txt").write_text(
                            "Celune agent diagnostic",
                            encoding="utf-8",
                        )
                    arguments = _tool_arguments(tool_root, engine)[name]
                    context_task = _working_task(runtime, name)
                    if name == "resume_task":
                        runtime.pause_task(context_task.task_id)
                    context = runtime.get_context(context_task.task_id)
                    call = {
                        "id": f"agent-diagnostic-{name}",
                        "name": name,
                        "arguments": arguments,
                    }
                    with ExitStack() as stack:
                        effects = _patch_tool_side_effects(
                            stack,
                            engine,
                            name,
                            memory,
                        )
                        result = tool.execute(call, context)
                    if name in {"local_launch_process", "local_launch_application"}:
                        if effects.launcher is None or not effects.launcher.called:
                            raise RuntimeError(
                                "launch did not use the process test double"
                            )
                        if effects.launcher.call_args.kwargs.get("shell") is not False:
                            raise RuntimeError(
                                "process launch did not disable shell mode"
                            )
                    if name in {
                        "local_terminate_process",
                        "local_close_application",
                    } and (effects.process is None or not effects.process.terminated):
                        raise RuntimeError(
                            "termination did not use the process test double"
                        )
                    execution_result = cast(ToolExecutionResult, result)
                    status = execution_result["status"].value
                    if not schema.available:
                        if status != "failed":
                            raise RuntimeError("unavailable tool unexpectedly executed")
                    elif status != "succeeded":
                        raise RuntimeError(execution_result.get("error") or status)
                    checks.append(_record(name, "passed", f"status={status}"))
                except Exception as exc:
                    checks.append(
                        _record(name, "failed", str(exc) or type(exc).__name__)
                    )
    except Exception as exc:
        checks.append(
            _record("tools.fixture", "failed", str(exc) or type(exc).__name__)
        )
    finally:
        engine.agent_runtime = original_runtime
        engine._agent_tools = original_tools
        engine._agent_tool_schemas = original_schemas
        engine.mode = original_mode
        engine.speed = original_speed
        engine.reverb.strength = original_reverb
        engine.voice_prompt = original_voice_prompt
        engine._agent_needle_selector = old_needle
    return checks


def _tool_arguments(root: Path, engine: Celune) -> dict[str, JSON]:
    """Return safe valid arguments for each registered production operation."""
    arguments = {name: dict(values) for name, values in _AGENT_TOOL_ARGUMENTS.items()}
    current_voice = engine.current_voice
    if not isinstance(current_voice, str) or not current_voice:
        current_voice = "diagnostic"
    arguments["set_voice"] = {"voice": current_voice}
    arguments.update(
        {
            "local_list_directory": {"path": str(root), "limit": 5},
            "local_file_metadata": {"path": str(root / "source.txt")},
            "local_read_text": {"path": str(root / "source.txt")},
            "local_write_text": {
                "path": str(root / "written.txt"),
                "text": "temporary diagnostic file",
            },
            "local_make_directory": {"path": str(root / "directory")},
            "local_copy": {
                "source": str(root / "source.txt"),
                "destination": str(root / "copy.txt"),
            },
            "local_move": {
                "source": str(root / "move-source.txt"),
                "destination": str(root / "moved.txt"),
            },
            "local_delete": {"path": str(root / "delete.txt")},
            "local_inspect_process": {"pid": os.getpid()},
            "local_launch_process": {
                "executable": sys.executable,
                "arguments": [],
                "cwd": str(root),
            },
            "local_terminate_process": {
                "pid": 1,
                "expected_name": "diagnostic",
            },
            "local_discover_application": {"name": sys.executable},
            "local_launch_application": {
                "executable": sys.executable,
                "arguments": [],
                "cwd": str(root),
            },
            "local_close_application": {
                "pid": 1,
                "expected_name": "diagnostic",
            },
        }
    )
    return arguments


def _patch_tool_side_effects(
    stack: ExitStack,
    engine: Celune,
    name: str,
    memory: _DiagnosticMemoryStore,
) -> _ExternalEffects:
    """Replace persistent or external effects while retaining real handlers."""
    effects = _ExternalEffects()
    if name == "speak":
        stack.enter_context(patch.object(engine, "say", return_value=True))
    elif name == "stop_speech":
        stack.enter_context(
            patch.object(engine, "force_stop_speech", return_value=True)
        )
    elif name == "set_voice":
        stack.enter_context(
            patch.object(engine, "set_voice_and_wait", return_value=True)
        )
    elif name == "set_character":
        stack.enter_context(
            patch.object(engine, "set_cevoice_and_wait", return_value=True)
        )
    elif name == "sleep":
        stack.enter_context(patch.object(engine, "enter_sleep_mode", return_value=True))
    elif name == "wake":
        stack.enter_context(patch.object(engine, "wake_from_sleep", return_value=True))
    elif name == "clear_speech_queue":
        stack.enter_context(patch.object(engine, "_clear_queue"))
    elif name == "clear_recent_context":
        stack.enter_context(patch.object(engine, "_reset_persona_conversation"))
    elif name in {"remember", "recall", "forget"}:
        stack.enter_context(
            patch("celune.agent.tools._memory_store", return_value=memory)
        )
        stack.enter_context(
            patch("celune.agent.tools._character", return_value="diagnostic")
        )
    elif name in {
        "local_launch_process",
        "local_launch_application",
        "local_discover_application",
    }:
        stack.enter_context(
            patch("celune.agent.tools.shutil.which", return_value=sys.executable)
        )
        if name != "local_discover_application":
            effects.launcher = stack.enter_context(
                patch(
                    "celune.agent.tools.subprocess.Popen",
                    return_value=SimpleNamespace(pid=os.getpid()),
                )
            )
    elif name in {
        "local_inspect_process",
        "local_terminate_process",
        "local_close_application",
    }:
        effects.process = _DiagnosticProcess()
        stack.enter_context(
            patch(
                "celune.agent.tools.psutil.Process",
                return_value=effects.process,
            )
        )
    return effects


def _local_management_enabled(engine: Celune) -> bool:
    """Resolve the active opt-in using Celune's normal environment override."""
    from ..config import config_bool

    config = engine.config.get("agent")
    return config_bool(
        config if isinstance(config, dict) else None,
        "CELUNE_AGENT_FS_TOOLS",
        "fs_tools",
    )
