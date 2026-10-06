# SPDX-License-Identifier: Apache-2.0
"""Explicit test-mode workflows for the Celune engine."""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Optional

from .i18n import string
from .utils import format_error
from .typing.agent import (
    AgentRoute,
    AgentTaskState,
    AgentToolExecutionStatus,
    AgentClassificationResult,
)
from .typing.common import JSON
from .agent.diagnostics import run_agent_feature_checks

_AGENT_TEST_REQUEST = "Check the current working directory and report the result."
_LIVE_CHECKS = (
    "live.persona",
    "live.routing",
    "live.task",
    "live.playback",
)

if TYPE_CHECKING:
    from .celune import Celune


def _task_state(engine: Celune, task_id: Optional[str]) -> Optional[str]:
    """Return a task state label when the controlled route created a task."""
    if task_id is None:
        return None
    try:
        return engine.agent_runtime.get_task(task_id).state.value
    except ValueError:
        return None


def _start_agent_test_pipeline(engine: Celune) -> None:
    """Start the production speech workers for the CLI's fake backend."""
    if not engine.backend.is_fake:
        return

    playback_thread = engine.playback_thread
    if playback_thread is None or not playback_thread.is_alive():
        engine.loaded = True
        engine.model_ready.set()
        pipeline_thread = threading.Thread(
            target=engine._run_pipeline_jobs,
            daemon=True,
        )
        engine._playback_thread = pipeline_thread
        pipeline_thread.start()

    if engine.locked:
        engine._release_pipeline()


def _agent_test_route_failure(
    route: AgentClassificationResult, task_id: Optional[str]
) -> Optional[str]:
    """Describe why the controlled agent input did not start a task."""
    if route.failure is not None:
        return (
            f"{string('test.agent_classification_failed')}: {route.failure.kind.value}"
        )
    if route.route != AgentRoute.TASK:
        return string("test.agent_no_task_detected")
    if task_id is None:
        return string("test.agent_task_not_started")
    return None


def _wait_for_persona(engine: Celune, timeout_seconds: float) -> None:
    """Wait for the real Persona model to finish loading for agent test mode."""
    deadline = time.monotonic() + timeout_seconds
    while not engine.persona_ready:
        if not engine.persona_loading:
            raise RuntimeError("agent test Persona is unavailable")
        if time.monotonic() >= deadline:
            raise TimeoutError("agent test Persona loading timed out")
        time.sleep(0.1)


def _detail_for_error(engine: Celune, exc: Exception) -> str:
    """Return the normal or debug diagnostic for one live check failure."""
    detail = str(exc) or string("test.agent_failed")
    if getattr(engine, "log_level", "info") == "debug":
        return format_error(exc, "debug")
    return detail


def _check(name: str, status: str, detail: str) -> JSON:
    """Create one JSON-compatible diagnostic check record."""
    return {"name": name, "status": status, "detail": detail}


def _agent_test_succeeded(result: Optional[JSON]) -> bool:
    """Return whether a completed agent diagnostic recorded overall success."""
    return result is not None and result.get("success") is True


def _skip_live_checks(checks: list[JSON], names: tuple[str, ...], reason: str) -> None:
    """Mark live stages skipped when an earlier prerequisite failed."""
    checks.extend(_check(name, "skipped", reason) for name in names)


def _run_live_checks(
    engine: Celune,
    timeout_seconds: float,
    *,
    startup_success: bool,
    startup_detail: Optional[str],
) -> tuple[list[JSON], Optional[str]]:
    """Run the configured Persona, Needle, tool, response, and speech path."""
    checks: list[JSON] = []
    task_id: Optional[str] = None
    if not startup_success:
        checks.append(
            _check(
                "live.startup",
                "failed",
                startup_detail or string("test.agent_startup_failed"),
            )
        )
        _skip_live_checks(checks, _LIVE_CHECKS, string("test.agent_startup_required"))
        return checks, task_id

    try:
        _wait_for_persona(engine, timeout_seconds)
    except Exception as exc:
        checks.append(_check("live.persona", "failed", _detail_for_error(engine, exc)))
        _skip_live_checks(
            checks,
            _LIVE_CHECKS[1:],
            string("test.agent_persona_required"),
        )
        return checks, task_id
    checks.append(_check("live.persona", "passed", string("test.agent_persona_ready")))

    try:
        _start_agent_test_pipeline(engine)
        route = engine.route_input(_AGENT_TEST_REQUEST, persona_ready=True)
    except Exception as exc:
        checks.append(_check("live.routing", "failed", _detail_for_error(engine, exc)))
        _skip_live_checks(
            checks,
            _LIVE_CHECKS[2:],
            string("test.agent_routing_required"),
        )
        return checks, task_id

    metadata = route.routing_metadata
    task_id = (
        metadata.get("task_id")
        if isinstance(metadata, dict) and isinstance(metadata.get("task_id"), str)
        else None
    )
    route_failure = _agent_test_route_failure(route, task_id)
    if route_failure is None and task_id is None:
        route_failure = string("test.agent_task_not_started")
    if route_failure is not None or task_id is None:
        checks.append(
            _check(
                "live.routing",
                "failed",
                route_failure or string("test.agent_task_not_started"),
            )
        )
        _skip_live_checks(
            checks,
            _LIVE_CHECKS[2:],
            string("test.agent_routing_required"),
        )
        return checks, task_id
    checks.append(_check("live.routing", "passed", string("test.agent_routing_passed")))

    try:
        delivered = engine._run_agent_route(route)
        final_state = _task_state(engine, task_id)
        if final_state != AgentTaskState.COMPLETED.value:
            if final_state in {
                AgentTaskState.FAILED.value,
                AgentTaskState.ABORTED.value,
                AgentTaskState.CANCELLED.value,
            }:
                engine.playback_done.wait(timeout=timeout_seconds)
            raise RuntimeError(
                f"agent test task ended in unexpected state: {final_state or 'none'}"
            )
        if not delivered:
            raise RuntimeError("agent test response was not queued")
        tool_result = engine.agent_runtime.get_context(task_id).last_tool_result
        if not isinstance(tool_result, dict):
            raise TypeError("agent test completed without executing a tool")
        tool_id = tool_result.get("tool_id")
        tool_status = tool_result.get("status")
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise RuntimeError("agent test tool result did not identify a tool")
        status_value = (
            tool_status.value
            if isinstance(tool_status, AgentToolExecutionStatus)
            else tool_status
            if isinstance(tool_status, str)
            else "unknown"
        )
        if status_value != AgentToolExecutionStatus.SUCCEEDED.value:
            raise RuntimeError(f"agent test tool execution failed: {status_value}")
        checks.append(
            _check(
                "live.task",
                "passed",
                f"tool={tool_id} status={status_value} state={final_state}",
            )
        )
    except Exception as exc:
        checks.append(_check("live.task", "failed", _detail_for_error(engine, exc)))
        checks.append(
            _check(
                "live.playback",
                "failed" if isinstance(exc, TimeoutError) else "skipped",
                _detail_for_error(engine, exc)
                if isinstance(exc, TimeoutError)
                else string("test.agent_task_required"),
            )
        )
        return checks, task_id

    if not engine.playback_done.wait(timeout=timeout_seconds):
        checks.append(
            _check(
                "live.playback",
                "failed",
                string("test.agent_playback_timeout"),
            )
        )
    else:
        checks.append(
            _check("live.playback", "passed", string("test.agent_playback_passed"))
        )
    return checks, task_id


def run_agent_test(
    engine: Celune,
    timeout_seconds: float = 30.0,
    *,
    startup_success: bool = True,
    startup_detail: Optional[str] = None,
) -> JSON:
    """Run isolated feature checks and the configured live agent workflow.

    The test-only model catalog remains read-only. Every active production tool
    handler is exercised separately using temporary files and adapters for
    speech, memory, process, and application side effects.

    Args:
        engine: A Celune engine instance.
        timeout_seconds: Maximum wait for Persona readiness and speech playback.
        startup_success: Whether the UI startup callback reported success.
        startup_detail: Optional startup failure detail for the report.

    Returns:
        JSON: The aggregate report recorded by the engine.
    """
    if timeout_seconds <= 0:
        raise ValueError("agent test timeout_seconds must be positive")

    checks: list[JSON] = []
    try:
        checks.extend(run_agent_feature_checks(engine))
    except Exception as exc:
        checks.append(
            _check("features.setup", "failed", _detail_for_error(engine, exc))
        )

    live_checks, task_id = _run_live_checks(
        engine,
        timeout_seconds,
        startup_success=startup_success,
        startup_detail=startup_detail,
    )
    checks.extend(live_checks)
    final_state = _task_state(engine, task_id)
    success = not any(check.get("status") == "failed" for check in checks)
    return engine.finish_test_mode(
        "agent",
        success,
        task_state=final_state,
        checks=checks,
    )
