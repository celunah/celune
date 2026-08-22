# SPDX-License-Identifier: Apache-2.0
"""Deterministic explicit test-mode workflows for the Celune engine."""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Optional

from .i18n import string
from .typing.common import JSON
from .typing.agent import (
    AgentRoute,
    AgentTaskState,
    AgentToolExecutionStatus,
    AgentClassificationResult,
)

_AGENT_TEST_REQUEST = "Check the current working directory and report the result."

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
        return string(
            "test.agent_classification_failed",
            reason=route.failure.kind.value,
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


def run_agent_test(
    engine: Celune,
    timeout_seconds: float = 30.0,
) -> JSON:
    """Run one scripted action through Celune's production boundaries.

    The engine, router, runtime, permission policy, registered tool executor,
    Persona bridge, TTS model, TTS pipeline, and speech delivery remain the
    production implementations. The test registry contains one safe
    read-only operation, while Needle still loads and selects it through
    the normal model and schema-validation path. Persona still produces both
    the action intent and final response.

    Args:
        engine: A loaded Celune engine instance.
        timeout_seconds: Maximum time to wait for the final speech playback.

    Returns:
        JSON: The final test result recorded by the engine.
    """
    if timeout_seconds <= 0:
        raise ValueError("agent test timeout_seconds must be positive")

    task_id: Optional[str] = None
    try:
        _wait_for_persona(engine, timeout_seconds)
        _start_agent_test_pipeline(engine)
        route = engine.route_input(
            _AGENT_TEST_REQUEST,
            persona_ready=True,
        )
        metadata = route.routing_metadata
        task_id = (
            metadata.get("task_id")
            if isinstance(metadata, dict) and isinstance(metadata.get("task_id"), str)
            else None
        )
        route_failure = _agent_test_route_failure(route, task_id)
        if route_failure is not None:
            raise RuntimeError(route_failure)
        if task_id is None:
            raise RuntimeError(string("test.agent_task_not_started"))

        delivered = engine._run_agent_route(route)
        final_state = _task_state(engine, task_id)
        if final_state == AgentTaskState.IDLE.value:
            raise RuntimeError(string("test.agent_task_not_started"))
        if final_state != AgentTaskState.COMPLETED.value:
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
        if not engine.playback_done.wait(timeout=timeout_seconds):
            raise TimeoutError("agent test speech playback timed out")
        return engine.finish_test_mode(
            "agent",
            True,
            task_state=final_state,
            detail=f"tool={tool_id} status={status_value}",
        )
    except Exception as exc:
        return engine.finish_test_mode(
            "agent",
            False,
            task_state=_task_state(engine, task_id),
            detail=str(exc),
        )
