# SPDX-License-Identifier: MIT
"""Deterministic explicit test-mode workflows for the Celune engine."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, cast

from .agent.needle import NeedleHandler, NeedleToolSelector
from .typing.agent import (
    AgentRoute,
    AgentTaskState,
    AgentTool,
    AgentToolSchema,
    NeedleToolCatalog,
    NeedleToolCall,
)
from .typing.common import JSON

if TYPE_CHECKING:
    from .celune import Celune


class _ControlledNeedleHandler:
    """Provide one deterministic Needle selection for the controlled workflow."""

    def __init__(self) -> None:
        self.closed = False

    @staticmethod
    def catalog_for_tools(
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        available_only: bool = False,
    ) -> NeedleToolCatalog:
        """Reuse Needle's production catalog conversion for the test boundary."""
        return NeedleHandler.catalog_for_tools(
            tools,
            schemas=schemas,
            available_only=available_only,
        )

    def select_one_tool(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = 96,
    ) -> NeedleToolCall:
        """Select the registered status tool without model or network work."""
        del query, tools, max_new_tokens
        return {"name": "read_agent_status", "arguments": {}}

    def close(self) -> None:
        """Mark the controlled selector as closed."""
        self.closed = True


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
    production implementations. The single read-only Needle selection is
    scripted so the test cannot choose an unrelated tool. Persona still
    produces both the action intent and final response.

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
        selector = NeedleToolSelector(
            cast(NeedleHandler, _ControlledNeedleHandler()),
            engine._agent_tools,
            schemas=engine._agent_tool_schemas,
        )
        engine._agent_needle_selector = selector

        route = engine.route_input(
            "Check the current agent status.",
            persona_ready=True,
        )
        metadata = route.routing_metadata
        task_id = (
            metadata.get("task_id")
            if isinstance(metadata, dict) and isinstance(metadata.get("task_id"), str)
            else None
        )
        if route.route != AgentRoute.TASK or task_id is None:
            raise RuntimeError("agent test input did not create a task")

        delivered = engine._run_agent_route(route)
        final_state = _task_state(engine, task_id)
        if final_state != AgentTaskState.COMPLETED.value:
            raise RuntimeError(
                f"agent test task ended in unexpected state: {final_state or 'none'}"
            )
        if not delivered:
            raise RuntimeError("agent test response was not queued")
        if not engine.playback_done.wait(timeout=timeout_seconds):
            raise TimeoutError("agent test speech playback timed out")
        return engine.finish_test_mode(
            "agent",
            True,
            task_state=final_state,
        )
    except Exception as exc:
        return engine.finish_test_mode(
            "agent",
            False,
            task_state=_task_state(engine, task_id),
            detail=str(exc),
        )
