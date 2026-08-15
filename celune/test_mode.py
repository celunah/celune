# SPDX-License-Identifier: MIT
"""Deterministic explicit test-mode workflows for the Celune engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
from typing import TYPE_CHECKING, Optional, cast

from .agent.needle import NeedleHandler, NeedleToolSelector
from .persona.impl import PersonaClient
from .typing.agent import (
    AgentRoute,
    AgentTaskState,
    AgentTool,
    AgentToolSchema,
    NeedleToolCatalog,
    NeedleToolCall,
)
from .typing.common import JSON
from .typing.persona import PersonaClientResponse

if TYPE_CHECKING:
    from .celune import Celune


class _ControlledPersonaClient:
    """Return fixed Persona responses while preserving the production bridge."""

    def __init__(self) -> None:
        self.request_count = 0

    def post(self, json: JSON) -> PersonaClientResponse:
        """Return one action intent followed by one final response."""
        del json
        self.request_count += 1
        response = (
            "Read the current agent status."
            if self.request_count == 1
            else "The controlled agent test completed successfully."
        )
        return PersonaClientResponse({"text": response})


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


def run_agent_test(
    engine: Celune,
    timeout_seconds: float = 30.0,
) -> JSON:
    """Run one controlled agent task through Celune's production boundaries.

    The engine, router, runtime, permission policy, registered tool executor,
    Needle selector adapter, Persona bridge, and speech delivery remain the
    production implementations. Only the model responses at the Persona and
    Needle boundaries are deterministic so this explicit test is guaranteed to
    complete without downloading or loading a model checkpoint.

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
        controlled_persona = _ControlledPersonaClient()
        engine.vision = cast(PersonaClient, controlled_persona)
        engine.persona_ready = True
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
