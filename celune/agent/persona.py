# SPDX-License-Identifier: Apache-2.0
"""Adapters between the existing Persona boundary and AgentRuntime."""

from __future__ import annotations

from uuid import uuid4
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from ..typing.persona import PersonaClientResponse
from ..pipeline import _extract_persona_text, build_persona_request
from ..typing.agent import (
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentToolSchema,
)
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)

if TYPE_CHECKING:
    from ..celune import Celune


class PersonaAgentBridge:
    """Use the active Persona client for agent intent and result responses."""

    def __init__(
        self,
        engine: Celune,
        schemas: Mapping[str, AgentToolSchema],
    ) -> None:
        """Bind the bridge to one engine and its registered tool schemas."""
        self.engine = engine
        self.tool_schemas = tuple(schemas.values())

    def plan(self, context: AgentContext) -> AgentOutput:
        """Ask Persona for one natural-language action intent."""
        return self._generate(context, terminal=False)

    def respond(self, context: AgentContext) -> AgentOutput:
        """Ask Persona for the final response when no tool is selected."""
        return self._generate(context, terminal=True)

    def handle_tool_result(
        self,
        context: AgentContext,
        result: ToolResult,
    ) -> AgentOutput:
        """Return the structured tool result to Persona for the final reply."""
        del result
        return self._generate(context, terminal=True)

    def _generate(self, context: AgentContext, *, terminal: bool) -> AgentOutput:
        """Generate one Persona response through the existing request boundary."""
        vision = getattr(self.engine, "vision", None)
        post = getattr(vision, "post", None)
        if not callable(post):
            raise TypeError("Persona agent boundary is unavailable")

        lease = None
        manager = getattr(self.engine, "component_locks", None)
        if manager is not None:
            task = context.task
            owner = ComponentLockOwner(
                operation_id=(
                    f"agent-vlm:{task.task_id}:{task.generation}"
                    if task is not None
                    else f"persona:{uuid4().hex}"
                ),
                task_id=task.task_id if task is not None else None,
                session_id=task.session_id if task is not None else None,
                generation_id=task.generation if task is not None else None,
            )
            acquisition, lease = manager.try_acquire_lease(
                (ComponentLockRequirement(ComponentLockName.VLM),),
                owner,
            )
            if not acquisition.acquired:
                busy = acquisition.busy
                assert busy is not None
                return {
                    "tool_call": None,
                    "response": None,
                    "end": False,
                    "paused": True,
                    "busy": busy,
                }

        try:
            payload = build_persona_request(
                self.engine,
                context.request.request,
                agent_context=context,
                tool_schemas=self.tool_schemas,
            )
            response = cast(PersonaClientResponse, post(json=payload))
            response.raise_for_status()
            spoken_text = _extract_persona_text(response.json())
            if not spoken_text:
                raise RuntimeError("Persona returned an empty agent response")
            return {
                "tool_call": None,
                "response": spoken_text,
                "end": terminal,
                "paused": False,
            }
        finally:
            if lease is not None:
                lease.release()
