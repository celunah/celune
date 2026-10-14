# SPDX-License-Identifier: Apache-2.0
"""Adapters between the existing Persona boundary and AgentRuntime."""

from __future__ import annotations

from uuid import uuid4
from dataclasses import replace
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

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
from ..typing.persona import PersonaClientResponse
from ..conversation import _extract_persona_text, build_persona_request
from ..persona.impl import compact_persona_history, persona_history_messages

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

    def compact(self, context: AgentContext) -> AgentContext:
        """Compact Persona history and drop stale task-history references."""
        task = context.task
        if task is None:
            return context

        lease = None
        manager = getattr(self.engine, "component_locks", None)
        summarize = True
        if manager is not None:
            acquisition, lease = manager.try_acquire_lease(
                (ComponentLockRequirement(ComponentLockName.VLM),),
                ComponentLockOwner(
                    operation_id=f"agent-compact:{task.task_id}:{task.generation}",
                    task_id=task.task_id,
                    session_id=task.session_id,
                    generation_id=task.generation,
                ),
            )
            summarize = acquisition.acquired
        try:
            released_tokens = compact_persona_history(
                self.engine,
                context_size=task.config.context_size,
                compact_at=task.config.compact_at,
                force=True,
                summarize=summarize,
            )
        finally:
            if lease is not None:
                lease.release()
        request = replace(
            task.request,
            history=tuple(persona_history_messages(self.engine)),
        )
        task.request = request
        task.update_context_tokens(max(0, task.context_tokens - released_tokens))
        return replace(context, request=request)

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
            response_payload = response.json()
            prompt_tokens = getattr(response, "prompt_tokens", None)
            completion_tokens = getattr(response, "completion_tokens", None)
            if (
                context.task is not None
                and isinstance(prompt_tokens, int)
                and isinstance(completion_tokens, int)
            ):
                context.task.update_context_tokens(prompt_tokens + completion_tokens)
            spoken_text = _extract_persona_text(response_payload)
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
