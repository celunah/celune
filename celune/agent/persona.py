# SPDX-License-Identifier: Apache-2.0
"""Adapters between the existing Persona boundary and AgentRuntime."""

from __future__ import annotations

from uuid import uuid4
from typing import TYPE_CHECKING, Optional, cast
from dataclasses import replace
from collections.abc import Mapping

from ..persona.impl import compact_persona_history, persona_history_messages
from ..typing.agent import (
    ToolCall,
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentToolSchema,
    AgentApprovalRequest,
)
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..typing.common import JSON, JSONSerializable
from ..typing.persona import PersonaClientResponse
from ..persona.conversation import extract_persona_text, build_persona_request


def _repair_persona_request(payload: JSON) -> JSON:
    """Ask Persona to replace a punctuation-only agent response once."""
    repaired = dict(payload)
    system = repaired.get("system")
    if not isinstance(system, str):
        return payload
    repair_instruction = (
        f"{system}\n\nThe previous response contained no spoken content. "
        "Follow the task-specific response instruction and return one complete, "
        "natural spoken sentence using only the supplied facts."
    )
    repaired["system"] = repair_instruction
    raw_messages = repaired.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        first_message = raw_messages[0]
        if isinstance(first_message, dict) and isinstance(
            first_message.get("content"), str
        ):
            repaired_message = dict(first_message)
            repaired_message["content"] = repair_instruction
            repaired_messages = list(raw_messages)
            repaired_messages[0] = repaired_message
            repaired["messages"] = cast(JSONSerializable, repaired_messages)
    return repaired


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

    def opening(self, context: AgentContext) -> AgentOutput:
        """Generate a short character-voiced acknowledgment before planning."""
        return self._generate(replace(context, opening=True), terminal=True)

    def approval(
        self,
        context: AgentContext,
        request: AgentApprovalRequest,
    ) -> AgentOutput:
        """Generate a character-voiced explanation of a pending approval."""
        return self._generate(
            context,
            terminal=True,
            pending_tool_call=request.tool_call,
        )

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

    def _generate(
        self,
        context: AgentContext,
        *,
        terminal: bool,
        pending_tool_call: Optional[ToolCall] = None,
    ) -> AgentOutput:
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
                pending_tool_call=pending_tool_call,
            )
            response = cast(PersonaClientResponse, post(json=payload))
            response.raise_for_status()
            response_payload = response.json()
            prompt_tokens = getattr(response, "prompt_tokens", None)
            completion_tokens = getattr(response, "completion_tokens", None)
            generated_tokens = 0
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                generated_tokens += prompt_tokens + completion_tokens
            spoken_text = extract_persona_text(response_payload)
            if not spoken_text or not any(
                character.isalnum() for character in spoken_text
            ):
                payload = _repair_persona_request(payload)
                response = cast(PersonaClientResponse, post(json=payload))
                response.raise_for_status()
                response_payload = response.json()
                prompt_tokens = getattr(response, "prompt_tokens", None)
                completion_tokens = getattr(response, "completion_tokens", None)
                if isinstance(prompt_tokens, int) and isinstance(
                    completion_tokens, int
                ):
                    generated_tokens += prompt_tokens + completion_tokens
                spoken_text = extract_persona_text(response_payload)
            if context.task is not None and generated_tokens:
                context.task.update_context_tokens(generated_tokens)
            if not spoken_text or not any(
                character.isalnum() for character in spoken_text
            ):
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
