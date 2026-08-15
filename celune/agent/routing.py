# SPDX-License-Identifier: MIT
"""Conversation-first routing for Celune's typed agent boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, cast

from ..i18n import string
from ..pipeline import build_agent_classification_request
from ..typing.agent import (
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentChoiceResponse,
    AgentClassificationResult,
    AgentInputClassification,
    AgentInterruption,
    AgentInterruptionKind,
    AgentRequest,
    AgentRoute,
    AgentSession,
    AgentTask,
    AgentTaskState,
)
from ..typing.common import JSON, JSONSerializable
from ..typing.persona import PersonaClientResponse
from .runtime import AgentRuntime

if TYPE_CHECKING:
    from ..celune import Celune


_ACTION_START_RE = re.compile(
    r"^(?:(?:please|kindly)\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?"
    r"(?:(?:i\s+want\s+you\s+to|help\s+me\s+to)\s+)?"
    r"(?:delete|remove|open|read|inspect|check|set|change|create|make|move|"
    r"rename|send|close|launch|start|stop|list|find|run|install|download|"
    r"clear|empty|turn\s+(?:on|off))\b"
)
_QUESTION_START_RE = re.compile(
    r"^(?:how do i|how can i|what should i|why should i|can you explain|"
    r"could you explain|tell me how)\b"
)
_AMBIGUOUS_RE = re.compile(
    r"^(?:please\s+)?(?:do something|handle (?:this|that|it)|take care of "
    r"(?:this|that|it)|deal with (?:this|that|it)|make it better|fix (?:this|that|it)|"
    r"can you help(?: me)?|could you help(?: me)?)\??$"
)
_INTERRUPTION_INPUTS = {
    "hold on",
    "interrupt",
    "pause",
    "wait",
    "stop for a moment",
}
_CANCELLATION_INPUTS = {
    "cancel",
    "cancel it",
    "cancel the task",
    "abort",
    "abort it",
    "never mind",
    "nevermind",
    "stop",
    "stop it",
    "stop the task",
}
_APPROVAL_INPUTS = {
    "yes": AgentApprovalDecision.APPROVED,
    "y": AgentApprovalDecision.APPROVED,
    "approve": AgentApprovalDecision.APPROVED,
    "approved": AgentApprovalDecision.APPROVED,
    "allow": AgentApprovalDecision.APPROVED,
    "allowed": AgentApprovalDecision.APPROVED,
    "proceed": AgentApprovalDecision.APPROVED,
    "go ahead": AgentApprovalDecision.APPROVED,
    "no": AgentApprovalDecision.DENIED,
    "n": AgentApprovalDecision.DENIED,
    "deny": AgentApprovalDecision.DENIED,
    "denied": AgentApprovalDecision.DENIED,
    "decline": AgentApprovalDecision.DENIED,
    "declined": AgentApprovalDecision.DENIED,
}


class AgentInputRouter:
    """Classify input and route it through Persona or AgentRuntime."""

    def __init__(
        self,
        engine: Celune,
        runtime: AgentRuntime,
        session_id: str = "default",
    ) -> None:
        """Create a router bound to one engine and agent session."""
        if not session_id.strip():
            raise ValueError("agent routing session_id must not be empty")
        self.engine = engine
        self.runtime = runtime
        self.session_id = session_id

    def classify(
        self,
        text: str,
        *,
        persona_ready: bool = False,
    ) -> AgentClassificationResult:
        """Classify input without creating a task or changing lifecycle state."""
        clean_text = self._clean_text(text)
        local = self._classify_locally(clean_text)
        if local.requires_clarification and persona_ready:
            return self._classify_with_persona(clean_text, local)
        return local

    def route(
        self,
        text: str,
        *,
        persona_ready: bool = False,
    ) -> AgentClassificationResult:
        """Route one input through the active task or the conversation boundary."""
        clean_text = self._clean_text(text)
        active_task = self.runtime.get_active_task(self.session_id)
        if active_task is not None:
            return self._route_active_task(active_task, clean_text)

        result = self.classify(clean_text, persona_ready=persona_ready)
        if result.route != AgentRoute.TASK or result.task_request is None:
            return result

        task = self.runtime.create_task(result.task_request)
        metadata: JSON = {
            "task_id": task.task_id,
            "session_id": task.session_id,
        }
        return AgentClassificationResult(
            classification=result.classification,
            confidence=result.confidence,
            task_request=result.task_request,
            reason=result.reason,
            routing_metadata=metadata,
            route=result.route,
        )

    def _route_active_task(
        self,
        task: AgentTask,
        text: str,
    ) -> AgentClassificationResult:
        """Route control answers and steering input to an active task."""
        normalized = self._normalize(text)
        if normalized in _CANCELLATION_INPUTS:
            self.runtime.cancel_task(task.task_id)
            return self._active_result(task, AgentRoute.CANCELLATION, "cancellation")

        if task.state == AgentTaskState.AWAITING_APPROVAL:
            return self._route_approval(task, normalized)
        if task.state == AgentTaskState.AWAITING_CHOICE:
            return self._route_choice(task, text, normalized)

        if normalized in _INTERRUPTION_INPUTS:
            self.runtime.interrupt_task(
                task.task_id,
                AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
            )
            return self._active_result(task, AgentRoute.INTERRUPTION, "interruption")

        if task.state == AgentTaskState.IDLE:
            self.runtime.start_task(task.task_id)
        self.runtime.interrupt_task(
            task.task_id,
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING,
                instruction=text,
            ),
        )
        return self._active_result(task, AgentRoute.TASK_INPUT, "task_follow_up")

    def _route_approval(
        self,
        task: AgentTask,
        normalized: str,
    ) -> AgentClassificationResult:
        """Route a short approval answer to the pending typed request."""
        pending = self.runtime.get_pending_approval(task.task_id)
        if pending is None:
            return self._clarification(string("agent.approval_clarification"))
        decision = _APPROVAL_INPUTS.get(normalized)
        if decision is None:
            return self._clarification(string("agent.approval_clarification"))
        self.runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(pending.request_id, decision),
        )
        return self._active_result(
            task,
            AgentRoute.APPROVAL_RESPONSE,
            "approval_response",
        )

    def _route_choice(
        self,
        task: AgentTask,
        text: str,
        normalized: str,
    ) -> AgentClassificationResult:
        """Route an option or allowed freeform answer to a pending choice."""
        pending = self.runtime.get_pending_choice(task.task_id)
        if pending is None:
            return self._clarification(string("agent.choice_clarification"))

        choice_id: Optional[str] = None
        for option in pending.options:
            if normalized in {
                self._normalize(option.choice_id),
                self._normalize(option.label),
            }:
                choice_id = option.choice_id
                break

        freeform = text if choice_id is None and pending.allow_freeform else None
        if choice_id is None and freeform is None:
            return self._clarification(string("agent.choice_clarification"))

        self.runtime.respond_to_choice(
            task.task_id,
            AgentChoiceResponse(
                pending.request_id,
                choice_id=choice_id,
                freeform=freeform,
            ),
        )
        return self._active_result(task, AgentRoute.CHOICE_RESPONSE, "choice_response")

    def _classify_locally(self, text: str) -> AgentClassificationResult:
        """Handle obvious routing cases without asking the Persona model."""
        normalized = self._normalize(text)
        if _QUESTION_START_RE.match(normalized):
            return self._conversation_result(0.98, "question")
        if _AMBIGUOUS_RE.match(normalized):
            return self._clarification(string("agent.clarification_prompt"))
        if _ACTION_START_RE.match(normalized):
            return self._task_result(text, 0.96, "explicit_action")
        return self._conversation_result(0.98, "ordinary_conversation")

    def _classify_with_persona(
        self,
        text: str,
        fallback: AgentClassificationResult,
    ) -> AgentClassificationResult:
        """Resolve genuinely ambiguous input through the existing Persona VLM."""
        vision = getattr(self.engine, "vision", None)
        post = getattr(vision, "post", None)
        if not callable(post):
            return fallback

        try:
            response = cast(
                PersonaClientResponse,
                post(json=build_agent_classification_request(self.engine, text)),
            )
            response.raise_for_status()
            payload = response.json()
            candidate = self._classification_payload(payload)
            if candidate is None:
                return fallback
            return self._result_from_payload(text, candidate)
        except Exception:
            # Classifier failure must preserve the ordinary conversation fallback.
            return fallback

    def _classification_payload(
        self,
        payload: dict[str, JSONSerializable],
    ) -> Optional[JSON]:
        """Extract a structured classifier object from a Persona response."""
        if isinstance(payload.get("classification"), str):
            return payload

        for key in ("text", "response", "reply", "output", "content"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = value.strip()
            if candidate.startswith("```"):
                candidate = candidate.split("\n", 1)[-1]
                candidate = candidate.rsplit("```", 1)[0].strip()
            decoded = json.loads(candidate)
            if isinstance(decoded, dict):
                return cast(JSON, decoded)
        return None

    def _result_from_payload(
        self,
        text: str,
        payload: JSON,
    ) -> AgentClassificationResult:
        """Validate a Persona classifier payload into the public result type."""
        raw_classification = payload.get("classification")
        classification = (
            AgentInputClassification.TASK
            if isinstance(raw_classification, str)
            and raw_classification.casefold() in {"task", "action", "agent"}
            else AgentInputClassification.CONVERSATION
        )
        raw_confidence = payload.get("confidence", 0.5)
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (int, float))
            and not isinstance(raw_confidence, bool)
            else 0.5
        )
        requires_clarification = payload.get("requires_clarification") is True
        prompt_value = payload.get("clarification_prompt")
        prompt = prompt_value.strip() if isinstance(prompt_value, str) else None
        if requires_clarification and not prompt:
            prompt = string("agent.clarification_prompt")
        if confidence < 0.6:
            requires_clarification = True
            prompt = prompt or string("agent.clarification_prompt")

        raw_request = payload.get("task_request")
        task_text = raw_request.strip() if isinstance(raw_request, str) else text
        task_request = (
            self._make_request(task_text)
            if classification == AgentInputClassification.TASK
            and not requires_clarification
            else None
        )
        reason_value = payload.get("reason")
        reason = reason_value if isinstance(reason_value, str) else "persona_classifier"
        metadata_value = payload.get("routing_metadata")
        metadata = (
            cast(JSON, metadata_value) if isinstance(metadata_value, dict) else None
        )
        if requires_clarification:
            return AgentClassificationResult(
                classification=AgentInputClassification.CONVERSATION,
                confidence=confidence,
                requires_clarification=True,
                clarification_prompt=prompt,
                reason=reason,
                routing_metadata=metadata,
                route=AgentRoute.CLARIFICATION,
            )
        if classification == AgentInputClassification.TASK:
            return AgentClassificationResult(
                classification=classification,
                confidence=confidence,
                task_request=task_request,
                reason=reason,
                routing_metadata=metadata,
                route=AgentRoute.TASK,
            )
        return AgentClassificationResult(
            classification=classification,
            confidence=confidence,
            reason=reason,
            routing_metadata=metadata,
            route=AgentRoute.CONVERSATION,
        )

    def _task_result(
        self,
        text: str,
        confidence: float,
        reason: str,
    ) -> AgentClassificationResult:
        """Build an unambiguous task classification result."""
        return AgentClassificationResult(
            classification=AgentInputClassification.TASK,
            confidence=confidence,
            task_request=self._make_request(text),
            reason=reason,
            route=AgentRoute.TASK,
        )

    @staticmethod
    def _conversation_result(
        confidence: float,
        reason: str,
    ) -> AgentClassificationResult:
        """Build an ordinary conversation classification result."""
        return AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=confidence,
            reason=reason,
            route=AgentRoute.CONVERSATION,
        )

    @staticmethod
    def _clarification(prompt: str) -> AgentClassificationResult:
        """Build a clarification route without guessing a task."""
        return AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.4,
            requires_clarification=True,
            clarification_prompt=prompt,
            reason="ambiguous",
            route=AgentRoute.CLARIFICATION,
        )

    def _active_result(
        self,
        task: AgentTask,
        route: AgentRoute,
        reason: str,
    ) -> AgentClassificationResult:
        """Build a result for input consumed by an existing task."""
        return AgentClassificationResult(
            classification=AgentInputClassification.TASK,
            confidence=1.0,
            reason=reason,
            routing_metadata={"task_id": task.task_id},
            route=route,
        )

    def _make_request(self, text: str) -> AgentRequest:
        """Build a typed task request with the current conversation context."""
        raw_history = getattr(self.engine, "persona_history", ())
        history = tuple(
            cast(JSON, dict(message))
            for message in raw_history
            if isinstance(message, Mapping)
        )
        return AgentRequest(
            request=text,
            history=history,
            session=AgentSession(session_id=self.session_id),
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize and validate one input string."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("agent routing input must not be empty")
        return clean_text

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for conservative control and intent matching."""
        return re.sub(r"\s+", " ", text.casefold().strip(" \t\r\n.!?"))
