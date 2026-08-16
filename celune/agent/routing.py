# SPDX-License-Identifier: MIT
"""Conversation-first routing for Celune's typed agent boundary."""

from __future__ import annotations

import json
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
        if not persona_ready:
            return self._conversation_result(0.0, "persona_classifier_unavailable")
        return self._classify_with_persona(
            clean_text,
            self._conversation_result(0.0, "persona_classifier_unavailable"),
        )

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
            return self._route_active_task(active_task, clean_text, persona_ready)

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
            approval_decision=result.approval_decision,
            choice_id=result.choice_id,
            choice_freeform=result.choice_freeform,
            interruption_kind=result.interruption_kind,
        )

    def _route_active_task(
        self,
        task: AgentTask,
        text: str,
        persona_ready: bool,
    ) -> AgentClassificationResult:
        """Route semantic control answers and steering to an active task."""
        if not persona_ready:
            return self._clarification(string("agent.clarification_prompt"))

        result = self._classify_with_persona(
            text,
            self._clarification(string("agent.clarification_prompt")),
            task=task,
        )
        if result.route == AgentRoute.CLARIFICATION:
            return result
        if result.route == AgentRoute.CANCELLATION:
            self.runtime.cancel_task(task.task_id)
            return self._active_result(task, AgentRoute.CANCELLATION, "cancellation")

        if result.route == AgentRoute.INTERRUPTION:
            interruption_kind = (
                result.interruption_kind or AgentInterruptionKind.USER_INTERRUPT
            )
            self.runtime.interrupt_task(
                task.task_id,
                AgentInterruption(interruption_kind),
            )
            return self._active_result(
                task,
                AgentRoute.INTERRUPTION,
                "interruption",
                interruption_kind=interruption_kind,
            )

        if result.route in {AgentRoute.TASK_INPUT, AgentRoute.CONVERSATION}:
            if task.state == AgentTaskState.IDLE:
                self.runtime.start_task(task.task_id)
            self.runtime.steer_task(
                task.task_id,
                AgentInterruption(
                    AgentInterruptionKind.USER_STEERING,
                    instruction=text,
                ),
            )
            return self._active_result(task, AgentRoute.TASK_INPUT, "task_follow_up")

        if task.state == AgentTaskState.AWAITING_APPROVAL:
            return self._route_approval(task, result)
        if task.state == AgentTaskState.AWAITING_CHOICE:
            return self._route_choice(task, result)

        if task.state == AgentTaskState.IDLE:
            self.runtime.start_task(task.task_id)
        self.runtime.steer_task(
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
        result: AgentClassificationResult,
    ) -> AgentClassificationResult:
        """Route a semantically classified approval answer."""
        pending = self.runtime.get_pending_approval(task.task_id)
        if pending is None or result.approval_decision is None:
            return self._clarification(string("agent.approval_clarification"))
        self.runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(pending.request_id, result.approval_decision),
        )
        return self._active_result(
            task,
            AgentRoute.APPROVAL_RESPONSE,
            "approval_response",
            approval_decision=result.approval_decision,
        )

    def _route_choice(
        self,
        task: AgentTask,
        result: AgentClassificationResult,
    ) -> AgentClassificationResult:
        """Route a semantically classified answer to a pending choice."""
        pending = self.runtime.get_pending_choice(task.task_id)
        if pending is None:
            return self._clarification(string("agent.choice_clarification"))

        choice_id = result.choice_id
        if choice_id is not None and not any(
            option.choice_id == choice_id for option in pending.options
        ):
            choice_id = None
        freeform = result.choice_freeform if pending.allow_freeform else None
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
        return self._active_result(
            task,
            AgentRoute.CHOICE_RESPONSE,
            "choice_response",
            choice_id=choice_id,
            choice_freeform=freeform,
        )

    def _classify_with_persona(
        self,
        text: str,
        fallback: AgentClassificationResult,
        *,
        task: Optional[AgentTask] = None,
    ) -> AgentClassificationResult:
        """Resolve natural-language routing through the existing Persona VLM."""
        vision = getattr(self.engine, "vision", None)
        post = getattr(vision, "post", None)
        if not callable(post):
            return fallback

        try:
            if task is None:
                request_payload = build_agent_classification_request(
                    self.engine,
                    text,
                )
            else:
                request_payload = build_agent_classification_request(
                    self.engine,
                    text,
                    routing_context=self._routing_context(task),
                )
            response = cast(
                PersonaClientResponse,
                post(json=request_payload),
            )
            response.raise_for_status()
            payload = response.json()
            candidate = self._classification_payload(payload)
            if candidate is None:
                return fallback
            return self._result_from_payload(text, candidate, task=task)
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
        *,
        task: Optional[AgentTask] = None,
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

        route = self._route_from_payload(payload, classification, task)
        if task is not None and not requires_clarification:
            classification = AgentInputClassification.TASK

        raw_request = payload.get("task_request")
        task_text = raw_request.strip() if isinstance(raw_request, str) else text
        task_request = (
            self._make_request(task_text)
            if task is None
            and classification == AgentInputClassification.TASK
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
                route=route,
                approval_decision=self._approval_decision(
                    payload.get("approval_decision")
                ),
                choice_id=(
                    payload.get("choice_id")
                    if isinstance(payload.get("choice_id"), str)
                    else None
                ),
                choice_freeform=(
                    payload.get("choice_freeform")
                    if isinstance(payload.get("choice_freeform"), str)
                    else None
                ),
                interruption_kind=self._interruption_kind(
                    payload.get("interruption_kind")
                ),
            )
        return AgentClassificationResult(
            classification=classification,
            confidence=confidence,
            reason=reason,
            routing_metadata=metadata,
            route=route,
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
        *,
        approval_decision: Optional[AgentApprovalDecision] = None,
        choice_id: Optional[str] = None,
        choice_freeform: Optional[str] = None,
        interruption_kind: Optional[AgentInterruptionKind] = None,
    ) -> AgentClassificationResult:
        """Build a result for input consumed by an existing task."""
        return AgentClassificationResult(
            classification=AgentInputClassification.TASK,
            confidence=1.0,
            reason=reason,
            routing_metadata={"task_id": task.task_id},
            route=route,
            approval_decision=approval_decision,
            choice_id=choice_id,
            choice_freeform=choice_freeform,
            interruption_kind=interruption_kind,
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
    def _route_from_payload(
        payload: JSON,
        classification: AgentInputClassification,
        task: Optional[AgentTask],
    ) -> AgentRoute:
        """Read a validated route value, deriving the default route when absent."""
        raw_route = payload.get("route")
        valid_routes = tuple(route.value for route in AgentRoute)
        if isinstance(raw_route, str) and raw_route in valid_routes:
            return AgentRoute(raw_route)
        if task is not None:
            return AgentRoute.TASK_INPUT
        if classification == AgentInputClassification.TASK:
            return AgentRoute.TASK
        return AgentRoute.CONVERSATION

    @staticmethod
    def _approval_decision(
        value: JSONSerializable,
    ) -> Optional[AgentApprovalDecision]:
        """Convert a classifier approval value into its typed decision."""
        if not isinstance(value, str):
            return None
        try:
            return AgentApprovalDecision(value)
        except ValueError:
            return None

    @staticmethod
    def _interruption_kind(
        value: JSONSerializable,
    ) -> Optional[AgentInterruptionKind]:
        """Convert a classifier interruption value into its typed kind."""
        if not isinstance(value, str):
            return None
        try:
            return AgentInterruptionKind(value)
        except ValueError:
            return None

    def _routing_context(self, task: Optional[AgentTask]) -> Optional[JSON]:
        """Build the classifier context for an active task, if one exists."""
        if task is None:
            return None
        context: JSON = {
            "active_task": {
                "task_id": task.task_id,
                "state": task.state.value,
            }
        }
        pending_approval = self.runtime.get_pending_approval(task.task_id)
        if pending_approval is not None:
            context["pending_approval"] = {
                "request_id": pending_approval.request_id,
                "prompt": pending_approval.prompt,
            }
        pending_choice = self.runtime.get_pending_choice(task.task_id)
        if pending_choice is not None:
            context["pending_choice"] = {
                "request_id": pending_choice.request_id,
                "prompt": pending_choice.prompt,
                "options": [
                    {"id": option.choice_id, "label": option.label}
                    for option in pending_choice.options
                ],
                "allow_freeform": pending_choice.allow_freeform,
            }
        return context
