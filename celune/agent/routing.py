# SPDX-License-Identifier: Apache-2.0
"""Conversation-first routing for Celune's typed agent boundary."""

from __future__ import annotations

import json
from uuid import uuid4
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, cast

from ..i18n import string
from .runtime import AgentRuntime
from ..modes import mode_allows_agents
from ..typing.common import JSON, JSONSerializable
from ..typing.persona import PersonaClientResponse
from ..pipeline import build_agent_classification_request
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..typing.agent import (
    AgentTask,
    AgentRoute,
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentInterruption,
    AgentFailureReason,
    AgentChoiceResponse,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentInterruptionKind,
    AgentInputClassification,
    AgentClassificationResult,
    AgentClassificationFailure,
    AgentClassificationFailureKind,
)

if TYPE_CHECKING:
    from ..celune import Celune


class _EmptyClassifierOutput(ValueError):
    """Identify a response that contains an empty structured-output field."""


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
            return self._classifier_failure(
                AgentClassificationFailureKind.PERSONA_UNAVAILABLE,
                "Persona classifier is unavailable",
            )
        return self._classify_with_persona(clean_text)

    def route(
        self,
        text: str,
        *,
        persona_ready: bool = False,
    ) -> AgentClassificationResult:
        """Route one input through the active task or the conversation boundary."""
        clean_text = self._clean_text(text)
        active_task = self.runtime.get_active_task(self.session_id)
        agents_enabled = mode_allows_agents(getattr(self.engine, "mode", "agent"))
        if getattr(self.engine, "backend_mode", None) == "agent_test":
            agents_enabled = True
        if not agents_enabled:
            return self._conversation_result(1.0, "agent_mode_disabled")
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
            intent=result.intent,
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
            return self._classifier_failure(
                AgentClassificationFailureKind.PERSONA_UNAVAILABLE,
                "Persona classifier is unavailable",
                task=task,
            )

        result = self._classify_with_persona(
            text,
            task=task,
        )
        if result.failure is not None:
            if task.state != AgentTaskState.CANCELLING:
                self.runtime.fail_task(
                    task.task_id,
                    AgentFailureReason.INTERNAL_ERROR,
                    result.failure.detail,
                )
            return result
        if result.route == AgentRoute.CLARIFICATION:
            return result
        if result.route == AgentRoute.CANCELLATION:
            self.runtime.cancel_task(task.task_id)
            return self._active_result(
                task,
                AgentRoute.CANCELLATION,
                "cancellation",
                intent=result.intent,
            )

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
                intent=result.intent,
            )

        if task.state == AgentTaskState.AWAITING_APPROVAL:
            return self._route_approval(task, result)
        if task.state == AgentTaskState.AWAITING_CHOICE:
            return self._route_choice(task, result)

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
            return self._active_result(
                task,
                AgentRoute.TASK_INPUT,
                "task_follow_up",
                intent=result.intent,
            )

        if task.state == AgentTaskState.IDLE:
            self.runtime.start_task(task.task_id)
        self.runtime.steer_task(
            task.task_id,
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING,
                instruction=text,
            ),
        )
        return self._active_result(
            task,
            AgentRoute.TASK_INPUT,
            "task_follow_up",
            intent=result.intent,
        )

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
            intent=result.intent,
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
            intent=result.intent,
        )

    def _classify_with_persona(
        self,
        text: str,
        *,
        task: Optional[AgentTask] = None,
    ) -> AgentClassificationResult:
        """Resolve natural-language routing through the existing Persona VLM."""
        vision = getattr(self.engine, "vision", None)
        post = getattr(vision, "post", None)
        if not callable(post):
            return self._classifier_failure(
                AgentClassificationFailureKind.PERSONA_UNAVAILABLE,
                "Persona classifier is unavailable",
                task=task,
            )

        manager = getattr(self.engine, "component_locks", None)
        lease = None
        if manager is not None:
            acquisition, lease = manager.try_acquire_lease(
                (ComponentLockRequirement(ComponentLockName.VLM),),
                ComponentLockOwner(operation_id=f"classifier:{uuid4().hex}"),
            )
            if not acquisition.acquired:
                return self._classifier_failure(
                    AgentClassificationFailureKind.BUSY,
                    "Persona classifier is busy",
                    task=task,
                )

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
            for attempt in range(2):
                try:
                    response = cast(
                        PersonaClientResponse,
                        post(json=request_payload),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    candidate = self._classification_payload(payload)
                    return self._result_from_payload(text, candidate, task=task)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    if attempt == 0:
                        request_payload = self._repair_classification_request(
                            request_payload,
                            reason=str(exc),
                        )
                        continue
                    raise
            raise RuntimeError("classifier retry loop ended unexpectedly")
        except json.JSONDecodeError as exc:
            return self._classifier_failure(
                AgentClassificationFailureKind.MALFORMED_OUTPUT,
                str(exc),
                task=task,
            )
        except _EmptyClassifierOutput as exc:
            return self._classifier_failure(
                AgentClassificationFailureKind.EMPTY_OUTPUT,
                str(exc),
                task=task,
            )
        except (TypeError, ValueError) as exc:
            return self._classifier_failure(
                AgentClassificationFailureKind.INVALID_SCHEMA,
                str(exc),
                task=task,
            )
        except Exception as exc:
            return self._classifier_failure(
                AgentClassificationFailureKind.TRANSPORT,
                str(exc),
                task=task,
            )
        finally:
            if lease is not None:
                lease.release()

    def _classification_payload(
        self,
        payload: JSONSerializable,
    ) -> JSON:
        """Extract a structured classifier object from a Persona response."""
        if not isinstance(payload, dict):
            raise TypeError("classifier response must be a JSON object")
        if isinstance(payload.get("classification"), str):
            return cast(JSON, payload)

        found_field = False
        found_text = False
        for key in ("text", "response", "reply", "output", "content"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            found_field = True
            if not value.strip():
                continue
            found_text = True
            candidate = value.strip()
            if candidate.startswith("```"):
                candidate = candidate.split("\n", 1)[-1]
                candidate = candidate.rsplit("```", 1)[0].strip()
            decoded = json.loads(candidate)
            if isinstance(decoded, dict):
                return cast(JSON, decoded)
            raise TypeError("classifier response content must be a JSON object")
        if found_field and not found_text:
            raise _EmptyClassifierOutput("classifier response output was empty")
        if not found_text:
            raise ValueError("classifier response did not contain structured output")
        raise ValueError("classifier response content was empty")

    @staticmethod
    def _repair_classification_request(
        request_payload: dict[str, JSONSerializable],
        *,
        reason: str,
    ) -> dict[str, JSONSerializable]:
        """Request one corrected retry through the existing Persona path."""
        repaired = dict(request_payload)
        system = repaired.get("system")
        if isinstance(system, str):
            repair_instruction = (
                f"{system}\n\nThe previous routing output was rejected: {reason}. "
                "Return the corrected classification object now, with no prose or "
                "Markdown. If there is no active task context and the input is an "
                "action request, use classification task and route task exactly. "
                "Use task_input only when an active task context is supplied."
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
                    repaired["messages"] = repaired_messages
        return repaired

    def _result_from_payload(
        self,
        text: str,
        payload: JSON,
        *,
        task: Optional[AgentTask] = None,
    ) -> AgentClassificationResult:
        """Validate a Persona classifier payload into the public result type."""
        raw_classification = payload.get("classification")
        if not isinstance(raw_classification, str):
            raise TypeError("classifier classification is required")
        normalized_classification = raw_classification.casefold()
        if normalized_classification in {"task", "action", "agent"}:
            classification = AgentInputClassification.TASK
        elif normalized_classification == "conversation":
            classification = AgentInputClassification.CONVERSATION
        else:
            raise ValueError("classifier classification is invalid")

        raw_confidence = payload.get("confidence")
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not 0.0 <= raw_confidence <= 1.0
        ):
            raise ValueError("classifier confidence must be between 0 and 1")
        confidence = float(raw_confidence)
        requires_clarification = payload.get("requires_clarification") is True
        prompt_value = payload.get("clarification_prompt")
        prompt = prompt_value.strip() if isinstance(prompt_value, str) else None
        if requires_clarification and not prompt:
            prompt = string("agent.clarification_prompt")
        route = self._route_from_payload(payload, classification, task)
        if classification == AgentInputClassification.CONVERSATION and route not in {
            AgentRoute.CONVERSATION,
            AgentRoute.CLARIFICATION,
        }:
            raise ValueError("conversation classification has an incompatible route")
        if (
            classification == AgentInputClassification.TASK
            and task is None
            and route != AgentRoute.TASK
        ):
            raise ValueError("new task classification has an incompatible route")
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
        raw_intent = payload.get("intent")
        if raw_intent is not None and not isinstance(raw_intent, str):
            raise ValueError("classifier intent must be a string or null")
        intent = raw_intent.strip() if isinstance(raw_intent, str) else None
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
                intent=None,
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
                intent=intent,
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
            intent=None,
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

    def _classifier_failure(
        self,
        kind: AgentClassificationFailureKind,
        detail: str,
        *,
        task: Optional[AgentTask] = None,
    ) -> AgentClassificationResult:
        """Return an observable fail-closed classifier result."""
        failure = AgentClassificationFailure(kind, detail or kind.value)
        metadata: JSON = {"classifier_failure": failure.to_json()}
        if task is not None:
            metadata["task_id"] = task.task_id
        log = getattr(self.engine, "log", None)
        if callable(log):
            log_level = getattr(self.engine, "log_level", "info")
            if log_level in {"verbose", "debug"}:
                log(
                    f"{string('agent.classifier_failed')}: {detail or kind.value}",
                    "warning",
                    loglevel="verbose",
                )
            else:
                log(string("agent.classifier_failed_summary"), "warning")
        if task is not None:
            return AgentClassificationResult(
                classification=AgentInputClassification.CONVERSATION,
                confidence=0.0,
                requires_clarification=True,
                clarification_prompt=string("agent.classifier_unavailable"),
                reason="classifier_failure",
                routing_metadata=metadata,
                route=AgentRoute.CLARIFICATION,
                failure=failure,
            )
        return AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.0,
            reason="classifier_failure",
            routing_metadata=metadata,
            route=AgentRoute.CONVERSATION,
            failure=failure,
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
        intent: Optional[str] = None,
    ) -> AgentClassificationResult:
        """Build a result for input consumed by an existing task."""
        return AgentClassificationResult(
            classification=AgentInputClassification.TASK,
            confidence=1.0,
            intent=intent,
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
        if raw_route is not None:
            if not isinstance(raw_route, str) or raw_route not in valid_routes:
                raise ValueError("classifier route is invalid")
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
