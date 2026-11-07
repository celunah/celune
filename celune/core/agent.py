# SPDX-License-Identifier: Apache-2.0
"""Agent routing and tool execution methods."""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Union, Optional, cast

from ..i18n import string
from ..exceptions import (
    ModelConsentDeclined,
    ModelConsentRequired,
    NeedleSelectionError,
)
from ..typing.agent import (
    ToolCall,
    AgentOutput,
    AgentContext,
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentAbortReason,
    AgentInterruption,
    AgentToolSelector,
    ToolExecutionResult,
    AgentInterruptionKind,
    AgentInputClassification,
    AgentToolExecutionStatus,
    AgentClassificationResult,
    AgentClassificationFailure,
)
from ..paths import huggingface_progress
from ..utils import format_error_message
from ..agent.needle import NeedleToolSelector
from ..typing.locks import ComponentLockName
from .conversation import deliver_persona_response, wait_for_persona_playback

_UNSAFE_AGENT_PREAMBLE_RE = re.compile(
    r"\b(?:i\s+will|i['’]ll|let\s+me|ready|available|standing|"
    r"start(?:ed|ing)?|begin(?:ning)?|check(?:ed|ing)?|"
    r"look|found|verif(?:y|ied|ying)|confirm(?:ed|ing)?|result|work(?:ing|ed)?|"
    r"access(?:ible)?|permission|exist(?:s|ed)?)\b",
    re.IGNORECASE,
)


def _is_plain_agent_preamble(text: str) -> bool:
    """Reject generated openings that claim status, facts, or task progress."""
    normalized = " ".join(text.split())
    sentence_count = len(re.findall(r"[.!?](?:[\"'”’)]*)?(?=\s|$)", normalized))
    return (
        bool(normalized)
        and any(character.isalnum() for character in normalized)
        and len(normalized) <= 100
        and len(normalized.split()) <= 12
        and sentence_count <= 1
        and not any(
            character.isdigit() or character in "\\/:" for character in normalized
        )
        and _UNSAFE_AGENT_PREAMBLE_RE.search(normalized) is None
    )


def _wait_for_persona_initialization(self) -> bool:
    """Wait for Persona initialization and VLM ownership to settle."""
    from .lifecycle import start_persona_background_load

    waiter = threading.Event()
    while not self.exit_requested:
        with self._model_lock:
            persona_ready = self.persona_ready
            persona_loading = self.persona_loading
            vision = self.vision
        component_locks = self.component_locks
        persona_busy = (
            component_locks is not None
            and component_locks.owner_for(ComponentLockName.VLM) is not None
        )
        if persona_ready and not persona_busy:
            return True
        if not persona_loading and not persona_busy:
            if self.loaded and vision is not None and not persona_ready:
                start_persona_background_load(self)
            else:
                return True
        waiter.wait(0.05)
    return False


def interrupt_active_agent_for_input(self, text: str) -> bool:
    """Invalidate active agent work before accepting replacement user input."""
    del text
    task = self.agent_runtime.get_active_task("default")
    if task is None or task.state in {
        AgentTaskState.IDLE,
        AgentTaskState.AWAITING_APPROVAL,
        AgentTaskState.AWAITING_CHOICE,
        AgentTaskState.PAUSED,
        AgentTaskState.INTERRUPTED,
    }:
        return False
    try:
        self.agent_runtime.interrupt_task(
            task.task_id,
            AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
        )
    except ValueError:
        return False
    return True


def classify_input(
    self,
    text: str,
    *,
    persona_ready: Optional[bool] = None,
) -> AgentClassificationResult:
    """Classify one input without creating a task or starting execution.

    Args:
        text: The text or transcription to classify.
        persona_ready: Whether the loaded Persona model may resolve ambiguity.

    Returns:
        AgentClassificationResult: The typed conversation-first classification.
    """
    ready = self.persona_ready if persona_ready is None else persona_ready
    return self.agent_router.classify(text, persona_ready=ready)


def route_input(
    self,
    text: str,
    *,
    persona_ready: Optional[bool] = None,
) -> AgentClassificationResult:
    """Route one input through the active task or ordinary Persona conversation.

    Args:
        text: The text or transcription to route.
        persona_ready: Whether the loaded Persona model may resolve ambiguity.

    Returns:
        AgentClassificationResult: The typed route selected for the input.
    """
    if self.test_finished:
        raise RuntimeError("Celune test mode has finished")
    ready = self.persona_ready if persona_ready is None else persona_ready
    result = self.agent_router.route(text, persona_ready=ready)
    _log_route_decision(self, text, result)
    return result


def _log_route_decision(
    self,
    request: str,
    result: AgentClassificationResult,
) -> None:
    """Log the semantic route selected before downstream processing begins."""
    if result.failure is not None:
        route_type = "say"
    elif result.classification == AgentInputClassification.TASK:
        route_type = "agent"
    else:
        route_type = "persona"

    request_value = request.replace("\\", "\\\\").replace("\r", "\\r")
    request_value = request_value.replace("\n", "\\n")
    fields = [f"request={request_value}", f"type={route_type}"]
    if route_type == "agent" and result.intent is not None:
        fields.append(f"intent={result.intent}")
    if result.failure is None:
        fields.append(f"confidence={round(result.confidence * 100):.0f}%")
    self.log(f"[ROUTE] {' '.join(fields)}", loglevel="debug")


def agent_needle_ready(self) -> bool:
    """Return whether the production Needle selector is loaded and usable."""
    return self.agent_needle_selector is not None


def agent_needle_error(self) -> Optional[str]:
    """Return the latest production Needle loading failure, if any."""
    return self._agent_needle_error


def _load_agent_tool_selector(self) -> AgentToolSelector:
    """Load the verified Needle selector only when an agent task needs it."""
    if self.agent_needle_selector is not None:
        return self.agent_needle_selector
    try:
        with huggingface_progress(self.progress_callback):
            selector = NeedleToolSelector.from_pretrained(
                self.agent_tools,
                schemas=self.agent_tool_schemas,
            )
    except (ModelConsentDeclined, ModelConsentRequired):
        raise
    except Exception as exc:
        self._agent_needle_error = str(exc)
        self.log(
            format_error_message(
                string("agent.needle_loading_failed"),
                exc,
                self.log_level,
            ),
            "error",
            loglevel="verbose",
        )
        raise NeedleSelectionError(
            "Needle selector is unavailable for the agent runtime"
        ) from exc
    self.agent_needle_selector = selector
    self._agent_needle_error = None
    return selector


def select_agent_tool(
    self,
    context: AgentContext,
    output: AgentOutput,
) -> Optional[Union[ToolCall, Sequence[ToolCall]]]:
    """Select and validate one registered tool through the Needle boundary."""
    from ..agent.tools import recover_local_tool_call

    selection = _load_agent_tool_selector(self)(context, output)
    if isinstance(selection, Sequence):
        return [recover_local_tool_call(call, context) for call in selection]
    return (
        recover_local_tool_call(selection, context) if selection is not None else None
    )


def execute_agent_tool(
    self,
    _context: AgentContext,
    call: ToolCall,
) -> ToolExecutionResult:
    """Execute one allowlisted production tool through its typed boundary."""
    tool = next(
        (candidate for candidate in self.agent_tools if candidate.name == call["name"]),
        None,
    )
    if tool is None:
        return {
            "tool_call_id": call["id"],
            "output": None,
            "error": "agent tool is not registered",
            "tool_id": call["name"],
            "status": AgentToolExecutionStatus.FAILED,
        }
    try:
        return cast(ToolExecutionResult, tool.execute(call, _context))
    except Exception as exc:
        self.log(
            format_error_message(
                f"[AGENT] tool_failed tool={call['name']}",
                exc,
                self.log_level,
            ),
            "error",
            loglevel="verbose",
        )
        return {
            "tool_call_id": call["id"],
            "output": None,
            "error": str(exc),
            "tool_id": call["name"],
            "status": AgentToolExecutionStatus.FAILED,
        }


def run_agent_route(self, route: AgentClassificationResult) -> bool:
    """Consume a routed task through the shared agent runtime."""
    if self.test_finished:
        return False
    request = route.task_request
    metadata = route.routing_metadata
    task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
    if request is None:
        task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
        if not isinstance(task_id, str):
            return False
        request = self.agent_runtime.get_task(task_id).request
    if not isinstance(task_id, str):
        active_task = self.agent_runtime.get_active_task(request.session.session_id)
        task_id = active_task.task_id if active_task is not None else None
    delivery_failed = False
    initial_task = (
        self.agent_runtime.get_task(task_id) if isinstance(task_id, str) else None
    )
    if initial_task is None:
        initial_task = self.agent_runtime.create_task(request)
        task_id = initial_task.task_id
    opening_pending = initial_task.state == AgentTaskState.IDLE
    suppress_initial_plan = opening_pending
    if opening_pending:
        if not _wait_for_persona_initialization(self):
            return False
        try:
            self.agent_runtime.start_task(initial_task.task_id)
        except ValueError:
            opening_pending = False
    if opening_pending:
        opening_preamble = None
        if isinstance(task_id, str):
            try:
                opening_context = self.agent_runtime.get_context(task_id)
                opening = self._agent_persona_bridge.opening(opening_context)
                generated_preamble = opening.get("response")
                if (
                    not isinstance(generated_preamble, str)
                    or not _is_plain_agent_preamble(generated_preamble)
                ) and not opening.get("paused"):
                    opening = self._agent_persona_bridge.opening(
                        opening_context,
                        retry=True,
                    )
                    generated_preamble = opening.get("response")
                if isinstance(generated_preamble, str) and _is_plain_agent_preamble(
                    generated_preamble
                ):
                    opening_preamble = generated_preamble
            except Exception as exc:
                self.log(
                    format_error_message(
                        "[AGENT] opening_response_generation_failed",
                        exc,
                        self.log_level,
                    ),
                    "warning",
                    loglevel="verbose",
                )
        if opening_preamble is not None:
            try:
                if self.say(opening_preamble, maintain_status=True):
                    wait_for_persona_playback(self)
            except Exception as exc:
                self.log(
                    format_error_message(
                        "[AGENT] opening_speech_failed",
                        exc,
                        self.log_level,
                    ),
                    "warning",
                    loglevel="verbose",
                )

    def deliver_output(output: AgentOutput) -> None:
        """Deliver generated agent responses through the shared speech path."""
        nonlocal delivery_failed, suppress_initial_plan
        terminal = output.get("terminal")
        if suppress_initial_plan and terminal is None:
            suppress_initial_plan = False
            return
        if output.get("tool_call") is not None:
            return
        if terminal is not None:
            if terminal.state == AgentTaskState.COMPLETED:
                return
            if not wait_for_persona_playback(self):
                delivery_failed = True
                return
            if task_id is None:
                delivery_failed = True
                raise RuntimeError("agent terminal output has no task context")
            try:
                context = self.agent_runtime.get_context(task_id)
                response_output = self._agent_persona_bridge.respond(context)
                response = response_output.get("response")
                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError("agent failure response was empty")
                if not deliver_persona_response(
                    self, request.request, response, maintain_status=True
                ):
                    raise RuntimeError("agent failure response could not be queued")
            except Exception as exc:
                self.log(
                    format_error_message(
                        "[AGENT] failure_response_generation_failed",
                        exc,
                        self.log_level,
                    ),
                    "warning",
                    loglevel="verbose",
                )
                fallback = string("agent.failure_final")
                if terminal.state == AgentTaskState.CANCELLED:
                    fallback = string("agent.cancelled_final")
                elif (
                    terminal.state == AgentTaskState.ABORTED
                    and terminal.abort_reason == AgentAbortReason.STUCK_TASK
                ):
                    fallback = string("agent.stuck_final")
                elif terminal.state == AgentTaskState.ABORTED:
                    fallback = string("agent.limit_final")
                if not self.say(fallback, maintain_status=True):
                    delivery_failed = True
            return
        if self.backend_mode == "agent_test" and not output.get("end"):
            return
        response = output.get("response")
        if not isinstance(response, str) or not response.strip():
            return
        if self.cur_state in {
            "generating",
            "speaking",
        } and not wait_for_persona_playback(self):
            delivery_failed = True
            raise RuntimeError("agent response speech was interrupted")
        if not deliver_persona_response(
            self, request.request, response, maintain_status=True
        ):
            delivery_failed = True
            raise RuntimeError("agent response speech could not be queued")

    output = self.agent_runtime.run(request, callback=deliver_output)
    if output["paused"] and isinstance(task_id, str):
        task = self.agent_runtime.get_task(task_id)
        if task.state == AgentTaskState.AWAITING_APPROVAL:
            approval = self.agent_runtime.get_pending_approval(task_id)
            if approval is not None:
                approval_text = approval.prompt
                try:
                    approval_output = self._agent_persona_bridge.approval(
                        self.agent_runtime.get_context(task_id),
                        approval,
                    )
                    generated_text = approval_output.get("response")
                    if isinstance(generated_text, str) and generated_text.strip():
                        approval_text = generated_text
                except Exception as exc:
                    self.log(
                        format_error_message(
                            "[AGENT] approval_response_generation_failed",
                            exc,
                            self.log_level,
                        ),
                        "warning",
                        loglevel="verbose",
                    )
                if not deliver_persona_response(
                    self,
                    request.request,
                    approval_text,
                    maintain_status=True,
                ):
                    self.log(
                        "[AGENT] approval_response_speech_failed",
                        "warning",
                        loglevel="verbose",
                    )
    if output["end"] and isinstance(task_id, str):
        task = self.agent_runtime.get_task(task_id)
        if (
            task.is_terminal
            and self.playback_done.is_set()
            and not self.locked
            and self.cur_state
            not in {
                "idle",
                "error",
                "reloading",
                "exiting",
                "sleeping",
                "stopped",
                "waking",
            }
        ):
            self.cur_state = "idle"
            self.idle_callback()
    return output["end"] and not delivery_failed


def speak_agent_classification_failure(
    self,
    request: str,
    failure: AgentClassificationFailure,
    route: AgentClassificationResult,
) -> bool:
    """Ask the active Persona to explain a classifier failure naturally."""
    metadata = route.routing_metadata
    task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
    if isinstance(task_id, str):
        try:
            context = self.agent_runtime.get_context(task_id)
        except ValueError:
            context = None
    else:
        context = None
    if context is None:
        context = self.agent_runtime.create_context(
            AgentRequest(
                request=request,
                session=AgentSession(session_id="default"),
            ),
            classification_failure=failure,
        )
    try:
        output = self._agent_persona_bridge.respond(context)
        response = output.get("response")
        if isinstance(response, str) and response.strip():
            return deliver_persona_response(
                self, request, response, maintain_status=True
            )
    except Exception as exc:
        self.log(
            format_error_message(
                "[AGENT] failure_response_generation_failed",
                exc,
                self.log_level,
            ),
            "warning",
            loglevel="verbose",
        )
    return self.say(string("agent.classifier_unavailable"), maintain_status=True)
