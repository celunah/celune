# SPDX-License-Identifier: Apache-2.0
"""Agent routing and tool execution methods."""

from __future__ import annotations

from typing import Optional, cast

from .needle import NeedleToolSelector
from ..exceptions import NeedleSelectionError
from ..i18n import string
from ..paths import huggingface_progress
from ..speech import deliver_persona_response
from ..typing.agent import (
    AgentAbortReason,
    AgentClassificationFailure,
    AgentClassificationResult,
    AgentContext,
    AgentInputClassification,
    AgentInterruption,
    AgentInterruptionKind,
    AgentOutput,
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentToolExecutionStatus,
    AgentToolSelector,
    ToolCall,
    ToolExecutionResult,
)
from ..utils import format_error_message
from ..binding import install_class_functions

__all__ = (
    "_execute_agent_tool",
    "_interrupt_active_agent_for_input",
    "_load_agent_tool_selector",
    "_log_route_decision",
    "_run_agent_route",
    "_select_agent_tool",
    "_speak_agent_classification_failure",
    "agent_needle_error",
    "agent_needle_ready",
    "classify_input",
    "route_input",
)


def _interrupt_active_agent_for_input(self, text: str) -> bool:
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
    return self._agent_router.classify(text, persona_ready=ready)


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
    result = self._agent_router.route(text, persona_ready=ready)
    self._log_route_decision(text, result)
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
    return self._agent_needle_selector is not None


def agent_needle_error(self) -> Optional[str]:
    """Return the latest production Needle loading failure, if any."""
    return self._agent_needle_error


def _load_agent_tool_selector(self) -> AgentToolSelector:
    """Load the verified Needle selector only when an agent task needs it."""
    if self._agent_needle_selector is not None:
        return self._agent_needle_selector
    try:
        with huggingface_progress(self.progress_callback):
            selector = NeedleToolSelector.from_pretrained(
                self._agent_tools,
                schemas=self._agent_tool_schemas,
            )
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
    self._agent_needle_selector = selector
    self._agent_needle_error = None
    return selector


def _select_agent_tool(
    self,
    context: AgentContext,
    output: AgentOutput,
) -> Optional[ToolCall]:
    """Select and validate one registered tool through the Needle boundary."""
    return self._load_agent_tool_selector()(context, output)


def _execute_agent_tool(
    self,
    _context: AgentContext,
    call: ToolCall,
) -> ToolExecutionResult:
    """Execute one allowlisted production tool through its typed boundary."""
    tool = next(
        (
            candidate
            for candidate in self._agent_tools
            if candidate.name == call["name"]
        ),
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


def _run_agent_route(self, route: AgentClassificationResult) -> bool:
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

    def deliver_output(output: AgentOutput) -> None:
        """Deliver generated agent responses through the shared speech path."""
        nonlocal delivery_failed
        if output.get("tool_call") is not None:
            return
        terminal = output.get("terminal")
        if terminal is not None:
            if terminal.state == AgentTaskState.COMPLETED:
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
                if not deliver_persona_response(self, request.request, response):
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
                if not self.say(fallback):
                    delivery_failed = True
            return
        if self.backend_mode == "agent_test" and not output.get("end"):
            return
        response = output.get("response")
        if not isinstance(response, str) or not response.strip():
            return
        if (
            self.cur_state in {"generating", "speaking"}
            and not self._wait_for_persona_playback()
        ):
            delivery_failed = True
            raise RuntimeError("agent response speech was interrupted")
        if not deliver_persona_response(self, request.request, response):
            delivery_failed = True
            raise RuntimeError("agent response speech could not be queued")

    output = self.agent_runtime.run(request, callback=deliver_output)
    return output["end"] and not delivery_failed


def _speak_agent_classification_failure(
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
            return deliver_persona_response(self, request, response)
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
    return self.say(string("agent.classifier_unavailable"))


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(
        target,
        {name: globals()[name] for name in __all__},
        properties=("agent_needle_ready", "agent_needle_error"),
    )
