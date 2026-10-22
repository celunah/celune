# SPDX-License-Identifier: Apache-2.0
"""Engine orchestration for Persona conversation and speech delivery."""

from __future__ import annotations
import queue
import threading
from uuid import uuid4
from typing import TYPE_CHECKING

from ..i18n import string
from ..utils import format_error_message
from ..threads import run_in_daemon_thread
from ..pipeline.submission import say, queue_speech
from ..pipeline.buffering import notify_component_busy
from ..persona.impl import persona_enabled, compact_persona_history
from ..typing.agent import AgentRoute, AgentClassificationFailureKind
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..persona.conversation import (
    extract_persona_text,
    build_persona_request,
    store_persona_memories,
    classify_persona_memories,
)

if TYPE_CHECKING:
    from ..celune import Celune


def think(engine: Celune, text: str) -> bool:
    """Queue one text input for Persona or the active agent route."""
    if engine.test_finished or engine.backend_mode == "agent_test":
        return False
    if not persona_enabled(engine.config):
        return say(engine, text)
    if engine.input_mode != "text_to_speech":
        engine.log(string("celune.text_input_unavailable_vc"), "warning")
        engine.error_callback(string("celune.not_possible"))
        return False

    if engine.is_in_tutorial:
        engine.log(string("celune.speech_input_disabled_tutorial"), "warning")
        return False

    from . import agent as core_agent

    core_agent.interrupt_active_agent_for_input(engine, text)
    with engine.say_lock:
        engine._persona_queue.put(text)
        thread = engine._persona_thread
        if thread is not None and thread.is_alive():
            return True

        engine.status_callback(string("status.thinking"))
        engine.progress_callback(None, None)
        engine._ready_announced = False
        thread = threading.Thread(target=_think_worker, args=(engine,), daemon=True)
        engine._persona_thread = thread
        thread.start()
    return True


async def think_async(engine: Celune, text: str) -> bool:
    """Queue one Persona input without blocking an async caller."""
    return await run_in_daemon_thread(think, engine, text)


def deliver_persona_response(engine: Celune, request: str, response: str) -> bool:
    """Record and queue one response generated through the agent Persona bridge."""
    spoken_text = response.strip()
    if not spoken_text:
        return False

    store_persona_memories(engine, request)
    history = getattr(engine, "persona_history", None)
    if isinstance(history, list):
        history.extend(
            [
                {"role": "user", "content": request.strip()},
                {"role": "assistant", "content": spoken_text},
            ]
        )

    queued = queue_speech(
        engine,
        spoken_text,
        save=False,
        display_text=spoken_text,
    )
    if isinstance(history, list):
        compact_persona_history(engine)
    classify_persona_memories(engine, request)
    return queued


def think_persona(engine: Celune, request: str) -> bool:
    """Generate, record, and speak one ordinary Persona response."""
    store_persona_memories(engine, request)
    payload = build_persona_request(engine, request)
    attachments = getattr(engine, "persona_attachments", None)
    lease = None
    manager = getattr(engine, "component_locks", None)
    if manager is not None:
        acquisition, lease = manager.try_acquire_lease(
            (ComponentLockRequirement(ComponentLockName.VLM),),
            ComponentLockOwner(operation_id=f"persona:{uuid4().hex}"),
        )
        if not acquisition.acquired:
            busy = acquisition.busy
            assert busy is not None
            notify_component_busy(engine, "think", busy)
            if isinstance(attachments, list):
                attachments.clear()
            return False

    try:
        vision = engine.vision
        if vision is None:
            engine.log(string("pipeline.persona_not_connected"), "warning")
            return False

        response = vision.post(json=payload)
        response.raise_for_status()
        spoken_text = extract_persona_text(response.json())
    except Exception as error:
        engine.log(
            format_error_message(
                string("pipeline.persona_request_failed"),
                error,
                engine.log_level,
            ),
            "warning",
        )
        return False
    finally:
        if lease is not None:
            lease.release()
        if isinstance(attachments, list):
            attachments.clear()

    if not spoken_text:
        engine.log(string("pipeline.persona_empty_response"), "warning")
        return False

    history = getattr(engine, "persona_history", None)
    if isinstance(history, list):
        history.extend(
            [
                {"role": "user", "content": request.strip()},
                {"role": "assistant", "content": spoken_text},
            ]
        )

    queued = queue_speech(
        engine,
        spoken_text,
        save=False,
        display_text=spoken_text,
    )
    if isinstance(history, list):
        compact_persona_history(engine)
    classify_persona_memories(engine, request)
    return queued


def _think_worker(engine: Celune) -> None:
    """Fetch queued Persona responses without blocking Celune's UI thread."""
    from . import agent as core_agent
    from .lifecycle import persona_conn, start_persona_background_load

    current_thread = threading.current_thread()
    try:
        while not engine.exit_requested:
            try:
                text = engine._persona_queue.get(timeout=0.1)
            except queue.Empty:
                with engine.say_lock:
                    if not engine._persona_queue.empty():
                        continue
                    if engine._persona_thread is current_thread:
                        engine._persona_thread = None
                    return

            if not wait_for_persona_playback(engine):
                return

            engine.status_callback(string("status.thinking"))
            engine.cur_state = "thinking"
            engine.progress_callback(None, None)

            with engine._model_lock:
                persona_loading = engine.persona_loading
                persona_ready = engine.persona_ready
                vision = engine.vision

            route = core_agent.route_input(engine, text, persona_ready=persona_ready)
            if route.failure is not None:
                can_prepare_persona = (
                    route.failure.kind
                    == AgentClassificationFailureKind.PERSONA_UNAVAILABLE
                    and not persona_ready
                )
                if not can_prepare_persona:
                    core_agent.speak_agent_classification_failure(
                        engine, text, route.failure, route
                    )
                    continue
            if route.route == AgentRoute.CLARIFICATION:
                clarification = route.clarification_prompt
                if clarification:
                    say(engine, clarification)
                continue
            if route.route != AgentRoute.CONVERSATION:
                core_agent.run_agent_route(engine, route)
                continue

            if persona_loading:
                say(engine, text)
                continue

            if not persona_ready:
                if vision is None:
                    with engine._model_lock:
                        if engine.vision is None:
                            engine.vision = persona_conn(engine)
                        vision = engine.vision
                if vision is None:
                    say(engine, text)
                    continue
                start_persona_background_load(engine)
                say(engine, text)
                continue

            if not think_persona(engine, text):
                engine.log(string("celune.say_instead"), "warning")
                say(engine, text)
    finally:
        with engine.say_lock:
            if engine._persona_thread is current_thread:
                engine._persona_thread = None


def wait_for_persona_playback(engine: Celune) -> bool:
    """Wait until the shared speech pipeline is available for another Persona turn."""
    while not engine.exit_requested:
        engine.playback_done.wait(timeout=0.1)
        with engine.say_lock:
            if engine._reload_pending or engine.cur_state == "reloading":
                continue
            if not engine.locked and engine.cur_state not in {"generating", "speaking"}:
                return True
    return False
