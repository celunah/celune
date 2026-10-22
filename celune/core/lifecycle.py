# SPDX-License-Identifier: Apache-2.0
"""Celune engine construction and lifecycle coordination."""

from __future__ import annotations

import gc
import os
import time
import queue
import shutil
import asyncio
import threading
import contextlib
from pathlib import Path
from functools import partial
from typing import Union, Optional, cast
from collections.abc import Mapping, Callable, Generator

import torch

from .callbacks import _buffer_startup_log

from ..chroma import AudioRGBGlow
from ..exceptions import BackendError
from ..typing.aliases import LogLevel
from ..locks import ComponentLockLease
from ..typing.modes import BackendMode
from ..agent.runtime import AgentRuntime
from ..agent.routing import AgentInputRouter
from ..agent.needle import NeedleToolSelector
from ..agent.persona import PersonaAgentBridge
from ..dataclasses.events import ShutdownEvent
from ..extensions.events import EventDispatcher
from ..paths import api_key_path, temp_data_dir
from ..backends.tts import BACKENDS, CeluneBackend
from ..typing.common import JSON, JSONSerializable
from ..persona.emotion import PersonaEmotionAnalyzer
from ..backends.vc import VC_BACKENDS, CeluneVCBackend
from ..config import (
    config_int as _config_int,
)
from ..i18n import string, set_locale, get_system_locale
from ..modes import resolve_operation_mode
from ..utils import discard, available, clear_queue, format_error_message
from ..vram import backend_allowed, resolve_vram_preset, resolve_backend_name
from ..typing.agent import AgentTaskConfig, AgentToolSelector, AgentCancellationReason
from ..constants import APP_NAME, AGENT_MAX_LOOPS, AGENT_COMPACT_AT, AGENT_CONTEXT_SPACE
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..cevoice import (
    default_loader,
    select_voice_bundle,
    close_default_loader,
    is_protected_temp_path,
)
from ..agent.tools import (
    agent_test_tools,
    production_agent_tools,
    agent_test_tool_schemas,
    production_agent_tool_schemas,
)
from ..config import (
    Config,
    config_bool,
    config_value,
    config_api_settings,
    normalize_log_level,
    config_backend_options,
)
from ..persona.impl import (
    PersonaClient,
    persona_enabled,
    persona_model_id,
    persona_is_available,
    persona_quantization,
    create_persona_client,
)
from ..dataclasses.celune import (
    CeluneAudioState,
    CeluneModelState,
    CeluneVoiceState,
    CeluneBackendState,
    CeluneRuntimeState,
    CeluneCallbackState,
    CelunePipelineState,
)
from ..typing.celune import (
    VCBackendSpec,
    TTSBackendSpec,
    CaptionCallback,
    CoreBackendSpec,
    MessageCallback,
    ProgressCallback,
    InputStateCallback,
    VoiceLockStateCallback,
    CaptionProgressCallback,
)

TTS_KV_CACHE_BACKENDS = frozenset({"qwen3", "fireredtts3", "voxcpm2"})


def initialize(
    self,
    config: Config,
    backend: Optional[CoreBackendSpec] = None,
    tts_backend: Optional[CoreBackendSpec] = None,
    vc_backend: Optional[CoreBackendSpec] = None,
    vc_f0_condition: Optional[bool] = None,
    input_mode: Optional[str] = None,
    chunk_size: int = 0,  # defaulted to 0 because not all backends use this
    target_chunk_length: float = 0.64,
    language: str = "Auto",  # Qwen3 backend accepts a language, others may not
    log_callback: Optional[MessageCallback] = None,
    status_callback: Optional[MessageCallback] = None,
    error_callback: Optional[Callable[[str], None]] = None,
    idle_callback: Optional[Callable[[], None]] = None,
    queue_avail_callback: Optional[Callable[[], None]] = None,
    voice_changed_callback: Optional[Callable[[str], None]] = None,
    change_input_state_callback: Optional[InputStateCallback] = None,
    change_voice_lock_state_callback: Optional[VoiceLockStateCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
    caption_progress_callback: Optional[CaptionProgressCallback] = None,
    caption_callback: Optional[CaptionCallback] = None,
    log_level: LogLevel = "info",
    agent_tool_selector: Optional[AgentToolSelector] = None,
    backend_mode: BackendMode = "normal",
    startup_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Create the Celune engine and retain its deferred startup callback.

    The optional ``startup_callback`` receives the initialization-core
    checkpoint when :meth:`load` begins.
    """
    if type(self)._instance is not None:
        raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")
    if backend_mode not in {"normal", "ui_test", "agent_test"}:
        raise ValueError(f"unknown Celune backend mode: '{backend_mode}'")

    self._startup_log_buffer = []
    self._startup_banner_emitted = False
    self._startup_log_sink = log_callback or _noop_message
    self._startup_callback = startup_callback
    self._callbacks = CeluneCallbackState(
        log_callback=partial(_buffer_startup_log, self),
        status_callback=status_callback or _noop_message,
        error_callback=error_callback or (lambda error: None),
        idle_callback=idle_callback or (lambda: None),
        queue_avail_callback=queue_avail_callback or (lambda: None),
        voice_changed_callback=voice_changed_callback or (lambda name: None),
        change_input_state_callback=(change_input_state_callback or _noop_input_state),
        change_voice_lock_state_callback=(
            change_voice_lock_state_callback or _noop_voice_lock_state
        ),
        progress_callback=progress_callback or _noop_progress,
        caption_progress_callback=(caption_progress_callback or _noop_progress),
        caption_callback=caption_callback or _noop_caption,
    )
    self._event_dispatcher = EventDispatcher(
        log_warning=self.log,
        log_level=normalize_log_level(log_level),
        log_debug=lambda message: self.log(message, loglevel="debug"),
    )

    self._backend_state = CeluneBackendState(
        config=config,
        log_level=normalize_log_level(log_level),
    )
    self._model_state = CeluneModelState()
    self._voice_state = CeluneVoiceState()
    self._pipeline_state = CelunePipelineState(audio_queue=queue.Queue(maxsize=8))
    self._audio_state = CeluneAudioState()
    self._runtime_state = CeluneRuntimeState()
    self._reload_backend = None
    self._normalizer_load_epoch = 0
    self._async_runtime_lock = threading.Lock()
    self._voice_reload_guard = threading.Lock()
    self._voice_reload_active = False
    self._pipeline_state.model_ready.set()
    self._pipeline_state.playback_done.set()

    self.config = config
    self._api_tunnel = None
    self._runtime_state.backend_mode = backend_mode
    set_locale(_configured_locale(config) or get_system_locale())
    self.mode = resolve_operation_mode(config)
    if backend_mode == "agent_test":
        self.agent_tools = agent_test_tools(self)
        self.agent_tool_schemas = agent_test_tool_schemas()
    else:
        agent_config = config.get("agent")
        local_management = (
            config_bool(
                agent_config,
                "CELUNE_AGENT_FS_TOOLS",
                "fs_tools",
            )
            if isinstance(agent_config, dict)
            else False
        )
        self.agent_tools = production_agent_tools(
            self,
            include_local_management=local_management,
        )
        self.agent_tool_schemas = production_agent_tool_schemas(
            include_local_management=local_management,
        )
        if local_management and config.get("mode") == "agent":
            self.log(
                "\n".join(
                    (
                        string("agent.unsandboxed_title"),
                        string("agent.unsandboxed_body", app_name=APP_NAME),
                        string("agent.unsandboxed_rollback", app_name=APP_NAME),
                    )
                ),
                "warning",
            )
    self.agent_needle_selector = agent_tool_selector
    self._agent_needle_error = None
    self._agent_persona_bridge = PersonaAgentBridge(
        self,
        self.agent_tool_schemas,
    )
    self.input_mode = _resolve_input_mode(config, input_mode)
    from .agent import select_agent_tool, execute_agent_tool

    self.agent_runtime = AgentRuntime(
        tools=self.agent_tools,
        event_dispatcher=self._event_dispatcher,
        celune=self,
        planner=self._agent_persona_bridge.plan,
        tool_selector=lambda context, output: select_agent_tool(self, context, output),
        tool_executor=lambda context, call: execute_agent_tool(self, context, call),
        tool_result_handler=self._agent_persona_bridge.handle_tool_result,
        responder=self._agent_persona_bridge.respond,
        compactor=self._agent_persona_bridge.compact,
        tool_schemas=self.agent_tool_schemas,
        task_config=agent_task_config(config),
    )
    self.agent_router = AgentInputRouter(self, self.agent_runtime)
    glow_color = "#cebaff"
    loader = default_loader()
    if loader is not None:
        theme = loader.bundle.metadata.get("theme")
        if isinstance(theme, dict):
            accent = theme.get("accent")
            if isinstance(accent, str):
                glow_color = accent

            configured_glow = theme.get("glow_color")
            if isinstance(configured_glow, str):
                glow_color = configured_glow

    self.glow = AudioRGBGlow(celune=self, color=glow_color)
    _wrap_fatal_glow(self)
    self.glow.start()

    tts_backend, vc_backend = _resolve_core_backend_specs(
        self.log_callback,
        self.input_mode,
        backend,
        tts_backend,
        vc_backend,
    )
    self.vc_f0_condition = False if vc_f0_condition is None else vc_f0_condition
    select_voice_bundle(config_str(config_value(config, "voice_bundle")))
    preset = resolve_vram_preset(config)

    if tts_backend is None:
        tts_backend = preset.default_backend

    backend_kwargs: dict[str, JSONSerializable] = {
        "quantize": tts_quantization_enabled(config, tts_backend),
    }
    if isinstance(tts_backend, CeluneBackend):
        tts_backend.quantization_requested = bool(backend_kwargs["quantize"])
    if isinstance(tts_backend, CeluneBackend):
        if not backend_allowed(config, tts_backend.name):
            raise BackendError(
                f"backend '{tts_backend.name}' is not available for VRAM tier '{preset.tier}'"
            )
    elif isinstance(tts_backend, type) and issubclass(tts_backend, CeluneBackend):
        backend_type_name = getattr(tts_backend, "name", "").strip().lower()
        if not backend_allowed(config, backend_type_name):
            raise BackendError(
                f"backend '{backend_type_name}' is not available for VRAM tier '{preset.tier}'"
            )

    if isinstance(tts_backend, str):
        requested_backend = tts_backend
        tts_backend = resolve_backend_name(config, tts_backend)
        if tts_backend != requested_backend.strip().lower():
            self.log(
                (
                    f"Backend '{requested_backend}' is not available for VRAM tier "
                    f"'{preset.tier}', using '{tts_backend}' instead."
                ),
                "warning",
            )

    # please for the love of god do not import backend-specific deps here
    raw_name = getattr(tts_backend, "name", None)
    backend_name = (
        tts_backend.strip().lower()
        if isinstance(tts_backend, str)
        else raw_name
        if isinstance(raw_name, str)
        else None
    )
    quantize_kv_cache = config_value(config, "quantize_kv_cache", True) is not False
    if isinstance(tts_backend, CeluneBackend):
        tts_backend.quantize_kv_cache = quantize_kv_cache
    if backend_name in TTS_KV_CACHE_BACKENDS:
        backend_kwargs["quantize_kv_cache"] = quantize_kv_cache
    if backend_name == "qwen3":
        backend_kwargs["x_vector_only"] = config_bool(
            config,
            "CELUNE_QWEN3_X_VECTOR_ONLY",
            "qwen3_x_vector_only",
        )
        backend_kwargs["clone_model_id"] = preset.qwen3_clone_model_id
    backend_kwargs.update(config_backend_options(config, backend_name))

    try:
        resolved_tts_backend = cast(TTSBackendSpec, tts_backend)
        if not isinstance(tts_backend, CeluneBackend):
            self._backend_spec = cast(
                Union[str, type[CeluneBackend]],
                resolved_tts_backend,
            )
            self._backend_kwargs = dict(backend_kwargs)
        from .reload import resolve_tts_backend, validate_backend_against_preset

        self.backend = resolve_tts_backend(
            self,
            resolved_tts_backend,
            log=self.log_callback,
            fatal=self.fatal,
            **backend_kwargs,
        )
        self.backend.bind_fatal(self.fatal)
        validate_backend_against_preset(self.backend, preset)
        self.tts_backend = self.backend.name
    except ValueError as e:
        raise BackendError(str(e)) from e
    except TypeError as e:
        raise BackendError(f"invalid backend specification: '{tts_backend}'") from e
    except ModuleNotFoundError as e:
        raise BackendError(
            f"backend '{tts_backend}' has unmet dependencies: '{e.name}'"
        ) from e
    except FileNotFoundError as e:
        raise BackendError(
            "you must install at least one valid CEVOICE/CECHAR package"
        ) from e
    except Exception as e:
        raise BackendError(
            format_error_message(
                string("celune.internal_backend_error"),
                e,
                log_level,
            )
        ) from e

    if vc_backend is None and self.input_mode == "voice_conversion":
        vc_backend = config_str(
            config_value(config, "voice_conversion_backend")
        ) or config_str(config_value(config, "vc_backend"))
        if vc_backend is None:
            vc_backend = "seed-vc"

    try:
        if vc_backend is not None:
            resolved_vc_backend = cast(VCBackendSpec, vc_backend)
            if not isinstance(vc_backend, CeluneVCBackend):
                self._vc_backend_spec = cast(
                    Union[str, type[CeluneVCBackend]],
                    resolved_vc_backend,
                )
                from .reload import resolve_vc_backend

                self.vc_backend = resolve_vc_backend(
                    self,
                    resolved_vc_backend,
                    log=self.log_callback,
                )
            if available("f0_condition", obj=self.vc_backend):
                self.vc_backend.f0_condition = self.vc_f0_condition
            self.voice_conversion_backend = self.vc_backend.name
        else:
            self.vc_backend = None
            self.voice_conversion_backend = ""
    except ValueError as e:
        raise BackendError(str(e)) from e
    except TypeError as e:
        raise BackendError(
            f"invalid voice-conversion backend specification: '{vc_backend}'"
        ) from e
    except ModuleNotFoundError as e:
        raise BackendError(
            f"voice-conversion backend '{vc_backend}' has unmet dependencies: '{e.name}'"
        ) from e
    except Exception as e:
        raise BackendError(
            format_error_message(
                string("celune.internal_vc_backend_error"),
                e,
                log_level,
            )
        ) from e

    if chunk_size:
        self.chunk_size = chunk_size
    else:
        # chunk length must be evenly divisible by target backend's base chunk size
        # e.g. if chunk rate = 12.5, then chunk length must be evenly divisible by 0.08s
        #
        # examples:
        # Qwen3 = length must be divisible by 0.08s (12.5 Hz)
        # VoxCPM2 = length must be divisible by 0.16s (6.25 Hz)
        multiple = target_chunk_length * self.backend.chunk_rate
        nearest = round(multiple)

        if abs(multiple - nearest) > 1e-6:
            raise BackendError(
                f"invalid chunk length: {target_chunk_length}s is not divisible by {1 / self.backend.chunk_rate}s"
            )

        self.chunk_size = max(
            1, round(target_chunk_length / (1 / self.backend.chunk_rate))
        )

    self.language = language
    self.log_level = normalize_log_level(log_level)
    self.use_normalization = config_bool(config, "CELUNE_NORMALIZE", "use_normalizer")

    self.vision = persona_conn(self)

    from ..registry.consent import set_commercial_mode

    set_commercial_mode(config.get("commercial") is True)

    type(self)._instance = self


def _noop_message(
    msg: str,
    severity: str = "info",
    *,
    loglevel: LogLevel = "info",
) -> None:
    """Discard a message callback."""
    del msg, severity, loglevel


def _noop_input_state(locked: bool) -> None:
    """Discard an input lock-state callback."""
    del locked


def _noop_voice_lock_state(locked: bool) -> None:
    """Discard a voice lock-state callback."""
    del locked


def _noop_progress(progress: Optional[float], total: Optional[float]) -> None:
    """Discard a progress callback."""
    del progress, total


def _noop_caption(caption: Optional[str]) -> None:
    """Discard a speech caption update."""
    del caption


def finish_test_mode(
    self,
    mode: str,
    success: bool,
    *,
    task_state: Optional[str] = None,
    detail: Optional[str] = None,
    checks: Optional[list[JSON]] = None,
) -> JSON:
    """Finish an explicit test mode and leave the engine stopped but alive.

    Args:
        mode: The selected test mode name.
        success: Whether the controlled test completed successfully.
        task_state: Final agent task state, when the mode created a task.
        detail: Optional diagnostic detail retained with the final result.
        checks: Optional per-feature agent diagnostic results.

    Returns:
        JSON: The synchronously recorded final test result.
    """
    if self.test_result is not None:
        return self.test_result

    self._runtime_state.test_finished = True
    self.locked = True
    cleanup_errors: list[str] = []
    for callback in (
        lambda: self.change_input_state_callback(locked=True),
        lambda: self.change_voice_lock_state_callback(locked=True),
        self.stop_live_audio,
    ):
        try:
            callback()
        except Exception as exc:
            cleanup_errors.append(
                format_error_message(
                    string("test.cleanup_failed"),
                    exc,
                    self.log_level,
                )
            )
    clear_queue(self.persona_queue)

    active_task = self.agent_runtime.get_active_task("default")
    if active_task is not None:
        with contextlib.suppress(Exception):
            self.agent_runtime.cancel_task(active_task.task_id)

    cleanup_errors.extend(_stop_test_runtime(self))

    if cleanup_errors:
        success = False
        detail = detail or "; ".join(cleanup_errors)
        if checks is not None:
            checks.append(
                {
                    "name": "cleanup",
                    "status": "failed",
                    "detail": "; ".join(cleanup_errors),
                }
            )

    if checks is not None:
        failed_check = next(
            (check for check in checks if check.get("status") == "failed"),
            None,
        )
        success = success and failed_check is None
        if detail is None and failed_check is not None:
            failed_detail = failed_check.get("detail")
            detail = failed_detail if isinstance(failed_detail, str) else None

    result: JSON = {
        "mode": mode,
        "success": success,
        "engine_state": "stopped",
        "task_state": task_state,
        "detail": detail,
    }
    summary = {"passed": 0, "failed": 0, "skipped": 0}
    if checks is not None:
        result["checks"] = cast(JSONSerializable, checks)
        summary = {
            "passed": sum(check.get("status") == "passed" for check in checks),
            "failed": sum(check.get("status") == "failed" for check in checks),
            "skipped": sum(check.get("status") == "skipped" for check in checks),
        }
        result["summary"] = cast(JSONSerializable, summary)
    self._runtime_state.test_result = result
    try:
        self.cur_state = "stopped"
    except Exception:
        self._runtime_state.cur_state = "stopped"
    if mode == "agent" and checks is not None:
        self.log(
            string(
                "test.agent_report_summary",
                passed=summary["passed"],
                failed=summary["failed"],
                skipped=summary["skipped"],
            ),
            "error" if not success else "info",
        )
        for check in checks:
            status = check.get("status")
            message_key = "test.agent_check_failed"
            if isinstance(status, str):
                message_key = {
                    "passed": "test.agent_check_passed",
                    "failed": "test.agent_check_failed",
                    "skipped": "test.agent_check_skipped",
                }.get(status, message_key)
            self.log(
                string(
                    message_key,
                    check=check.get("name", "unknown"),
                    message=check.get("detail", ""),
                ),
                "error" if status == "failed" else "info",
            )
    elif mode == "agent":
        message_key = "test.finished_success" if success else "test.finished_failure"
        with contextlib.suppress(Exception):
            self.log(
                string(
                    message_key,
                    mode=mode,
                    task_state=task_state or "none",
                    detail=detail or "none",
                    app_name=APP_NAME,
                ),
                "info" if success else "error",
            )
    return result


def _stop_test_runtime(self) -> list[str]:
    """Stop test-owned workers without entering normal process shutdown."""
    from ..pipeline.playback import close_stream

    errors: list[str] = []
    try:
        with self.queue_lock:
            self._speech_generation = getattr(self, "_speech_generation", 0) + 1
            self._playback_generation = getattr(self, "_playback_generation", 0) + 1
            self.utterance_force_stop.set()
            clear_queue(self.text_queue)
            clear_queue(self.audio_queue)
            self.text_queue.put(self.sentinel)
            self.audio_queue.put(self.sentinel)
    except Exception as exc:
        errors.append(
            format_error_message(
                string("test.cleanup_failed"),
                exc,
                self.log_level,
            )
        )

    try:
        close_stream(self, abort=True)
    except Exception as exc:
        errors.append(
            format_error_message(
                string("test.cleanup_failed"),
                exc,
                self.log_level,
            )
        )

    current_thread = threading.current_thread()
    for worker in (self.generation_thread, self.playback_thread):
        if worker is not None and worker is not current_thread:
            try:
                worker.join(timeout=2)
            except Exception as exc:
                errors.append(
                    format_error_message(
                        string("test.cleanup_failed"),
                        exc,
                        self.log_level,
                    )
                )

    persona_thread = self._persona_thread
    if persona_thread is not None and persona_thread is not current_thread:
        try:
            persona_thread.join(timeout=2)
        except Exception as exc:
            errors.append(
                format_error_message(
                    string("test.cleanup_failed"),
                    exc,
                    self.log_level,
                )
            )

    try:
        _close_agent_tool_selector(self)
    except Exception as exc:
        errors.append(
            format_error_message(
                string("test.cleanup_failed"),
                exc,
                self.log_level,
            )
        )

    try:
        self.component_locks.release_all()
        self._pipeline_lock_owner = None
        self.locked = True
        self.playback_done.set()
    except Exception as exc:
        errors.append(
            format_error_message(
                string("test.cleanup_failed"),
                exc,
                self.log_level,
            )
        )

    try:
        self.glow.leave()
        self.glow.finished.wait(timeout=5)
    except Exception as exc:
        errors.append(
            format_error_message(
                string("test.cleanup_failed"),
                exc,
                self.log_level,
            )
        )
    return errors


def _cleanup_residual_temp_data(self, temp_dir: Path) -> None:
    """Delete residual Celune temp artifacts that are not currently protected."""
    if not temp_dir.is_dir():
        return

    disposable_paths = [
        path for path in temp_dir.iterdir() if not is_protected_temp_path(path)
    ]
    trailing_files = len(disposable_paths)

    if trailing_files <= 0:
        return

    if trailing_files == 1:
        self.log(
            string("celune.residual_temp_item", app_name=APP_NAME),
            "warning",
        )
    else:
        self.log(
            string(
                "celune.residual_temp_items",
                app_name=APP_NAME,
                count=trailing_files,
            ),
            "warning",
        )
    self.log(string("celune.deleting"), "warning")

    with contextlib.suppress(OSError):
        for path in disposable_paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def _enter_fatal_error_state(self) -> None:
    """Mark the runtime as unrecoverably failed before fatal handlers run."""
    self.cur_state = "error"
    self.loaded = False
    self.locked = True
    self._ready_announced = False


def config_str(value: JSONSerializable) -> Optional[str]:
    """Return a config value only when it is a string."""
    return value if isinstance(value, str) else None


def config_int(value: JSONSerializable, default: int) -> int:
    """Preserve the core module's integer configuration helper surface."""
    return _config_int(value, default)


def _configured_locale(config: Config) -> Optional[str]:
    """Return one explicit locale override from env or config when provided."""
    env_value = config_str(os.getenv("CELUNE_LOCALE"))
    configured_value = config_str(config_value(config, "locale"))
    candidate = env_value if env_value is not None else configured_value
    if candidate is None:
        return None

    normalized = candidate.strip()
    if not normalized or normalized.lower() in {"auto", "system", "default"}:
        return None
    return normalized


def tts_quantization_enabled(
    config: Optional[Mapping[str, JSONSerializable]],
    backend_spec: CoreBackendSpec,
) -> bool:
    """Resolve the default TTS quantization policy, excluding LuxTTS."""
    if isinstance(backend_spec, CeluneVCBackend):
        return False
    if isinstance(backend_spec, type) and issubclass(backend_spec, CeluneVCBackend):
        return False
    raw_name = (
        backend_spec
        if isinstance(backend_spec, str)
        else getattr(backend_spec, "name", "")
    )
    backend_name = raw_name.strip().lower() if isinstance(raw_name, str) else ""
    return backend_name != "luxtts" and config_bool(
        config,
        "CELUNE_TTS_QUANTIZE",
        "quantize",
        default=True,
    )


def _resolve_input_mode(config: Config, requested_mode: Optional[str] = None) -> str:
    """Resolve Celune's active input mode from config and optional override."""
    candidate = requested_mode
    if candidate is None:
        candidate = config_str(config_value(config, "input_mode"))
    if candidate is None:
        configured_mode = config_str(config_value(config, "mode"))
        if configured_mode in {"text_to_speech", "tts", "voice_conversion", "revoice"}:
            candidate = configured_mode
    if candidate is None:
        return "text_to_speech"

    normalized = candidate.strip().lower()
    if normalized in {"text_to_speech", "tts"}:
        return "text_to_speech"
    if normalized in {"voice_conversion", "revoice"}:
        return "voice_conversion"
    raise ValueError(f"unknown input mode: '{candidate}'")


def _core_backend_target(
    backend_spec: CoreBackendSpec,
    log_callback: MessageCallback,
    input_mode: str,
) -> tuple[str, CoreBackendSpec]:
    """Return whether one backend specification targets TTS or VC mode."""
    discard(log_callback)
    if isinstance(backend_spec, CeluneVCBackend):
        return "vc", backend_spec

    if isinstance(backend_spec, CeluneBackend):
        return "tts", backend_spec

    if isinstance(backend_spec, type):
        if issubclass(backend_spec, CeluneVCBackend):
            return "vc", cast(VCBackendSpec, backend_spec)
        if issubclass(backend_spec, CeluneBackend):
            return "tts", cast(TTSBackendSpec, backend_spec)

    if isinstance(backend_spec, str):
        normalized_backend = backend_spec.strip().lower()
        if normalized_backend in BACKENDS:
            return "tts", cast(TTSBackendSpec, backend_spec)
        if normalized_backend in VC_BACKENDS:
            return "vc", cast(VCBackendSpec, backend_spec)

    return (
        ("vc", backend_spec)
        if input_mode == "voice_conversion"
        else ("tts", backend_spec)
    )


def agent_task_config(config: Config) -> AgentTaskConfig:
    """Build the agent task limits from the nested ``agent`` settings."""
    raw = config.get("agent")
    values = raw if isinstance(raw, dict) else {}

    def positive(name: str, default: int) -> int:
        value = values.get(name)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else default
        )

    max_tokens = values.get("max_tokens")
    return AgentTaskConfig(
        max_loops=positive("max_loops", AGENT_MAX_LOOPS),
        max_tokens=(
            max_tokens
            if (
                isinstance(max_tokens, int)
                and not isinstance(max_tokens, bool)
                and max_tokens > 0
            )
            else None
        ),
        context_size=min(
            positive("context_size", AGENT_CONTEXT_SPACE), AGENT_CONTEXT_SPACE
        ),
        compact_at=(
            values["compact_at"]
            if isinstance(values.get("compact_at"), int)
            and not isinstance(values["compact_at"], bool)
            and 1 <= values["compact_at"] <= 100
            else AGENT_COMPACT_AT
        ),
    )


def _resolve_core_backend_specs(
    log_callback: MessageCallback,
    input_mode: str,
    backend: Optional[CoreBackendSpec],
    tts_backend: Optional[CoreBackendSpec],
    vc_backend: Optional[CoreBackendSpec],
) -> tuple[Optional[CoreBackendSpec], Optional[CoreBackendSpec]]:
    """Merge the unified backend alias into the TTS and VC constructor slots."""
    if backend is None:
        return tts_backend, vc_backend

    backend_target, resolved_backend = _core_backend_target(
        backend,
        log_callback,
        input_mode,
    )
    if backend_target == "vc":
        if vc_backend is not None:
            raise BackendError("cannot specify both 'backend' and 'vc_backend'")
        return tts_backend, resolved_backend

    if tts_backend is not None:
        raise BackendError("cannot specify both 'backend' and 'tts_backend'")
    return resolved_backend, vc_backend


def _wrap_fatal_glow(self) -> None:
    """Ensure fatal glow paths also stamp Celune's runtime state."""
    if getattr(self.glow, "_celune_fatal_wrapped", False):
        return

    original_fatal = self.glow.fatal

    def wrapped_fatal() -> None:
        _enter_fatal_error_state(self)
        original_fatal()

    self.glow.fatal = wrapped_fatal
    self.glow._celune_fatal_wrapped = True


def is_voice_conversion_mode(self) -> bool:
    """Return whether Celune is operating with a voice-conversion backend."""
    return self.input_mode == "voice_conversion" or isinstance(
        self.vc_backend, CeluneVCBackend
    )


def active_runtime_backend_name(self) -> str:
    """Return the backend name representing the active speech runtime."""
    if is_voice_conversion_mode(self) and self.vc_backend is not None:
        return self.vc_backend.name
    return self.backend.name


def api_settings(self) -> tuple[bool, str, int, Optional[str], int, str]:
    """Return normalized API settings and report invalid configuration."""
    return config_api_settings(self.config, self.log, path_factory=api_key_path)


def acquire_component_lease(
    self,
    operation_id: str,
    component: ComponentLockName,
) -> tuple[bool, Optional[ComponentLockLease]]:
    """Reserve a named component for one in-flight operation."""
    manager = getattr(self, "component_locks", None)
    if manager is None:
        return True, None

    owner = ComponentLockOwner(operation_id=operation_id)
    acquisition, lease = manager.try_acquire_lease(
        (ComponentLockRequirement(component),),
        owner,
    )
    if lease is not None:
        return True, lease

    busy = acquisition.busy
    if busy is not None:
        self._last_component_busy = busy
        labels = ", ".join(component.name for component in busy.components)
        self.log(string("pipeline.busy_components", components=labels), "warning")
    return False, None


def acquire_model_loading_lease(
    self,
    operation_id: str,
) -> tuple[bool, Optional[ComponentLockLease]]:
    """Reserve the shared model lifecycle resource."""
    return acquire_component_lease(
        self,
        operation_id,
        ComponentLockName.MODEL_LOADING,
    )


def persona_conn(self) -> Optional[PersonaClient]:
    """Create the optional Persona runtime client for this engine."""
    if not persona_enabled(self.config):
        return None
    if not persona_is_available():
        self.log(string("celune.persona_init_failed"), "warning")
        return None
    return create_persona_client(self.config, log=self.log)


def start_persona_background_load(self) -> None:
    """Load Persona after TTS startup without delaying speech readiness."""
    with self._model_lock:
        vision = self.vision
        if vision is None or self.persona_ready or self.persona_loading:
            return

        self.persona_loading = True
        thread = threading.Thread(
            target=load_persona_background,
            args=(self, vision),
            daemon=True,
        )
        self._persona_load_thread = thread
    self.log(string("celune.initializing_persona"))
    thread.start()


def load_persona_background(self, vision: PersonaClient) -> None:
    """Load Persona in the background and publish its ready state."""
    acquired, component_lease = acquire_component_lease(
        self,
        f"persona-load:{id(vision)}",
        ComponentLockName.VLM,
    )
    if not acquired:
        with self._model_lock:
            if self.vision is vision:
                self.persona_loading = False
        return
    try:
        vision.load(persona_model_id(self.config), persona_quantization(self.config))
    except Exception as error:
        self.log(string("celune.persona_not_initialized"), "warning")
        self.log(string("celune.speech_only_mode"), "warning")
        self.log(
            format_error_message(
                string("celune.persona_init_failed"),
                error,
                self.log_level,
            ),
            "warning",
        )
        with self._model_lock:
            if self.vision is vision:
                self.vision = None
            self.persona_ready = False
            self.persona_loading = False
        vision.close()
    else:
        with self._model_lock:
            should_close = self.exit_requested or self.vision is not vision
            self.persona_ready = not should_close
            self.persona_loading = False
        if should_close:
            vision.close()
        else:
            self.log(string("celune.persona_initialized"))
            self.change_input_state_callback(locked=False)
    finally:
        with self._model_lock:
            if self._persona_load_thread is threading.current_thread():
                self._persona_load_thread = None
        if component_lease is not None:
            component_lease.release()


def unload_persona_state(self) -> None:
    """Release Persona state while reserving the model lifecycle resource."""
    acquired, component_lease = acquire_model_loading_lease(
        self,
        f"persona-unload:{time.monotonic_ns()}",
    )
    if not acquired:
        return
    try:
        _unload_persona_state_impl(self)
    finally:
        if component_lease is not None:
            component_lease.release()


def _unload_persona_state_impl(self) -> None:
    """Release Persona runtime state and clear the active client."""
    with self._model_lock:
        vision = self.vision
        self.vision = None
        self.persona_ready = False
        self.persona_loading = False
        persona_thread = self._persona_load_thread
        self._persona_load_thread = None
        analyzer = getattr(self, "persona_emotion_analyzer", None)
        if isinstance(analyzer, PersonaEmotionAnalyzer):
            analyzer.clear_vlm()
        self.persona_emotion_analyzer = None
    if persona_thread is not None and persona_thread is not threading.current_thread():
        persona_thread.join(timeout=2)
    if vision is not None:
        with contextlib.suppress(Exception):
            vision.close()


def unload_normalizer_components(self) -> None:
    """Release normalizer references and invalidate pending loads."""
    from .reload import release_loaded_object

    self._normalizer_load_epoch += 1
    llm = self.llm
    tokenizer = self.tokenizer
    self.llm = None
    self.tokenizer = None
    if llm is not None:
        release_loaded_object(llm)
    if tokenizer is not None:
        release_loaded_object(tokenizer)


def unload_runtime_state(
    self,
    include_normalizer: bool = False,
    include_vc: bool = True,
    close_backends: bool = False,
    release_cuda_cache: bool = True,
) -> None:
    """Unload runtime state while reserving the model lifecycle resource."""
    acquired, component_lease = acquire_model_loading_lease(
        self,
        f"runtime-unload:{time.monotonic_ns()}",
    )
    if not acquired:
        return
    try:
        _unload_runtime_state_impl(
            self,
            include_normalizer=include_normalizer,
            include_vc=include_vc,
            close_backends=close_backends,
            release_cuda_cache=release_cuda_cache,
        )
    finally:
        if component_lease is not None:
            component_lease.release()


def _unload_runtime_state_impl(
    self,
    include_normalizer: bool = False,
    include_vc: bool = True,
    close_backends: bool = False,
    release_cuda_cache: bool = True,
) -> None:
    """Unload models and optional normalization state to regain memory."""
    from .reload import dispose_backend, shutdown_backend, unload_backend_model

    discard(self, "model")
    if self.exit_requested:
        shutdown_backend(self.backend, release_cuda_cache)
    elif close_backends:
        dispose_backend(self.backend, release_cuda_cache=release_cuda_cache)
    else:
        unload_backend_model(self.backend, release_cuda_cache)

    if include_vc and self.vc_backend is not None:
        if self.exit_requested:
            shutdown_backend(self.vc_backend, release_cuda_cache)
        elif close_backends:
            dispose_backend(self.vc_backend, release_cuda_cache=release_cuda_cache)
        else:
            unload_backend_model(self.vc_backend, release_cuda_cache)

    if include_normalizer:
        unload_normalizer_components(self)

    gc.collect()
    if release_cuda_cache and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def close(self) -> None:
    """Shut off Celune and release loaded runtime state."""
    self._exit_requested = True
    with self._model_lock:
        if self._closed:
            return
        self._closed = True

    if self._api_tunnel is not None:
        self._api_tunnel.stop()
        self._api_tunnel = None

    # call this or else you get a zombified Celune that can't proceed with her exit procedures
    _abort_backend_operations(self)
    with _shutdown_runtime_lock(self):
        self.log(
            f"[ENGINE] close requested state={self.cur_state} loaded={self.loaded} "
            f"sleeping={self.sleeping}",
            loglevel="debug",
        )
        active_task = self.agent_runtime.get_active_task("default")
        if active_task is not None:
            with contextlib.suppress(Exception):
                self.agent_runtime.cancel_task(
                    active_task.task_id,
                    AgentCancellationReason.RUNTIME_SHUTDOWN,
                )
        vision = self.vision
        if isinstance(vision, PersonaClient):
            with contextlib.suppress(Exception):
                vision.interrupt()
        self.emit_event("shutdown", ShutdownEvent(celune=self))
        try:
            if self.extension_manager is not None:
                self.extension_manager.unregister_all()
            _close_speech_pipeline(self)
            wake_background_thread = self._wake_background_thread
            if (
                wake_background_thread is not None
                and wake_background_thread is not threading.current_thread()
            ):
                wake_background_thread.join(timeout=2)
            _close_agent_tool_selector(self)
            unload_persona_state(self)
            with self._model_lock:
                unload_runtime_state(self, include_normalizer=True)
        finally:
            with contextlib.suppress(Exception):
                close_default_loader()
            with contextlib.suppress(Exception):
                _cleanup_residual_temp_data(self, temp_data_dir())
            from ..registry.consent import set_commercial_mode

            set_commercial_mode(False)
            type(self)._instance = None
            self.log("[ENGINE] close complete", loglevel="debug")


def run_pipeline_jobs(self) -> None:
    """Start Celune's async speech pipeline inside one engine thread."""
    asyncio.run(_pipeline_jobs(self))


def _close_speech_pipeline(self) -> None:
    """Cancel and join speech workers before engine state is released."""
    from ..pipeline.playback import close_stream, invalidate_speech_work

    if not getattr(self, "test_finished", False):
        self.log(string("pipeline.exiting"))
    self._exit_requested = True
    invalidate_speech_work(self)
    close_stream(self, abort=True)
    self.text_queue.put(self.sentinel)
    self.audio_queue.put(self.sentinel)

    if self.generation_thread is not None:
        self.generation_thread.join(timeout=0.5)
    if self.playback_thread is not None:
        self.playback_thread.join(timeout=0.5)

    manager = getattr(self, "component_locks", None)
    try:
        close_stream(self, abort=True)
        self.glow.leave()
        self.glow.finished.wait(timeout=5)
    finally:
        if manager is not None:
            manager.release_all()


def stop_pipeline_jobs(self) -> None:
    """Stop startup pipeline workers after an initialization failure."""
    with self.queue_lock:
        self.text_queue.put(self.sentinel)
        if is_voice_conversion_mode(self):
            self.audio_queue.put(self.sentinel)

    if self.playback_thread is not None:
        self.playback_thread.join()


def _close_agent_tool_selector(self) -> None:
    """Release the optional loaded Needle handler during engine shutdown."""
    selector = self.agent_needle_selector
    if isinstance(selector, NeedleToolSelector):
        with contextlib.suppress(Exception):
            selector.handler.close()
    self.agent_needle_selector = None


@contextlib.contextmanager
def _shutdown_runtime_lock(self) -> Generator[None, None, None]:
    """Run shutdown cleanup without waiting for a reload operation."""
    acquired = self._async_runtime_lock.acquire(blocking=False)
    try:
        yield
    finally:
        if acquired:
            self._async_runtime_lock.release()


async def _pipeline_jobs(self) -> None:
    """Run Celune's speech pipeline workers as async jobs."""
    from .reload import reload_model_for_language

    from ..pipeline.playback import playback_worker_job
    from ..pipeline.generation import generation_worker_job

    playback_task = asyncio.create_task(playback_worker_job(self))
    if is_voice_conversion_mode(self):
        await playback_task
        return

    generation_task = asyncio.create_task(
        generation_worker_job(
            self,
            partial(reload_model_for_language, self),
        )
    )
    await asyncio.gather(generation_task, playback_task)


def _abort_backend_operations(self) -> None:
    """Abort backend-owned work before pipeline and model teardown begins."""
    seen: set[int] = set()
    for backend in (self.backend, self.vc_backend, self._reload_backend):
        if backend is None or id(backend) in seen:
            continue
        seen.add(id(backend))
        abort = getattr(backend, "abort", None)
        if callable(abort):
            with contextlib.suppress(Exception):
                abort()


def fatal(self) -> None:
    """Mark Celune state as fatal and prevent further operations."""
    self.loaded = False
    self.cur_state = "error"
    self.glow.fatal()
    from .voice import try_play_signal

    if not try_play_signal(self, "error"):
        self.log(
            string("ui.error_signal_unavailable"),
            "warning",
            loglevel="verbose",
        )
