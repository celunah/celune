# SPDX-License-Identifier: Apache-2.0
"""Celune's backend layer."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import gc
import os
import time
import queue
import shutil
import asyncio
import threading
import contextlib
from typing import Never, Union, ClassVar, Optional, cast, final
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Generator

import numpy as np
import torch
import numpy.typing as npt
from transformers.modeling_utils import PreTrainedModel

from .vc import clamp_vc_pitch_shift
from .i18n import string, set_locale, tagged_string, get_system_locale
from .vram import (
    VramPreset,
    backend_allowed,
    resolve_vram_preset,
    resolve_backend_name,
)
from .locks import ComponentLockLease
from .modes import (
    OperationMode,
    resolve_operation_mode,
)
from .paths import temp_data_dir
from .utils import (
    available,
    discard,
    format_error_message,
)
from .chroma import AudioRGBGlow
from .config import Config, config_bool, config_value, normalize_log_level
from .cevoice import (
    CEVoicePersona,
    default_loader,
    active_bundle_path,
    select_voice_bundle,
    close_default_loader,
    bundle_character_name,
    is_protected_temp_path,
    persona_metadata_from_manifest,
    bundle_matches_default_pack_checksum,
)
from .pipeline import (
    clear_queue,
    close_stream,
    generation_worker_job,
    playback_worker_job,
    split_text,
)
from .speech import (
    play as play_pipeline,
    close as close_pipeline,
    queue_sfx_audio,
    stop_live_audio_input,
)
from .constants import (
    APP_NAME,
    AGENT_MAX_LOOPS,
    AGENT_COMPACT_AT,
    AGENT_CONTEXT_SPACE,
)
from .exceptions import (
    WarmupError,
    BackendError,
)
from .agent.tools import (
    agent_test_tools,
    production_agent_tools,
    agent_test_tool_schemas,
    production_agent_tool_schemas,
)
from .backends.vc import VC_BACKENDS, CeluneVCBackend, resolve_vc_backend
from .agent.needle import NeedleToolSelector
from .backends.tts import BACKENDS, CeluneBackend, resolve_backend
from .persona.impl import (
    PersonaClient,
    persona_enabled,
    persona_model_id,
    persona_is_available,
    persona_quantization,
    create_persona_client,
)
from .typing.agent import (
    AgentTaskConfig,
    AgentToolSelector,
    AgentCancellationReason,
)
from .typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from .typing.modes import BackendMode
from .agent.persona import PersonaAgentBridge
from .agent.routing import AgentInputRouter
from .agent.runtime import AgentRuntime
from .typing.celune import (
    VCBackendSpec,
    TTSBackendSpec,
    CaptionCallback,
    CoreBackendSpec,
    MessageCallback,
    ProgressCallback,
    ReleasableObject,
    InputStateCallback,
    CeluneStateAccessors,
    CeluneMethodSurface,
    CaptionTimingCallback,
    VoiceLockStateCallback,
    _BundleWithPath,
)
from .typing.common import JSON, JSONSerializable
from .typing.events import EventName, EventPayload
from .typing.aliases import LogLevel, AudioChunk
from .persona.emotion import PersonaEmotionAnalyzer
from .typing.backends import BackendModel
from .extensions.events import EventDispatcher
from .dataclasses.celune import (
    CELUNE_CONSTANT_PROPERTIES,
    CELUNE_FORWARDED_PROPERTIES,
    CeluneAudioState,
    CeluneModelState,
    CeluneVoiceState,
    CeluneBackendState,
    CeluneRuntimeState,
    CeluneCallbackState,
    CelunePipelineState,
)
from .dataclasses.events import (
    ShutdownEvent,
    StateChangedEvent,
    VoiceChangedEvent,
    CharacterLoadedEvent,
    CharacterChangedEvent,
    CharacterUnloadedEvent,
)
from .dataclasses.properties import (
    bind_constant_properties,
    bind_forwarded_properties,
)


def _config_str(value: JSONSerializable) -> Optional[str]:
    """Return a config value only when it is a string."""
    return value if isinstance(value, str) else None


def _configured_locale(config: Config) -> Optional[str]:
    """Return one explicit locale override from env or config when provided."""
    env_value = _config_str(os.getenv("CELUNE_LOCALE"))
    configured_value = _config_str(config_value(config, "locale"))
    candidate = env_value if env_value is not None else configured_value
    if candidate is None:
        return None

    normalized = candidate.strip()
    if not normalized or normalized.lower() in {"auto", "system", "default"}:
        return None
    return normalized


def _config_int(value: JSONSerializable, default: int) -> int:
    """Return a config value as an integer when it is scalar-like."""
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer config value")
    if isinstance(value, (int, float, str)):
        return int(value)
    if value is None:
        return default
    raise TypeError("config value cannot be converted to int")


def _resolve_input_mode(config: Config, requested_mode: Optional[str] = None) -> str:
    """Resolve Celune's active input mode from config and optional override."""
    candidate = requested_mode
    if candidate is None:
        candidate = _config_str(config_value(config, "input_mode"))
    if candidate is None:
        configured_mode = _config_str(config_value(config, "mode"))
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


def _agent_task_config(config: Config) -> AgentTaskConfig:
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
        context_size=positive("context_size", AGENT_CONTEXT_SPACE),
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


def _release_loaded_object(value: ReleasableObject) -> None:
    """Best-effort release hook for one loaded runtime object."""
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()
        return

    unload = getattr(value, "unload", None)
    if callable(unload):
        with contextlib.suppress(Exception):
            unload()


def _unload_backend_model(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool,
) -> None:
    """Unload one backend while retaining compatibility with older backend plugins."""
    try:
        backend.unload_model(release_cuda_cache=release_cuda_cache)
    except TypeError as error:
        if "release_cuda_cache" not in str(error):
            raise
        backend.unload_model()


def _dispose_backend(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool = True,
) -> None:
    """Unload one backend and close its worker process when supported."""
    try:
        _unload_backend_model(backend, release_cuda_cache)
    finally:
        _close_backend(backend)


def _close_backend(backend: Union[CeluneBackend, CeluneVCBackend]) -> None:
    """Close one backend process without repeating model unloading."""
    close = getattr(backend, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _shutdown_backend(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool,
) -> None:
    """Release one backend without waiting on an already-aborted operation."""
    if callable(getattr(backend, "close", None)):
        _close_backend(backend)
        return
    _unload_backend_model(backend, release_cuda_cache)


@final
class Celune(CeluneMethodSurface, CeluneStateAccessors):
    """The character engine for Celune."""

    _normalizer_load_epoch: int

    def __init_subclass__(cls, **kwargs: Never) -> Never:
        raise TypeError(f"{__class__.__name__} is final and cannot be subclassed")

    _instance: ClassVar[Optional["Celune"]] = None

    @dataclass
    class _ReloadSnapshot:
        """Rollback state captured before a backend or CEVOICE hot reload."""

        backend: CeluneBackend
        restorable_backend_spec: Union[str, type[CeluneBackend]]
        backend_spec: Optional[Union[str, type[CeluneBackend]]]
        backend_kwargs: dict[str, JSONSerializable]
        vc_backend: Optional[CeluneVCBackend]
        restorable_vc_backend_spec: Optional[Union[str, type[CeluneVCBackend]]]
        vc_backend_spec: Optional[Union[str, type[CeluneVCBackend]]]
        voice_conversion_backend: str
        tts_backend: str
        input_mode: str
        model: Optional[BackendModel]
        model_name: str
        voices: tuple[str, ...]
        current_voice: Optional[str]
        current_character: Optional[str]
        current_character_persona: Optional[CEVoicePersona]
        voice_bundle_is_default: bool
        loaded: bool
        cur_state: str

    def __init__(
        self,
        config: Config,
        backend: Optional[CoreBackendSpec] = None,
        tts_backend: Optional[CoreBackendSpec] = None,
        vc_backend: Optional[CoreBackendSpec] = None,
        vc_pitch_shift: Optional[int] = None,
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
        caption_progress_callback: Optional[ProgressCallback] = None,
        caption_callback: Optional[CaptionCallback] = None,
        caption_timing_callback: Optional[CaptionTimingCallback] = None,
        log_level: LogLevel = "info",
        agent_tool_selector: Optional[AgentToolSelector] = None,
        backend_mode: BackendMode = "normal",
        startup_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Create the Celune engine and retain its deferred startup callback.

        The optional ``startup_callback`` receives the initialization-core
        checkpoint when :meth:`load` begins.
        """
        if Celune._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")
        if backend_mode not in {"normal", "ui_test", "agent_test"}:
            raise ValueError(f"unknown Celune backend mode: '{backend_mode}'")

        self._startup_log_buffer: list[tuple[str, str, LogLevel]] = []
        self._startup_banner_emitted = False
        self._startup_log_sink = log_callback or self._noop_message
        self._startup_callback = startup_callback
        self._callbacks = CeluneCallbackState(
            log_callback=self._buffer_startup_log,
            status_callback=status_callback or self._noop_message,
            error_callback=error_callback or (lambda error: None),
            idle_callback=idle_callback or (lambda: None),
            queue_avail_callback=queue_avail_callback or (lambda: None),
            voice_changed_callback=voice_changed_callback or (lambda name: None),
            change_input_state_callback=(
                change_input_state_callback or self._noop_input_state
            ),
            change_voice_lock_state_callback=(
                change_voice_lock_state_callback or self._noop_voice_lock_state
            ),
            progress_callback=(progress_callback or self._noop_progress),
            caption_progress_callback=(
                caption_progress_callback or self._noop_progress
            ),
            caption_callback=(caption_callback or self._noop_caption),
            caption_timing_callback=(
                caption_timing_callback or self._noop_caption_timing
            ),
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
        self._runtime_state.backend_mode = backend_mode
        set_locale(_configured_locale(config) or get_system_locale())
        self.mode: OperationMode = resolve_operation_mode(config)
        if backend_mode == "agent_test":
            self._agent_tools = agent_test_tools(self)
            self._agent_tool_schemas = agent_test_tool_schemas()
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
            self._agent_tools = production_agent_tools(
                self,
                include_local_management=local_management,
            )
            self._agent_tool_schemas = production_agent_tool_schemas(
                include_local_management=local_management,
            )
            if local_management and config.get("mode") == "agent":
                self.log(
                    "\n".join(
                        (
                            string("agent.unsandboxed_title"),
                            string("agent.unsandboxed_body"),
                            string("agent.unsandboxed_rollback"),
                        )
                    ),
                    "warning",
                )
        self._agent_needle_selector = agent_tool_selector
        self._agent_needle_error: Optional[str] = None
        self._agent_persona_bridge = PersonaAgentBridge(
            self,
            self._agent_tool_schemas,
        )
        self.input_mode = _resolve_input_mode(config, input_mode)
        self.agent_runtime = AgentRuntime(
            tools=self._agent_tools,
            event_dispatcher=self._event_dispatcher,
            celune=self,
            planner=self._agent_persona_bridge.plan,
            tool_selector=self._select_agent_tool,
            tool_executor=self._execute_agent_tool,
            tool_result_handler=self._agent_persona_bridge.handle_tool_result,
            responder=self._agent_persona_bridge.respond,
            tool_schemas=self._agent_tool_schemas,
            task_config=_agent_task_config(config),
        )
        self._agent_router = AgentInputRouter(self, self.agent_runtime)
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
        self._wrap_fatal_glow()
        self.glow.start()

        tts_backend, vc_backend = _resolve_core_backend_specs(
            self.log_callback,
            self.input_mode,
            backend,
            tts_backend,
            vc_backend,
        )
        self.vc_pitch_shift = (
            0
            if vc_pitch_shift is None
            else clamp_vc_pitch_shift(_config_int(vc_pitch_shift, 0))
        )
        self.vc_f0_condition = False if vc_f0_condition is None else vc_f0_condition
        select_voice_bundle(_config_str(config_value(config, "voice_bundle")))
        preset = resolve_vram_preset(config)

        if tts_backend is None:
            tts_backend = preset.default_backend

        backend_kwargs: dict[str, Optional[Union[bool, str]]] = {}
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
        if backend_name == "qwen3":
            backend_kwargs["x_vector_only"] = config_bool(
                config,
                "CELUNE_QWEN3_X_VECTOR_ONLY",
                "qwen3_x_vector_only",
            )
            backend_kwargs["clone_model_id"] = preset.qwen3_clone_model_id

        try:
            resolved_tts_backend = cast(TTSBackendSpec, tts_backend)
            if not isinstance(tts_backend, CeluneBackend):
                self._backend_spec = cast(
                    Union[str, type[CeluneBackend]],
                    resolved_tts_backend,
                )
                self._backend_kwargs = dict(backend_kwargs)
            self.backend = self._resolve_tts_backend(
                resolved_tts_backend,
                log=self.log_callback,
                fatal=self.fatal,
                **backend_kwargs,
            )
            self.backend.bind_fatal(self.fatal)
            self._validate_backend_against_preset(self.backend, preset)
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
            vc_backend = _config_str(
                config_value(config, "voice_conversion_backend")
            ) or _config_str(config_value(config, "vc_backend"))
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
                self.vc_backend = self._resolve_vc_backend(
                    resolved_vc_backend,
                    log=self.log_callback,
                )
                if available("pitch_shift", obj=self.vc_backend):
                    self.vc_backend.pitch_shift = self.vc_pitch_shift
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
        self.use_normalization = config_bool(
            config, "CELUNE_NORMALIZE", "use_normalizer"
        )

        self.vision = self._persona_conn()

        Celune._instance = self

    bind_forwarded_properties(locals(), CELUNE_FORWARDED_PROPERTIES)
    bind_constant_properties(locals(), CELUNE_CONSTANT_PROPERTIES)

    @property
    def dev(self) -> bool:
        """Return whether compatibility developer diagnostics are enabled."""
        return self.log_level != "info"

    @dev.setter
    def dev(self, value: bool) -> None:
        """Map the legacy developer flag onto the verbose log level."""
        if value:
            self.log_level = "verbose"
        elif self.log_level == "verbose":
            self.log_level = "info"

    @property
    def cur_state(self) -> str:
        """Return Celune's current runtime state.

        Returns:
            str: The current runtime-state label.
        """
        return self._runtime_state.cur_state

    @cur_state.setter
    def cur_state(self, value: str) -> None:
        """Store Celune's runtime state and emit transition events on change.

        Args:
            value: The new runtime-state label to store.
        """
        if self.test_finished and value != "stopped":
            return
        old_state = self._runtime_state.cur_state
        self._runtime_state.cur_state = value
        if old_state == value:
            return
        self.log(
            f"[STATE] transition old={old_state} new={value}",
            loglevel="debug",
        )
        self._emit_event(
            "state_changed",
            StateChangedEvent(
                celune=self,
                old_state=old_state,
                new_state=value,
            ),
        )

    def finish_test_mode(
        self,
        mode: str,
        success: bool,
        *,
        task_state: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> JSON:
        """Finish an explicit test mode and leave the engine stopped but alive.

        Args:
            mode: The selected test mode name.
            success: Whether the controlled test completed successfully.
            task_state: Final agent task state, when the mode created a task.
            detail: Optional diagnostic detail retained with the final result.

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

        cleanup_errors.extend(self._stop_test_runtime())

        if cleanup_errors:
            success = False
            detail = detail or "; ".join(cleanup_errors)

        result: JSON = {
            "mode": mode,
            "success": success,
            "engine_state": "stopped",
            "task_state": task_state,
            "detail": detail,
        }
        self._runtime_state.test_result = result
        try:
            self.cur_state = "stopped"
        except Exception:
            self._runtime_state.cur_state = "stopped"
        if mode == "agent":
            message_key = (
                "test.finished_success" if success else "test.finished_failure"
            )
            with contextlib.suppress(Exception):
                self.log(
                    string(
                        message_key,
                        mode=mode,
                        task_state=task_state or "none",
                        detail=detail or "none",
                    ),
                    "info" if success else "error",
                )
        return result

    def _stop_test_runtime(self) -> list[str]:
        """Stop test-owned workers without entering normal process shutdown."""
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
            self._close_agent_tool_selector()
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

    @staticmethod
    def _noop_message(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Discard a message callback."""

    @staticmethod
    def _noop_input_state(locked: bool) -> None:
        """Discard an input lock-state callback."""

    @staticmethod
    def _noop_voice_lock_state(locked: bool) -> None:
        """Discard a voice lock-state callback."""

    @staticmethod
    def _noop_progress(progress: Optional[float], total: Optional[float]) -> None:
        """Discard a progress callback."""

    @staticmethod
    def _noop_caption(caption: Optional[str]) -> None:
        """Discard a speech caption update."""

    @staticmethod
    def _noop_caption_timing(
        caption: str,
        audio: AudioChunk,
        sample_rate: int,
        timing_text: Optional[str] = None,
    ) -> None:
        """Discard generated speech caption timing input."""

    def _enter_fatal_error_state(self) -> None:
        """Mark the runtime as unrecoverably failed before fatal handlers run."""
        self.cur_state = "error"
        self.loaded = False
        self.locked = True
        self._ready_announced = False

    def _wrap_fatal_glow(self) -> None:
        """Ensure all fatal glow paths also stamp the runtime state as failed."""
        if getattr(self.glow, "_celune_fatal_wrapped", False):
            return

        original_fatal = self.glow.fatal

        def wrapped_fatal() -> None:
            self._enter_fatal_error_state()
            original_fatal()

        self.glow.fatal = wrapped_fatal
        self.glow._celune_fatal_wrapped = True

    def _emit_event(self, event_name: EventName, event: EventPayload) -> None:
        """Dispatch one typed event through Celune's internal event bus."""
        self.log(
            f"[EVENT] emit name={event_name} payload={type(event).__name__}",
            loglevel="debug",
        )
        self._event_dispatcher.emit(event_name, event)
        self.log(
            f"[EVENT] emit_return name={event_name}",
            loglevel="debug",
        )

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

    @staticmethod
    def _bundle_path_string(bundle: Optional[_BundleWithPath]) -> Optional[str]:
        """Return one bundle path as a string when it is available."""
        if bundle is None:
            return None
        return str(bundle.path)

    def _emit_character_event_transition(
        self,
        old_character: Optional[str],
        old_bundle_path: Optional[str],
        new_character: Optional[str],
        new_bundle_path: Optional[str],
        new_is_default: bool,
    ) -> None:
        """Emit the appropriate character lifecycle event for one bundle transition."""
        if old_character and new_character:
            if old_character == new_character and old_bundle_path == new_bundle_path:
                return
            self._emit_event(
                "character_changed",
                CharacterChangedEvent(
                    celune=self,
                    old_character=old_character,
                    new_character=new_character,
                    old_bundle_path=old_bundle_path,
                    new_bundle_path=new_bundle_path,
                    new_is_default=new_is_default,
                ),
            )
            return

        if new_character:
            self._emit_event(
                "character_loaded",
                CharacterLoadedEvent(
                    celune=self,
                    character_name=new_character,
                    bundle_path=new_bundle_path,
                    is_default=new_is_default,
                ),
            )
            return

        if old_character:
            self._emit_event(
                "character_unloaded",
                CharacterUnloadedEvent(
                    celune=self,
                    character_name=old_character,
                    bundle_path=old_bundle_path,
                ),
            )

    @staticmethod
    def _validate_backend_against_preset(
        backend: CeluneBackend,
        preset: VramPreset,
    ) -> None:
        """Reject backend instances that bypass preset-specific runtime limits."""
        if (
            backend.name == "qwen3"
            and getattr(backend, "clone_model_id", None) != preset.qwen3_clone_model_id
        ):
            raise BackendError(
                f"backend '{backend.name}' is not available with model "
                f"'{getattr(backend, 'clone_model_id', None)}' for VRAM tier '{preset.tier}'"
            )

    def is_voice_conversion_mode(self) -> bool:
        """Public interface for Celune._is_voice_conversion_mode.

        Returns:
            bool: Whether this Celune instance is currently in voice conversion mode.
        """
        return self._is_voice_conversion_mode()

    def _is_voice_conversion_mode(self) -> bool:
        """Return whether Celune is currently running in voice-conversion mode."""
        return self.input_mode == "voice_conversion" or isinstance(
            self.vc_backend, CeluneVCBackend
        )

    def _active_runtime_backend_name(self) -> str:
        """Return the backend name that should represent the active speech runtime."""
        if self._is_voice_conversion_mode() and self.vc_backend is not None:
            return self.vc_backend.name
        return self.backend.name

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        """Drain all pending items from a queue."""
        clear_queue(q)

    def _acquire_component_lease(
        self,
        operation_id: str,
        component: ComponentLockName,
    ) -> tuple[bool, Optional[ComponentLockLease]]:
        """Reserve one component resource for an operation."""
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

    def _acquire_model_loading_lease(
        self,
        operation_id: str,
    ) -> tuple[bool, Optional[ComponentLockLease]]:
        """Reserve the shared model lifecycle resource for one operation."""
        return self._acquire_component_lease(
            operation_id,
            ComponentLockName.MODEL_LOADING,
        )

    def _persona_conn(self) -> Optional[PersonaClient]:
        """Return a connection to the Persona runtime, if available."""

        if not persona_enabled(self.config):
            return None

        if not persona_is_available():
            self.log(string("celune.persona_init_failed"), "warning")
            return None

        return create_persona_client(self.config, log=self.log)

    def _start_persona_background_load(self) -> None:
        """Load Persona after TTS startup without blocking speech readiness."""
        with self._model_lock:
            vision = self.vision
            if vision is None or self.persona_ready or self.persona_loading:
                return

            self.persona_loading = True
            thread = threading.Thread(
                target=self._load_persona_background,
                args=(vision,),
                daemon=True,
            )
            self._persona_load_thread = thread
        self.log(string("celune.initializing_persona"))
        thread.start()

    def _load_persona_background(self, vision: PersonaClient) -> None:
        """Load Persona in the background and publish its ready state."""
        acquired, component_lease = self._acquire_component_lease(
            f"persona-load:{id(vision)}",
            ComponentLockName.VLM,
        )
        if not acquired:
            with self._model_lock:
                if self.vision is vision:
                    self.persona_loading = False
            return
        try:
            vision.load(
                persona_model_id(self.config),
                persona_quantization(self.config),
            )
        except Exception as e:
            self.log(string("celune.persona_not_initialized"), "warning")
            self.log(string("celune.speech_only_mode"), "warning")
            self.log(
                format_error_message(
                    string("celune.persona_init_failed"),
                    e,
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

    def _close_stream(self, abort: bool = False) -> None:
        """Close the current audio stream if one exists."""
        close_stream(self, abort=abort)

    def _unload_persona_state(self) -> None:
        """Release Persona state while reserving the model lifecycle resource."""
        acquired, component_lease = self._acquire_model_loading_lease(
            f"persona-unload:{time.monotonic_ns()}"
        )
        if not acquired:
            return
        try:
            self._unload_persona_state_impl()
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
        if (
            persona_thread is not None
            and persona_thread is not threading.current_thread()
        ):
            persona_thread.join(timeout=2)
        if vision is not None:
            with contextlib.suppress(Exception):
                vision.close()

    def _unload_normalizer_components(self) -> None:
        """Release normalizer model references and invalidate pending loads."""
        self._normalizer_load_epoch += 1
        llm = self.llm
        tokenizer = self.tokenizer
        self.llm = None
        self.tokenizer = None
        if llm is not None:
            _release_loaded_object(llm)
        if tokenizer is not None:
            _release_loaded_object(tokenizer)

    def unload_runtime_state(
        self,
        include_normalizer: bool = False,
        include_vc: bool = True,
        close_backends: bool = False,
        release_cuda_cache: bool = True,
    ) -> None:
        """Unload runtime state while reserving the model lifecycle resource."""
        acquired, component_lease = self._acquire_model_loading_lease(
            f"runtime-unload:{time.monotonic_ns()}"
        )
        if not acquired:
            return
        try:
            self._unload_runtime_state_impl(
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
        """Unload unused models to regain memory.

        Args:
            include_normalizer: Whether to also unload the normalization model and tokenizer.
            include_vc: Whether to also unload the voice-conversion backend runtime.
            close_backends: Whether to terminate isolated backend workers after unloading.
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """
        discard(self, "model")

        if self.exit_requested:
            _shutdown_backend(self.backend, release_cuda_cache)
        elif close_backends:
            _dispose_backend(self.backend, release_cuda_cache=release_cuda_cache)
        else:
            _unload_backend_model(self.backend, release_cuda_cache)
        if include_vc and self.vc_backend is not None:
            if self.exit_requested:
                _shutdown_backend(self.vc_backend, release_cuda_cache)
            elif close_backends:
                _dispose_backend(
                    self.vc_backend,
                    release_cuda_cache=release_cuda_cache,
                )
            else:
                _unload_backend_model(self.vc_backend, release_cuda_cache)

        if include_normalizer:
            self._unload_normalizer_components()

        gc.collect()

        if release_cuda_cache and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def _recreate_tts_backend(self) -> bool:
        """Rebuild the TTS backend from its original constructor recipe."""
        if self._backend_spec is None:
            return False

        candidate_backend = self._resolve_tts_backend(
            self._backend_spec,
            log=self.log_callback,
            fatal=self.fatal,
            **self._backend_kwargs,
        )
        candidate_backend.bind_fatal(self.fatal)
        _close_backend(self.backend)
        self.backend = candidate_backend
        self.tts_backend = candidate_backend.name
        return True

    def _resolve_tts_backend(
        self,
        backend_spec: TTSBackendSpec,
        **backend_kwargs,
    ) -> CeluneBackend:
        """Resolve one named TTS backend through its CEDTS worker manifest."""
        backend = resolve_backend(backend_spec, **backend_kwargs)
        backend.bind_progress(self.progress_callback)
        backend.log_level = getattr(self, "log_level", "info")
        return backend

    def _recreate_vc_backend(self) -> bool:
        """Rebuild the VC backend from its original constructor recipe."""
        if self._vc_backend_spec is None:
            return False

        candidate_backend = self._resolve_vc_backend(
            self._vc_backend_spec,
            log=self.log_callback,
        )
        if available("pitch_shift", obj=candidate_backend):
            candidate_backend.pitch_shift = self.vc_pitch_shift
        if available("f0_condition", obj=candidate_backend):
            candidate_backend.f0_condition = self.vc_f0_condition
        previous_backend = self.vc_backend
        if previous_backend is not None:
            _close_backend(previous_backend)
        self.vc_backend = candidate_backend
        self.voice_conversion_backend = candidate_backend.name
        return True

    def _resolve_vc_backend(
        self,
        backend_spec: VCBackendSpec,
        log: Optional[MessageCallback] = None,
    ) -> CeluneVCBackend:
        """Resolve one VC backend from the active application environment."""
        backend = resolve_vc_backend(backend_spec, log=log)
        backend.bind_progress(self.progress_callback)
        backend.log_level = getattr(self, "log_level", "info")
        return backend

    def _restorable_vc_backend_spec(
        self,
    ) -> Optional[Union[str, type[CeluneVCBackend]]]:
        """Return a VC backend specification that does not pin the current instance."""
        if self.vc_backend is None:
            return None
        if self._vc_backend_spec is not None:
            return self._vc_backend_spec
        return type(self.vc_backend)

    def _restorable_active_backend_spec(
        self,
    ) -> Union[str, type[CeluneBackend], type[CeluneVCBackend]]:
        """Return a backend specification for whichever backend family is active."""
        if self._is_voice_conversion_mode() and self.vc_backend is not None:
            restorable_vc_backend = self._restorable_vc_backend_spec()
            if restorable_vc_backend is not None:
                return restorable_vc_backend
        return self._restorable_backend_spec()

    def _backend_reload_kwargs(
        self,
        backend_spec: Union[str, CeluneBackend, type[CeluneBackend]],
    ) -> dict[str, JSONSerializable]:
        """Return constructor kwargs needed to instantiate one backend specification."""
        backend_kwargs: dict[str, JSONSerializable] = {}
        raw_name = getattr(backend_spec, "name", None)
        backend_name = (
            backend_spec.strip().lower()
            if isinstance(backend_spec, str)
            else raw_name
            if isinstance(raw_name, str)
            else None
        )
        if isinstance(backend_spec, CeluneBackend):
            return backend_kwargs

        if backend_name == "qwen3":
            preset = resolve_vram_preset(self.config)
            backend_kwargs["x_vector_only"] = config_bool(
                self.config,
                "CELUNE_QWEN3_X_VECTOR_ONLY",
                "qwen3_x_vector_only",
            )
            backend_kwargs["clone_model_id"] = preset.qwen3_clone_model_id
        return backend_kwargs

    def _capture_reload_snapshot(self) -> _ReloadSnapshot:
        """Capture the current backend runtime state for rollback."""
        return self._ReloadSnapshot(
            backend=self.backend,
            restorable_backend_spec=self._restorable_backend_spec(),
            backend_spec=self._backend_spec,
            backend_kwargs=dict(self._backend_kwargs),
            vc_backend=self.vc_backend,
            restorable_vc_backend_spec=self._restorable_vc_backend_spec(),
            vc_backend_spec=self._vc_backend_spec,
            voice_conversion_backend=self.voice_conversion_backend,
            tts_backend=self.tts_backend,
            input_mode=self.input_mode,
            model=self.model,
            model_name=self.model_name,
            voices=self.voices,
            current_voice=self.current_voice,
            current_character=self.current_character,
            current_character_persona=self.current_character_persona,
            voice_bundle_is_default=self.voice_bundle_is_default,
            loaded=self.loaded,
            cur_state=self.cur_state,
        )

    def _release_reload_snapshot(self, snapshot: _ReloadSnapshot) -> None:
        """Drop snapshot references after a successful hot reload."""
        snapshot.model = None
        snapshot.backend = self.backend
        snapshot.vc_backend = self.vc_backend
        snapshot.current_character_persona = None

    def _restorable_backend_spec(self) -> Union[str, type[CeluneBackend]]:
        """Return a backend specification that does not pin the current instance."""
        if self._backend_spec is not None:
            return self._backend_spec
        return type(self.backend)

    def _restore_reload_snapshot(self, snapshot: _ReloadSnapshot) -> None:
        """Restore backend runtime state after a failed hot reload."""
        self.backend = snapshot.backend
        self._backend_spec = snapshot.backend_spec
        self._backend_kwargs = dict(snapshot.backend_kwargs)
        self.vc_backend = snapshot.vc_backend
        self._vc_backend_spec = snapshot.vc_backend_spec
        self.voice_conversion_backend = snapshot.voice_conversion_backend
        self.tts_backend = snapshot.tts_backend
        self.input_mode = snapshot.input_mode
        self.model = cast(Optional[PreTrainedModel], snapshot.model)
        self.backend.model = snapshot.model
        self.model_name = snapshot.model_name
        self.voices = snapshot.voices
        self.current_voice = snapshot.current_voice
        self.current_character = snapshot.current_character
        self.current_character_persona = snapshot.current_character_persona
        self.voice_bundle_is_default = snapshot.voice_bundle_is_default
        self.loaded = snapshot.loaded
        self.cur_state = snapshot.cur_state

    def _rebuild_reload_snapshot_runtime(self, snapshot: _ReloadSnapshot) -> None:
        """Recreate the previous backend runtime from a rollback snapshot."""
        _close_backend(snapshot.backend)
        restored_backend = self._resolve_tts_backend(
            snapshot.restorable_backend_spec,
            log=self.log_callback,
            fatal=self.fatal,
            **snapshot.backend_kwargs,
        )
        restored_backend.bind_fatal(self.fatal)
        if restored_backend.uses_voice_bundles:
            restored_backend.validate_refs()

        restored_model: Optional[PreTrainedModel] = None
        restored_model_name = snapshot.model_name
        if snapshot.loaded and snapshot.current_voice is not None:
            restored_model, restored_model_name = self._load_backend_voice_runtime(
                restored_backend,
                snapshot.current_voice,
            )

        snapshot.backend = restored_backend
        snapshot.model = cast(Optional[BackendModel], restored_model)
        snapshot.model_name = restored_model_name

    def _rebuild_reload_snapshot_vc_runtime(self, snapshot: _ReloadSnapshot) -> None:
        """Recreate the previous VC backend runtime from a rollback snapshot."""
        if snapshot.restorable_vc_backend_spec is None:
            return

        if snapshot.vc_backend is not None:
            _close_backend(snapshot.vc_backend)
        restored_vc_backend = self._resolve_vc_backend(
            snapshot.restorable_vc_backend_spec,
            log=self.log_callback,
        )
        if available("pitch_shift", obj=restored_vc_backend):
            restored_vc_backend.pitch_shift = self.vc_pitch_shift
        if available("f0_condition", obj=restored_vc_backend):
            restored_vc_backend.f0_condition = self.vc_f0_condition
        if snapshot.loaded:
            restored_vc_backend.preload_models()
        snapshot.vc_backend = restored_vc_backend

    @staticmethod
    def _resolve_voice_state(
        backend: CeluneBackend,
        preferred_voice: Optional[str] = None,
    ) -> tuple[
        tuple[str, ...],
        Optional[str],
        Optional[str],
        Optional[CEVoicePersona],
        bool,
    ]:
        """Resolve voices and character metadata for one backend and active bundle."""
        if backend.uses_voice_bundles:
            loader = default_loader()
            if loader is not None:
                voices = loader.bundle.voice_order
                configured_default = loader.bundle.metadata.get("default_voice")
                default_voice = (
                    configured_default
                    if isinstance(configured_default, str)
                    else backend.default_voice
                )
                current_voice = (
                    preferred_voice
                    if preferred_voice in voices
                    else default_voice
                    if default_voice in voices
                    else voices[0]
                    if voices
                    else None
                )
                return (
                    voices,
                    current_voice,
                    bundle_character_name(loader.bundle),
                    persona_metadata_from_manifest(loader.bundle.metadata),
                    bundle_matches_default_pack_checksum(loader.bundle.path),
                )

        voices = tuple(backend.voices)
        current_voice = (
            preferred_voice
            if preferred_voice in voices
            else backend.default_voice
            if backend.default_voice in voices
            else voices[0]
            if voices
            else None
        )
        return voices, current_voice, None, None, True

    @staticmethod
    def _load_backend_voice_runtime(
        backend: CeluneBackend,
        voice: str,
    ) -> tuple[Optional[PreTrainedModel], str]:
        """Load the TTS runtime for one backend and voice without disturbing the previous runtime."""
        model_name = backend.model_id_for_voice(voice)
        model = cast(PreTrainedModel, backend.load_model(model_name))
        backend.model = model
        return model, model_name

    def _hot_reload_backend(
        self,
        backend_spec: CoreBackendSpec,
        preferred_voice: Optional[str] = None,
    ) -> bool:
        """Switch backend while reserving the shared model lifecycle resource."""
        acquired, component_lease = self._acquire_model_loading_lease(
            f"backend-reload:{time.monotonic_ns()}"
        )
        if not acquired:
            self._release_unstarted_reload()
            return False
        try:
            return self._hot_reload_backend_impl(backend_spec, preferred_voice)
        finally:
            if component_lease is not None:
                component_lease.release()

    def _hot_reload_backend_impl(
        self,
        backend_spec: CoreBackendSpec,
        preferred_voice: Optional[str] = None,
    ) -> bool:
        """Synchronously switch to a new backend family with rollback on failure."""
        # noinspection PyProtectedMember
        snapshot: Optional[Celune._ReloadSnapshot] = None
        candidate_kwargs: dict[str, JSONSerializable] = {}
        candidate_backend: Optional[CeluneBackend] = None
        candidate_vc_backend: Optional[CeluneVCBackend] = None
        candidate_model: Optional[PreTrainedModel] = None
        candidate_voice: Optional[str] = None
        requested_name = (
            backend_spec.name
            if isinstance(backend_spec, (CeluneBackend, CeluneVCBackend))
            else str(backend_spec)
        )

        try:
            snapshot = self._capture_reload_snapshot()
            preset = resolve_vram_preset(self.config)
            backend_target, normalized_backend_spec = _core_backend_target(
                backend_spec,
                self.log_callback,
                self.input_mode,
            )
            if backend_target == "tts":
                candidate_kwargs = self._backend_reload_kwargs(
                    cast(TTSBackendSpec, normalized_backend_spec)
                )
            self.log(
                string(
                    "celune.switching_backend",
                    app_name=APP_NAME,
                    backend=requested_name,
                )
            )
            self._ready_announced = False
            self.status_callback(string("status.reloading_backend"))
            self.progress_callback(None, None)
            self.cur_state = "reloading"

            previous_backend = self.backend
            previous_vc_backend = self.vc_backend
            previous_voice = snapshot.current_voice

            if backend_target == "tts":
                candidate_backend = self._resolve_tts_backend(
                    cast(TTSBackendSpec, normalized_backend_spec),
                    log=self.log_callback,
                    fatal=self.fatal,
                    **candidate_kwargs,
                )
                if not self._track_reload_backend(candidate_backend):
                    _close_backend(candidate_backend)
                    return False
                candidate_backend.bind_fatal(self.fatal)
                self._validate_backend_against_preset(candidate_backend, preset)
                if candidate_backend.uses_voice_bundles:
                    candidate_backend.validate_refs()
                if previous_vc_backend is not None:
                    _dispose_backend(previous_vc_backend)
                if previous_backend is not candidate_backend:
                    _dispose_backend(previous_backend)
                candidate_backend.preload_models()

                (
                    candidate_voices,
                    candidate_voice,
                    candidate_character,
                    candidate_persona,
                    candidate_bundle_is_default,
                ) = self._resolve_voice_state(candidate_backend, preferred_voice)
                if candidate_voice is None:
                    raise BackendError("no voices found")

                model, model_name = self._load_backend_voice_runtime(
                    candidate_backend,
                    candidate_voice,
                )
                candidate_model = model

                if not self._warmup(
                    fatal_on_failure=False,
                    backend=candidate_backend,
                    model=model,
                    voice=candidate_voice,
                ):
                    self._raise_warmup_error("warmup failed after backend reload")

                self.backend = candidate_backend
                self.vc_backend = None
                self._vc_backend_spec = None
                self.voice_conversion_backend = ""
                self.input_mode = "text_to_speech"
                self.tts_backend = candidate_backend.name
                self.model = model
                self.model_name = model_name
                self.voices = candidate_voices
                self.current_voice = candidate_voice
                self.current_character = candidate_character
                self.current_character_persona = candidate_persona
                self.voice_bundle_is_default = candidate_bundle_is_default
                self._backend_spec = (
                    cast(Union[str, type[CeluneBackend]], normalized_backend_spec)
                    if not isinstance(normalized_backend_spec, CeluneBackend)
                    else type(candidate_backend)
                )
                self._backend_kwargs = dict(candidate_kwargs)
            else:
                candidate_vc_backend = self._resolve_vc_backend(
                    cast(VCBackendSpec, normalized_backend_spec),
                    log=self.log_callback,
                )
                if not self._track_reload_backend(candidate_vc_backend):
                    _close_backend(candidate_vc_backend)
                    return False
                if available("pitch_shift", obj=candidate_vc_backend):
                    candidate_vc_backend.pitch_shift = self.vc_pitch_shift
                if available("f0_condition", obj=candidate_vc_backend):
                    candidate_vc_backend.f0_condition = self.vc_f0_condition
                if previous_backend is not None:
                    _dispose_backend(previous_backend)
                if (
                    previous_vc_backend is not None
                    and previous_vc_backend is not candidate_vc_backend
                ):
                    _dispose_backend(previous_vc_backend)
                candidate_vc_backend.preload_models()

                self.vc_backend = candidate_vc_backend
                self._vc_backend_spec = (
                    cast(
                        Union[str, type[CeluneVCBackend]],
                        normalized_backend_spec,
                    )
                    if not isinstance(normalized_backend_spec, CeluneVCBackend)
                    else type(candidate_vc_backend)
                )
                self.voice_conversion_backend = candidate_vc_backend.name
                self.input_mode = "voice_conversion"
                self.model = None
                self.model_name = ""

            self._release_reload_snapshot(snapshot)
            if backend_target == "tts" and self.use_normalization:
                self._unload_normalizer_components()
                self.load_normalizer()
            elif backend_target == "vc":
                self._unload_normalizer_components()
            self.loaded = True
            if candidate_backend is not None:
                if candidate_voice is None:
                    raise BackendError("no voices found")
                self.voice_changed_callback(candidate_voice)
                if previous_voice != candidate_voice:
                    self._emit_event(
                        "voice_changed",
                        VoiceChangedEvent(
                            celune=self,
                            old_voice=previous_voice or candidate_voice,
                            new_voice=candidate_voice,
                        ),
                    )
            self.log(string("celune.switched_backend", backend=requested_name))
            self.progress_callback(1, 1)
            self.cur_state = "idle"
            self.status_callback(string("status.idle"))
            return True
        except Exception as error:
            self.log(
                format_error_message(
                    tagged_string("celune.reload_error", "RELOAD ERROR"),
                    error,
                    self.log_level,
                ),
                "error",
            )
            if self.exit_requested:
                if candidate_backend is not None:
                    _close_backend(candidate_backend)
                if candidate_vc_backend is not None:
                    _close_backend(candidate_vc_backend)
                return False
            self.status_callback(string("status.restoring_backend"))
            self.progress_callback(None, None)
            if (
                candidate_backend is not None
                and snapshot is not None
                and candidate_backend is not snapshot.backend
            ):
                _shutdown_backend(candidate_backend, release_cuda_cache=True)
            elif (
                candidate_model is not None
                and snapshot is not None
                and candidate_model is not snapshot.model
            ):
                _release_loaded_object(candidate_model)
            if (
                candidate_vc_backend is not None
                and snapshot is not None
                and candidate_vc_backend is not snapshot.vc_backend
            ):
                _shutdown_backend(candidate_vc_backend, release_cuda_cache=True)
            rollback_succeeded = True
            if snapshot is not None:
                try:
                    if snapshot.input_mode == "voice_conversion":
                        if (
                            snapshot.loaded
                            and snapshot.restorable_vc_backend_spec is not None
                        ):
                            self._rebuild_reload_snapshot_vc_runtime(snapshot)
                        self._restore_reload_snapshot(snapshot)
                    elif snapshot.backend.model is None and snapshot.loaded:
                        self._rebuild_reload_snapshot_runtime(snapshot)
                        self._restore_reload_snapshot(snapshot)
                    else:
                        self._restore_reload_snapshot(snapshot)
                except Exception as restore_error:
                    rollback_succeeded = False
                    self.log(
                        format_error_message(
                            string("celune.backend_restore_error"),
                            restore_error,
                            self.log_level,
                        ),
                        "error",
                    )
                    self._restore_reload_snapshot(snapshot)
                    self.model = None
                    self.backend.model = None
                    if self.input_mode == "voice_conversion":
                        self.vc_backend = None
                        self._vc_backend_spec = None
                        self.voice_conversion_backend = ""
                    self.loaded = False
                    self.model_name = ""
                    self.cur_state = "idle"
            else:
                self.cur_state = "idle"
            self._last_warmup_error = None
            self.status_callback(string("status.idle"))
            self.progress_callback(1, 1)
            if rollback_succeeded:
                self.log(string("celune.backend_restore_failed"), "warning")
            self.error_callback(string("status.could_not_reload", app_name=APP_NAME))
            return False
        finally:
            self._clear_reload_backend(
                candidate_backend
                if candidate_backend is not None
                else candidate_vc_backend
            )
            self._reload_pending = False
            self._model_ready.set()
            self._last_component_busy = None
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def _track_reload_backend(
        self,
        backend: Union[CeluneBackend, CeluneVCBackend],
    ) -> bool:
        """Publish a reload candidate to shutdown before it performs work."""
        with self._model_lock:
            if self._closed:
                return False
            self._reload_backend = backend
            return True

    def _clear_reload_backend(
        self,
        backend: Optional[Union[CeluneBackend, CeluneVCBackend]],
    ) -> None:
        """Drop one reload candidate without clearing a newer candidate."""
        if backend is None:
            return
        with self._model_lock:
            if self._reload_backend is backend:
                self._reload_backend = None

    def _hot_reload_cevoice(
        self,
        bundle: Optional[Union[str, Path]],
        preferred_voice: Optional[str] = None,
    ) -> bool:
        """Switch voice bundle while reserving the model lifecycle resource."""
        acquired, component_lease = self._acquire_model_loading_lease(
            f"cevoice-reload:{time.monotonic_ns()}"
        )
        if not acquired:
            self._release_unstarted_reload()
            return False
        try:
            return self._hot_reload_cevoice_impl(bundle, preferred_voice)
        finally:
            if component_lease is not None:
                component_lease.release()

    def _hot_reload_cevoice_impl(
        self,
        bundle: Optional[Union[str, Path]],
        preferred_voice: Optional[str] = None,
    ) -> bool:
        """Synchronously switch to a new CEVOICE bundle with rollback on failure."""
        # noinspection PyProtectedMember
        snapshot: Optional[Celune._ReloadSnapshot] = None
        previous_bundle = active_bundle_path()
        loaded_model: Optional[PreTrainedModel] = None

        try:
            snapshot = self._capture_reload_snapshot()
            previous_loader = default_loader()
            previous_bundle = (
                Path(previous_loader.bundle.path)
                if previous_loader is not None
                else active_bundle_path()
            )
            self.log(string("celune.reloading_character", app_name=APP_NAME))
            self._ready_announced = False
            self.status_callback(string("status.reloading_character"))
            self.progress_callback(None, None)
            self.cur_state = "reloading"

            select_voice_bundle(bundle)
            if self.backend.uses_voice_bundles:
                self.backend.validate_refs()

            (
                candidate_voices,
                candidate_voice,
                candidate_character,
                candidate_persona,
                candidate_bundle_is_default,
            ) = self._resolve_voice_state(self.backend, preferred_voice)
            if candidate_voice is None:
                raise BackendError("no voices found in current CEVOICE")

            self.voices = candidate_voices
            self.current_voice = candidate_voice
            self.current_character = candidate_character
            self.current_character_persona = candidate_persona
            self.voice_bundle_is_default = candidate_bundle_is_default

            if snapshot.current_character != self.current_character or str(
                previous_bundle
            ) != str(active_bundle_path()):
                self._reset_persona_conversation()

            if not self._is_voice_conversion_mode():
                model, model_name = self._load_backend_voice_runtime(
                    self.backend,
                    candidate_voice,
                )
                loaded_model = model
                previous_model = snapshot.model
                self.model = model
                self.model_name = model_name
                if not self._warmup(fatal_on_failure=False):
                    self._raise_warmup_error("warmup failed after CEVOICE reload")
                if previous_model is not None and previous_model is not model:
                    _release_loaded_object(cast(ReleasableObject, previous_model))

            self.loaded = True
            self.voice_changed_callback(candidate_voice)
            self._emit_character_event_transition(
                snapshot.current_character,
                str(previous_bundle),
                self.current_character,
                str(active_bundle_path()),
                self.voice_bundle_is_default,
            )
            if snapshot.current_voice != candidate_voice:
                self._emit_event(
                    "voice_changed",
                    VoiceChangedEvent(
                        celune=self,
                        old_voice=snapshot.current_voice or candidate_voice,
                        new_voice=candidate_voice,
                    ),
                )
            self.log(
                string(
                    "celune.switched_character",
                    character=(bundle if bundle is not None else previous_bundle.name),
                )
            )
            self.progress_callback(1, 1)
            self.cur_state = "idle"
            self.status_callback(string("status.idle"))
            return True
        except Exception as error:
            self.log(
                format_error_message(
                    tagged_string("celune.reload_error", "RELOAD ERROR"),
                    error,
                    self.log_level,
                ),
                "error",
            )
            if (
                loaded_model is not None
                and snapshot is not None
                and loaded_model is not snapshot.model
            ):
                _release_loaded_object(loaded_model)
            select_voice_bundle(previous_bundle)
            if snapshot is not None:
                self._restore_reload_snapshot(snapshot)
            else:
                self.cur_state = "idle"
            self.status_callback(string("status.idle"))
            self.progress_callback(1, 1)
            self.log(
                string("celune.character_restore_failed"),
                "warning",
            )
            return False
        finally:
            self._reload_pending = False
            self._model_ready.set()
            self._last_component_busy = None
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def _release_unstarted_reload(self) -> None:
        """Restore readiness after a reload could not acquire model ownership."""
        self._reload_pending = False
        self._model_ready.set()
        self._last_component_busy = None
        self.change_input_state_callback(locked=False)
        self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def _raise_warmup_error(self, message: str) -> None:
        """Raise a Celune warmup error while preserving any original cause."""
        if self._last_warmup_error is not None:
            raise WarmupError(message) from self._last_warmup_error
        raise WarmupError(message)

    raise_warmup_error = _raise_warmup_error

    def _wait_until_idle(
        self,
        timeout: float = 30.0,
        *,
        wait_for_speech: Optional[bool] = None,
    ) -> bool:
        """Wait until the model and speech pipeline are ready.

        Args:
            timeout: Maximum time to wait for readiness.
            wait_for_speech: Whether active speech must finish before returning.
        """
        # don't wait a timeout while Celune is downloading a model
        ok = self._model_ready.wait(timeout=timeout)
        if not ok:
            self.log(string("celune.ready_wait_timeout"), "warning")
            self.log(string("celune.ready_wait_reason"), "warning")
            self.log(string("celune.ready_wait_not_fatal"), "warning")
            return False

        if not self.loaded:
            self.log(string("celune.model_unloaded_while_waiting"), "warning")
            return False

        if wait_for_speech is None:
            wait_for_speech = self._speech_playback_active()

        if wait_for_speech:
            ok = self._playback_done.wait(timeout=timeout)
            if not ok:
                self.log(
                    string("celune.playback_idle_timeout"),
                    "warning",
                )
                return False

        with self._say_lock:
            return (not self.locked) and self.loaded

    wait_until_idle = _wait_until_idle

    @contextlib.contextmanager
    def with_backend(
        self,
        backend_spec: CoreBackendSpec,
        timeout: float = 30.0,
    ):
        """Temporarily switch Celune to another backend within a context block.

        Args:
            backend_spec: The backend name, type, or instance to activate temporarily.
            timeout: Maximum seconds to wait while switching or restoring the backend.

        Raises:
            BackendError: Celune could not switch to or restore the requested backend.
        """
        restore_backend = self._restorable_active_backend_spec()
        restore_voice = self.current_voice
        if not self.wait_until_idle(timeout=timeout):
            raise BackendError("timed out switching backend")
        if not self._hot_reload_backend(backend_spec, self.current_voice):
            raise BackendError("failed to switch backend")

        try:
            yield self
        finally:
            self.wait_until_idle(timeout=timeout)
            if not self._hot_reload_backend(restore_backend, restore_voice):
                raise BackendError("failed to restore old backend")

    @contextlib.contextmanager
    def with_cevoice(
        self,
        bundle: Optional[Union[str, Path]],
        timeout: float = 30.0,
    ):
        """Temporarily switch Celune to another CEVOICE bundle within a context block.

        Args:
            bundle: The CEVOICE bundle name or path to activate temporarily.
            timeout: Maximum seconds to wait while switching or restoring the CEVOICE pack.

        Raises:
            BackendError: Celune could not switch to or restore the requested CEVOICE pack.
        """
        previous_loader = default_loader()
        restore_bundle = (
            Path(previous_loader.bundle.path)
            if previous_loader is not None
            else active_bundle_path()
        )
        restore_voice = self.current_voice
        if not self.wait_until_idle(timeout=timeout):
            raise BackendError("timed out switching character")
        if not self._hot_reload_cevoice(bundle, None):
            raise BackendError("failed to switch character")

        try:
            yield self
        finally:
            self.wait_until_idle(timeout=timeout)
            if not self._hot_reload_cevoice(restore_bundle, restore_voice):
                raise BackendError("failed to restore character")

    def _api_settings(self) -> tuple[bool, str, int, Optional[str], int]:
        """Resolve API settings from Celune's configuration."""
        api_config = config_value(self.config, "api", {})

        if isinstance(api_config, bool):
            return api_config, "0.0.0.0", 2060, None, 60

        if api_config is None:
            return False, "0.0.0.0", 2060, None, 60

        if not isinstance(api_config, dict):
            return bool(api_config), "0.0.0.0", 2060, None, 60

        enabled = bool(api_config.get("enabled", True))
        host = str(api_config.get("host", "0.0.0.0"))
        token_value = api_config.get("token")
        token = str(token_value).strip() if token_value is not None else None
        if not token:
            self.log(
                string("celune.no_api_token", app_name=APP_NAME),
                "warning",
            )
            token = None
            host = "127.0.0.1"
        try:
            port = _config_int(api_config.get("port", 2060), 2060)
        except (TypeError, ValueError):
            invalid_port = api_config.get("port", 2060)
            self.log(
                string("celune.api_port_invalid", app_name=APP_NAME, port=invalid_port),
                "warning",
            )
            port = 2060

        if not 1 <= port <= 65535:
            self.log(
                string("celune.api_port_out_of_range", app_name=APP_NAME, port=port),
                "warning",
            )
            port = 2060

        try:
            requests_per_minute = _config_int(
                api_config.get("rate_limit_per_minute", 60),
                60,
            )
        except (TypeError, ValueError):
            invalid_ratelimit = api_config.get("rate_limit_per_minute", 60)
            self.log(
                string(
                    "celune.api_rate_limit_invalid",
                    app_name=APP_NAME,
                    rate=invalid_ratelimit,
                ),
                "warning",
            )
            requests_per_minute = 60

        return enabled, host, port, token, max(0, requests_per_minute)

    api_settings = _api_settings

    # as of CeluneNorm 2.0, normalization ACTUALLY works with long inputs

    def stop_live_audio(self) -> None:
        """Reset any state held by the active live voice-conversion backend."""
        stop_live_audio_input(self)

    def play(
        self,
        sound_path: str,
        keep: bool = False,
        volume: float = 1.0,
        on_started: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Play a sound via Celune's pipeline.

        Args:
            sound_path: The path to the audio file to play.
            keep: Whether to prepend this SFX to the next saved utterance.
            volume: How loud should the SFX be played at.
            on_started: Optional callback invoked immediately before SFX is queued.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise ``False``.
        """
        if self.test_finished or self.backend_mode == "agent_test":
            return False
        return play_pipeline(
            self,
            sound_path,
            keep=keep,
            volume=volume,
            on_started=on_started,
        )

    def play_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        label: str = "uploaded SFX",
        keep: bool = False,
    ) -> bool:
        """Play decoded audio via Celune's pipeline.

        Args:
            audio: Decoded mono or stereo audio.
            sample_rate: Source sample rate for the decoded audio.
            label: Human-readable label for logs and status.
            keep: Whether to prepend this SFX to the next saved utterance.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise ``False``.
        """
        if self.test_finished or self.backend_mode == "agent_test":
            return False
        return queue_sfx_audio(self, audio, sample_rate, label, keep=keep)

    def close(self) -> None:
        """Shut off Celune and release loaded runtime state."""
        self._exit_requested = True
        with self._model_lock:
            if self._closed:
                return
            self._closed = True

        # call this or else you get a zombified Celune that can't proceed with her exit procedures
        self._abort_backend_operations()
        with self._shutdown_runtime_lock():
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
            self._emit_event("shutdown", ShutdownEvent(celune=self))
            try:
                close_pipeline(self)
                wake_background_thread = self._wake_background_thread
                if (
                    wake_background_thread is not None
                    and wake_background_thread is not threading.current_thread()
                ):
                    wake_background_thread.join(timeout=2)
                self._close_agent_tool_selector()
                self._unload_persona_state()
                with self._model_lock:
                    self.unload_runtime_state(include_normalizer=True)
            finally:
                with contextlib.suppress(Exception):
                    close_default_loader()
                with contextlib.suppress(Exception):
                    self._cleanup_residual_temp_data(temp_data_dir())
                Celune._instance = None
                self.log("[ENGINE] close complete", loglevel="debug")

    @contextlib.contextmanager
    def _shutdown_runtime_lock(self) -> Generator[None, None, None]:
        """Run shutdown cleanup without waiting for a reload operation."""
        acquired = self._async_runtime_lock.acquire(blocking=False)
        try:
            yield
        finally:
            if acquired:
                self._async_runtime_lock.release()

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

    def _close_agent_tool_selector(self) -> None:
        """Release the optional loaded Needle handler during engine shutdown."""
        selector = self._agent_needle_selector
        if isinstance(selector, NeedleToolSelector):
            with contextlib.suppress(Exception):
                selector.handler.close()
        self._agent_needle_selector = None

    def fatal(self) -> None:
        """Mark Celune state as fatal and prevent further operations."""
        self.loaded = False
        self.cur_state = "error"
        self.glow.fatal()
        if not self._try_play_signal("error"):
            self.log(
                string("ui.error_signal_unavailable"),
                "warning",
                loglevel="verbose",
            )

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        return split_text(self, text)

    async def _pipeline_jobs(self) -> None:
        """Run Celune's speech pipeline workers as async jobs."""
        playback_task = asyncio.create_task(playback_worker_job(self))
        if self._is_voice_conversion_mode():
            await playback_task
            return

        generation_task = asyncio.create_task(generation_worker_job(self))
        await asyncio.gather(generation_task, playback_task)

    def _run_pipeline_jobs(self) -> None:
        """Start Celune's async speech pipeline inside one engine thread."""
        asyncio.run(self._pipeline_jobs())


def _install_celune_methods() -> None:
    """Install split engine methods after the concrete class exists."""
    from . import (
        conversation,
        loader as loader_methods,
        pipeline as pipeline_methods,
        playback as playback_methods,
        sleep,
        speech,
        vc as vc_methods,
        voice as voice_methods,
    )
    from .agent import core as agent_methods

    pipeline_methods.install_engine(Celune)
    playback_methods.install_engine(Celune)
    speech.install_engine(Celune)
    conversation.install(Celune)
    vc_methods.install(Celune)
    sleep.install(Celune)
    voice_methods.install(Celune)
    loader_methods.install(Celune)
    agent_methods.install(Celune)


_install_celune_methods()
