# SPDX-License-Identifier: Apache-2.0
"""Celune's backend layer."""

import gc
import os
import time
import queue
import shutil
import asyncio
import threading
import contextlib
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Union, Optional, cast

import torch
import numpy as np
import numpy.typing as npt
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import __version__
from .chroma import AudioRGBGlow
from .vc import clamp_vc_pitch_shift
from .locks import ComponentLockLease
from .typing.modes import BackendMode
from .agent.runtime import AgentRuntime
from .typing.backends import BackendModel
from .extensions.base import CeluneContext
from .agent.routing import AgentInputRouter
from .agent.needle import NeedleToolSelector
from .agent.persona import PersonaAgentBridge
from .paths import project_root, temp_data_dir
from .typing.pipeline import SpeechStreamQueue
from .extensions.events import EventDispatcher
from .typing.aliases import LogLevel, AudioChunk
from .typing.common import JSON, JSONSerializable
from .pipeline import (
    say as say_pipeline,
)
from .typing.events import EventName, EventPayload
from .persona.emotion import PersonaEmotionAnalyzer
from .pipeline import (
    play as play_pipeline,
)
from .constants import APP_NAME, NORMALIZER_MODEL_ID
from .pipeline import (
    close as close_pipeline,
)
from .pipeline import (
    think as think_pipeline,
)
from .extensions.manager import CeluneExtensionManager
from .i18n import string, set_locale, get_system_locale
from .runtime import validate_runtime, log_runtime_banner
from .pipeline import (
    say_async as say_pipeline_async,
)
from .dataclasses.pipeline import AudioOutput, AudioInputRequest
from .backends.tts import BACKENDS, CeluneBackend, resolve_backend
from .modeling import normalizer_device, load_normalizer_components
from .pipeline import (
    force_stop_speech as force_stop_pipeline,
)
from .backends.vc import VC_BACKENDS, CeluneVCBackend, resolve_vc_backend
from .config import Config, config_bool, config_value, normalize_log_level
from .utils import discard, format_error, custom_assert, format_number, is_port_usable
from .modes import (
    OperationMode,
    mode_allows_persona,
    resolve_operation_mode,
)
from .dataclasses.properties import (
    bind_constant_properties,
    bind_forwarded_properties,
)
from .typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from .exceptions import (
    WarmupError,
    BackendError,
    NotAvailableError,
    RuntimeCheckError,
    NeedleSelectionError,
)
from .agent.tools import (
    agent_test_tools,
    production_agent_tools,
    agent_test_tool_schemas,
    production_agent_tool_schemas,
)
from .vram import (
    QWEN3_0_6B_MODEL,
    VramPreset,
    backend_allowed,
    resolve_vram_preset,
    resolve_backend_name,
    validate_vram_preset,
)
from .persona.impl import (
    PersonaClient,
    persona_enabled,
    persona_model_id,
    persona_is_available,
    persona_quantization,
    create_persona_client,
)
from .dataclasses.events import (
    ReadyEvent,
    ShutdownEvent,
    StateChangedEvent,
    VoiceChangedEvent,
    CharacterLoadedEvent,
    CharacterChangedEvent,
    CharacterUnloadedEvent,
)
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
from .cevoice import (
    CEVoicePersona,
    default_loader,
    active_bundle_path,
    resolve_bundle_path,
    select_voice_bundle,
    close_default_loader,
    bundle_character_name,
    is_protected_temp_path,
    announce_default_bundle,
    persona_metadata_from_manifest,
    bundle_matches_default_pack_checksum,
)
from .typing.celune import (
    Generative,
    VCBackendSpec,
    TTSBackendSpec,
    CaptionCallback,
    CoreBackendSpec,
    MessageCallback,
    ProgressCallback,
    ReleasableObject,
    InputStateCallback,
    NormalizerTokenizer,
    CeluneStateAccessors,
    CaptionTimingCallback,
    VoiceLockStateCallback,
    _BundleWithPath,
)
from .pipeline import (
    split_text,
    clear_queue,
    play_signal,
    close_stream,
    queue_speech,
    queue_sfx_audio,
    acquire_pipeline,
    release_pipeline,
    handle_audio_input,
    queue_speech_async,
    convert_audio_input,
    playback_worker_job,
    generation_worker_job,
    stop_live_audio_input,
    deliver_persona_response,
    saved_output_speech_seconds,
)
from .typing.agent import (
    ToolCall,
    AgentRoute,
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
    AgentCancellationReason,
    AgentInputClassification,
    AgentToolExecutionStatus,
    AgentClassificationResult,
    AgentClassificationFailure,
    AgentClassificationFailureKind,
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


def _configured_pipeline_queue_size(config: Config) -> int:
    """Return the bounded audio-queue size configured for pipeline playback."""
    value = config.get("pipeline_cpu", {})
    if not isinstance(value, dict):
        return 8
    enabled = value.get("enabled", True)
    if isinstance(enabled, bool) and not enabled:
        return 0

    queue_size = value.get("max_queue_items", 8)
    if isinstance(queue_size, bool):
        return 8
    try:
        return min(128, max(1, _config_int(queue_size, 8)))
    except (TypeError, ValueError, OverflowError):
        return 8


def _configured_vc_pitch_shift(config: Config) -> int:
    """Return the configured default pitch shift for VC backends."""
    env_value = os.getenv("CELUNE_VC_PITCH_SHIFT")
    if env_value is not None and env_value.strip():
        return clamp_vc_pitch_shift(_config_int(env_value.strip(), 0))

    configured_value = config_value(config, "voice_conversion_pitch_shift")
    if configured_value is None:
        configured_value = config_value(config, "vc_pitch_shift")
    return clamp_vc_pitch_shift(_config_int(configured_value, 0))


def _configured_vc_f0_condition(config: Config) -> bool:
    """Return whether VC should run in Seed-VC singing mode by default."""
    return config_bool(
        config,
        "CELUNE_VC_F0_CONDITION",
        "voice_conversion_f0_condition",
        False,
    ) or config_bool(config, "CELUNE_VC_F0_CONDITION", "vc_f0_condition", False)


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


class Celune(CeluneStateAccessors):
    """The character engine for Celune."""

    _instance: Optional["Celune"] = None

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
    ) -> None:
        if Celune._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")
        if backend_mode not in {"normal", "ui_test", "agent_test"}:
            raise ValueError(f"unknown Celune backend mode: '{backend_mode}'")

        self._startup_log_buffer: list[tuple[str, str, LogLevel]] = []
        self._startup_banner_emitted = False
        self._startup_log_sink = log_callback or self._noop_message
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
        self._pipeline_state = CelunePipelineState(
            audio_queue=queue.Queue(maxsize=_configured_pipeline_queue_size(config))
        )
        self._audio_state = CeluneAudioState()
        self._runtime_state = CeluneRuntimeState()
        self._async_runtime_lock = threading.Lock()
        self._voice_reload_guard = threading.Lock()
        self._voice_reload_active = False
        self._pipeline_state.model_ready.set()
        self._pipeline_state.playback_done.set()

        self.config = config
        self._runtime_state.backend_mode = backend_mode
        self._isolated_backends = config_bool(
            config,
            "CELUNE_ISOLATED_BACKENDS",
            "isolated_backends",
        )
        set_locale(_configured_locale(config) or get_system_locale())
        self.mode: OperationMode = resolve_operation_mode(config)
        if backend_mode == "agent_test":
            self._agent_tools = agent_test_tools(self)
            self._agent_tool_schemas = agent_test_tool_schemas()
        else:
            local_management = config_bool(
                config,
                "CELUNE_LOCAL_MANAGEMENT",
                "agent_local_management",
            )
            self._agent_tools = production_agent_tools(
                self,
                include_local_management=local_management,
            )
            self._agent_tool_schemas = production_agent_tool_schemas(
                include_local_management=local_management,
            )
            if local_management:
                self.log(string("agent.unsandboxed_warning"), "warning")
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
            _configured_vc_pitch_shift(config)
            if vc_pitch_shift is None
            else clamp_vc_pitch_shift(_config_int(vc_pitch_shift, 0))
        )
        self.vc_f0_condition = (
            _configured_vc_f0_condition(config)
            if vc_f0_condition is None
            else vc_f0_condition
        )
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

        if not isinstance(tts_backend, CeluneBackend) and (
            (
                isinstance(tts_backend, str)
                and tts_backend.strip().lower() == "gpt-sovits"
            )
            or (
                isinstance(tts_backend, type)
                and getattr(tts_backend, "name", "").strip().lower() == "gpt-sovits"
            )
        ):
            backend_kwargs["root"] = _config_str(
                config_value(config, "gpt_sovits_root")
            )
            backend_kwargs["variant"] = _config_str(
                config_value(config, "gpt_sovits_variant")
            )
            backend_kwargs["t2s_weights_path"] = _config_str(
                config_value(config, "gpt_sovits_t2s_weights_path")
            )

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
                f"internal backend error: {format_error(e, log_level)}"
            ) from e

        if vc_backend is None and self.input_mode == "voice_conversion":
            vc_backend = _config_str(
                config_value(config, "voice_conversion_backend")
            ) or _config_str(config_value(config, "vc_backend"))
            if vc_backend is None:
                vc_backend = "passthrough"

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
                if hasattr(self.vc_backend, "pitch_shift"):
                    self.vc_backend.pitch_shift = self.vc_pitch_shift
                if hasattr(self.vc_backend, "f0_condition"):
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
                f"internal voice-conversion backend error: {format_error(e, log_level)}"
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
                cleanup_errors.append(str(exc))
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
            errors.append(str(exc))

        try:
            close_stream(self, abort=True)
        except Exception as exc:
            errors.append(str(exc))

        current_thread = threading.current_thread()
        for worker in (self.generation_thread, self.playback_thread):
            if worker is not None and worker is not current_thread:
                try:
                    worker.join(timeout=2)
                except Exception as exc:
                    errors.append(str(exc))

        persona_thread = self._persona_thread
        if persona_thread is not None and persona_thread is not current_thread:
            try:
                persona_thread.join(timeout=2)
            except Exception as exc:
                errors.append(str(exc))

        try:
            self._close_agent_tool_selector()
        except Exception as exc:
            errors.append(str(exc))

        try:
            self.component_locks.release_all()
            self._pipeline_lock_owner = None
            self.locked = True
            self.playback_done.set()
        except Exception as exc:
            errors.append(str(exc))

        try:
            self.glow.leave()
            self.glow.finished.wait(timeout=5)
        except Exception as exc:
            errors.append(str(exc))
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

    def _acquire_model_loading_lease(
        self,
        operation_id: str,
    ) -> tuple[bool, Optional[ComponentLockLease]]:
        """Reserve the shared model lifecycle resource for one operation."""
        manager = getattr(self, "component_locks", None)
        if manager is None:
            return True, None

        owner = ComponentLockOwner(operation_id=operation_id)
        acquisition, lease = manager.try_acquire_lease(
            (ComponentLockRequirement(ComponentLockName.MODEL_LOADING),),
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
        acquired, component_lease = self._acquire_model_loading_lease(
            f"persona-load:{id(vision)}"
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
            self.log(format_error(e, self.log_level), "warning")
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
        """Resolve one TTS backend using the configured process isolation mode."""
        if self._isolated_backends:
            return resolve_backend(
                backend_spec,
                isolated=True,
                **backend_kwargs,
            )
        return resolve_backend(backend_spec, **backend_kwargs)

    def _recreate_vc_backend(self) -> bool:
        """Rebuild the VC backend from its original constructor recipe."""
        if self._vc_backend_spec is None:
            return False

        candidate_backend = self._resolve_vc_backend(
            self._vc_backend_spec,
            log=self.log_callback,
        )
        if hasattr(candidate_backend, "pitch_shift"):
            candidate_backend.pitch_shift = self.vc_pitch_shift
        if hasattr(candidate_backend, "f0_condition"):
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
        """Resolve one VC backend using the configured process isolation mode."""
        return resolve_vc_backend(
            backend_spec,
            log=log,
            isolated=self._isolated_backends,
        )

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
        elif (
            isinstance(backend_spec, str)
            and backend_spec.strip().lower() == "gpt-sovits"
        ) or (
            isinstance(backend_spec, type)
            and getattr(backend_spec, "name", "").strip().lower() == "gpt-sovits"
        ):
            backend_kwargs["root"] = _config_str(
                config_value(self.config, "gpt_sovits_root")
            )
            backend_kwargs["variant"] = _config_str(
                config_value(self.config, "gpt_sovits_variant")
            )
            backend_kwargs["t2s_weights_path"] = _config_str(
                config_value(self.config, "gpt_sovits_t2s_weights_path")
            )
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

        restored_vc_backend = self._resolve_vc_backend(
            snapshot.restorable_vc_backend_spec,
            log=self.log_callback,
        )
        if hasattr(restored_vc_backend, "pitch_shift"):
            restored_vc_backend.pitch_shift = self.vc_pitch_shift
        if hasattr(restored_vc_backend, "f0_condition"):
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
                if hasattr(candidate_vc_backend, "pitch_shift"):
                    candidate_vc_backend.pitch_shift = self.vc_pitch_shift
                if hasattr(candidate_vc_backend, "f0_condition"):
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
        except Exception as e:
            self.log(
                string("celune.reload_error", error=format_error(e, self.log_level)),
                "error",
            )
            self.status_callback(string("status.restoring_backend"))
            self.progress_callback(None, None)
            if (
                candidate_backend is not None
                and snapshot is not None
                and candidate_backend is not snapshot.backend
            ):
                _dispose_backend(candidate_backend)
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
                _dispose_backend(candidate_vc_backend)
            if snapshot is not None:
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
            else:
                self.cur_state = "idle"
            self._last_warmup_error = None
            self.status_callback(string("status.idle"))
            self.progress_callback(1, 1)
            self.log(
                string("celune.backend_restore_failed"),
                "warning",
            )
            return False
        finally:
            self._reload_pending = False
            self._model_ready.set()
            self._last_component_busy = None
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

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
        except Exception as e:
            self.log(
                string("celune.reload_error", error=format_error(e, self.log_level)),
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

    def unload_normalizer_state(self, release_cuda_cache: bool = True) -> None:
        """Unload only CeluneNorm components and release unused memory.

        Args:
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """
        self._unload_normalizer_components()
        gc.collect()

        if release_cuda_cache and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def _sleep_config(self) -> tuple[bool, int, dict[str, bool]]:
        """Return sleep enablement, timeout, and unload settings."""
        sleep_config = config_value(self.config, "sleep", {})
        if isinstance(sleep_config, bool):
            return (
                sleep_config,
                10,
                {
                    "persona": True,
                    "normalizer": True,
                    "tts": False,
                    "vc": False,
                },
            )

        if not isinstance(sleep_config, dict):
            sleep_config = {}

        unload_config = sleep_config.get("unload", {})
        if not isinstance(unload_config, dict):
            unload_config = {}

        try:
            timeout = _config_int(sleep_config.get("timeout", 10), 10)
        except (TypeError, ValueError):
            timeout = 10

        unload_tts = bool(unload_config.get("tts", False))

        return (
            bool(sleep_config.get("enabled", False)),
            max(1, timeout),
            {
                "persona": bool(unload_config.get("persona", True)),
                "normalizer": bool(unload_config.get("normalizer", True)),
                "tts": unload_tts,
                "vc": bool(unload_config.get("vc", unload_tts)),
            },
        )

    def sleep_enabled(self) -> bool:
        """Return whether automatic sleep mode is enabled.

        Returns:
            bool: Whether automatic sleep mode is enabled.
        """
        enabled, _, _ = self._sleep_config()
        return enabled

    def sleep_timeout_seconds(self) -> float:
        """Return the configured idle timeout in seconds.

        Returns:
            float: The configured idle timeout in seconds.
        """
        _, timeout_minutes, _ = self._sleep_config()
        return timeout_minutes * 60.0

    def enter_sleep_mode(self) -> bool:
        """Put Celune to sleep and unload models according to configuration.

        Returns:
            bool: Whether Celune was put to sleep.
        """
        enabled, _, unload = self._sleep_config()
        self.log(
            f"[SLEEP] enter requested enabled={enabled} sleeping={self.sleeping} "
            f"unload={unload}",
            loglevel="debug",
        )
        if not enabled or self.sleeping:
            return False

        with self.say_lock:
            if self.locked or self.cur_state in {"generating", "speaking", "reloading"}:
                return False
            self.sleeping = True
            self.loaded = False
            self.cur_state = "sleeping"
            self.glow.sleep()

        if not self._try_play_signal("sleeping"):
            self.log(
                "Could not play the sleeping signal.",
                "warning",
                loglevel="verbose",
            )

        self._ready_announced = False
        self.model_ready.clear()
        self.progress_callback(0, 1)

        with (
            self._wake_background_lock,
            self._model_lock,
        ):
            if unload["persona"]:
                self._unload_persona_state()

            if unload["tts"]:
                self.unload_runtime_state(
                    include_normalizer=unload["normalizer"],
                    include_vc=unload["vc"],
                    close_backends=True,
                    release_cuda_cache=False,
                )
                self.model_name = ""
            elif unload["normalizer"]:
                self.unload_normalizer_state(release_cuda_cache=False)

            if unload["vc"] and not unload["tts"] and self.vc_backend is not None:
                _dispose_backend(self.vc_backend, release_cuda_cache=False)

        self.model_ready.set()
        self.log("[SLEEP] enter complete model_ready=True", loglevel="debug")
        return True

    def wake_from_sleep(self) -> bool:
        """Wake Celune and reload anything unloaded by sleep mode.

        Returns:
            bool: Whether Celune was woken up from sleep.

        Raises:
            NotAvailableError: Celune has no valid model ID to reload after waking up.
            WarmupError: Celune cannot warm up after waking up.
        """
        with self._wake_lock:
            self.log(
                f"[SLEEP] wake requested sleeping={self.sleeping}",
                loglevel="debug",
            )
            if not self.sleeping:
                return True

            _, _, unload = self._sleep_config()
            self.model_ready.clear()
            self.status_callback(string("status.waking_up"))
            self.progress_callback(None, None)
            self.cur_state = "waking"

            try:
                with self._model_lock:
                    if self._is_voice_conversion_mode():
                        if self.vc_backend is None:
                            raise NotAvailableError(
                                "cannot wake without a configured voice conversion backend"
                            )
                        if unload["vc"] and self._recreate_vc_backend():
                            self.log("[SLEEP] Recreated VC backend", loglevel="verbose")
                        self.vc_backend.preload_models()
                    else:
                        active_voice = self.current_voice or (
                            self.voices[0] if self.voices else None
                        )
                        if active_voice is None:
                            raise NotAvailableError(
                                "cannot wake without an active voice"
                            )

                        if unload["tts"] or self.model is None:
                            if unload["tts"] and self._recreate_tts_backend():
                                self.log(
                                    "[SLEEP] Recreated TTS backend", loglevel="verbose"
                                )
                            model_id = self.backend.model_id_for_voice(active_voice)
                            self.log(
                                f"[SLEEP] Loading model: {model_id}",
                                loglevel="verbose",
                            )
                            self.model = self.backend.load_model(model_id)
                            self.model_name = model_id
                            if not self._warmup():
                                self._raise_warmup_error("warmup failed after sleep")

                    is_voice_conversion = self._is_voice_conversion_mode()
                    if (
                        not is_voice_conversion
                        and unload["persona"]
                        and persona_enabled(self.config)
                    ):
                        self.vision = self._persona_conn()
                        self.persona_ready = False

                    self.loaded = True
                    self.sleeping = False
                    self.cur_state = "idle"
                    self.glow.wake()

                self.progress_callback(1, 1)
                self.status_callback(string("status.idle"))
                self.change_input_state_callback(locked=False)
                self.change_voice_lock_state_callback(locked=len(self.voices) < 2)
                self.log("[SLEEP] wake complete state=idle", loglevel="debug")
                if not is_voice_conversion:
                    self._start_wake_background_jobs(unload)
                return True
            except Exception as e:
                self.fatal()
                self.log(
                    string("celune.wake_error", error=format_error(e, self.log_level)),
                    "error",
                )
                self.status_callback(
                    string("status.could_not_wake", app_name=APP_NAME), "error"
                )
                self.error_callback(string("status.could_not_wake", app_name=APP_NAME))
                self.progress_callback(0, 1)
                return False
            finally:
                self.model_ready.set()

    def _start_wake_background_jobs(self, unload: dict[str, bool]) -> None:
        """Start optional wake-up restoration after TTS becomes ready."""
        existing = self._wake_background_thread
        if existing is not None and existing.is_alive():
            return

        thread = threading.Thread(
            target=self._run_wake_background_jobs,
            args=(unload,),
            daemon=True,
        )
        self._wake_background_thread = thread
        thread.start()

    def _run_wake_background_jobs(self, unload: dict[str, bool]) -> None:
        """Restore optional wake-up resources without delaying TTS readiness."""
        try:
            with self._wake_background_lock:
                if self.exit_requested or self.sleeping:
                    return

                if (
                    unload["vc"]
                    and self.vc_backend is not None
                    and not self._is_voice_conversion_mode()
                ):
                    with self._model_lock:
                        if self.exit_requested or self.sleeping:
                            return
                        if self._recreate_vc_backend():
                            self.log("[SLEEP] Recreated VC backend", loglevel="verbose")
                        self.vc_backend.preload_models()

                if unload["normalizer"] and self.use_normalization:
                    self.load_normalizer()

                if not unload["persona"] or not persona_enabled(self.config):
                    return

                with self._model_lock:
                    vision = self.vision
                    if vision is None or self.persona_ready or self.persona_loading:
                        return
                    self.persona_loading = True
                self._load_persona_background(vision)
        except Exception as e:
            self.log(format_error(e, self.log_level), "error")
        finally:
            if self._wake_background_thread is threading.current_thread():
                self._wake_background_thread = None

    def set_voices(self, voices: tuple[str, ...]) -> None:
        """Configure Celune's voice information.

        Args:
            voices: The list of available voice names.
        """
        self.voices = voices

    def _reset_persona_conversation(self) -> None:
        """Clear Persona conversation context when its character changes."""
        self.persona_history.clear()
        self.persona_session_summary = ""
        self.persona_attachments.clear()

    def load_voice_bundle(self, bundle: Optional[Union[str, Path]] = None) -> bool:
        """Select and load a CEVOICE bundle into Celune's active voice set.

        Args:
            bundle: A built-in bundle name, explicit bundle path, or ``None`` to use Celune's default bundle.

        Returns:
            bool: ``True`` when a CEVOICE bundle was loaded, otherwise ``False``.
        """
        previous_loader = default_loader()
        previous_bundle_path = (
            self._bundle_path_string(previous_loader.bundle)
            if previous_loader is not None
            else None
        )
        previous_character = self.current_character
        select_voice_bundle(bundle)
        loader = default_loader()
        if loader is None:
            if previous_bundle_path is not None or previous_character is not None:
                self._reset_persona_conversation()
            self.current_character_persona = None
            self.current_character = None
            self.voice_bundle_is_default = True
            voices = tuple(self.backend.voices)
            self.voices = voices
            self.current_voice = (
                self.backend.default_voice
                if self.backend.default_voice in voices
                else voices[0]
                if voices
                else None
            )
            self._emit_character_event_transition(
                previous_character,
                previous_bundle_path,
                None,
                None,
                True,
            )
            return bool(voices)

        new_bundle_path = self._bundle_path_string(loader.bundle)
        self.voice_bundle_is_default = bundle_matches_default_pack_checksum(
            loader.bundle.path
        )
        self.current_character_persona = persona_metadata_from_manifest(
            loader.bundle.metadata
        )
        self.current_character = bundle_character_name(loader.bundle)
        voices = loader.bundle.voice_order
        configured_default = loader.bundle.metadata.get("default_voice")
        preferred_voice = (
            configured_default
            if isinstance(configured_default, str)
            else self.backend.default_voice
        )
        self.voices = voices
        self.current_voice = (
            preferred_voice
            if preferred_voice in voices
            else voices[0]
            if voices
            else None
        )
        if (
            previous_bundle_path != new_bundle_path
            or previous_character != self.current_character
        ):
            self._reset_persona_conversation()
        self._emit_character_event_transition(
            previous_character,
            previous_bundle_path,
            self.current_character,
            new_bundle_path,
            self.voice_bundle_is_default,
        )
        return bool(voices)

    def load_available_voices(self) -> bool:
        """Load the active voice set appropriate for the selected backend.

        Returns:
            bool: ``True`` when at least one voice is available.
        """
        if self.backend.uses_voice_bundles:
            return self.load_voice_bundle(
                _config_str(config_value(self.config, "voice_bundle"))
            )

        voices = tuple(self.backend.voices)
        self.voices = voices
        self.current_voice = (
            self.backend.default_voice
            if self.backend.default_voice in voices
            else voices[0]
            if voices
            else None
        )
        return bool(voices)

    def set_voice(self, name: str) -> bool:
        """Extension method for changing Celune's voice.

        Args:
            name: The voice name to load.

        Returns:
            bool: ``True`` when the reload thread was started, otherwise ``False``.
        """
        if not self._prepare_voice_change(name):
            return False

        threading.Thread(
            target=self.change_voice,
            args=(name,),
            daemon=True,
        ).start()
        return True

    def _prepare_voice_change(self, name: str) -> bool:
        """Wait for speech to drain before preparing one voice switch."""
        if name not in self.voices:
            # this voice was not found in the current CEVOICE/CECHAR pack
            self.log(string("celune.unknown_voice", voice=name), "warning")
            return False

        self.change_input_state_callback(locked=True)
        wait_for_speech = self._speech_playback_active()
        previous_state = self.cur_state
        self.cur_state = "reloading"

        if not self._model_ready.is_set():
            self.log(string("celune.waiting_for_models"))
        if not self._wait_until_idle(wait_for_speech=wait_for_speech):
            self.cur_state = previous_state
            self.change_input_state_callback(locked=False)
            return False

        self._model_ready.clear()
        self.loaded = False
        return True

    def set_voice_and_wait(self, name: str, timeout: float = 30.0) -> bool:
        """Change Celune's voice and wait until the reload finishes.

        Args:
            name: The voice name to load.
            timeout: How long to wait before considering the reload a failure.

        Returns:
            bool: ``True`` when the requested voice finished loading, otherwise ``False``.
        """
        if not self.set_voice(name):
            return False

        if not self._model_ready.wait(timeout=timeout):
            self.log(string("celune.voice_switch_timeout"), "warning")
            return False
        return self._voice_switch_succeeded(name)

    async def set_voice_async(self, name: str, timeout: float = 30.0) -> bool:
        """Change Celune's voice without blocking the caller's event loop.

        Args:
            name: Voice name to load.
            timeout: Maximum time to wait for the reload.

        Returns:
            ``True`` when the voice reload completed successfully.

        Raises:
            Exception: If the asynchronous reload worker raises unexpectedly.
        """
        await asyncio.to_thread(self._async_runtime_lock.acquire)
        try:
            with self._voice_reload_guard:
                if self._voice_reload_active:
                    self.log(string("celune.reload_already_in_progress"), "warning")
                    return False
                self._voice_reload_active = True
            if not await asyncio.to_thread(self._prepare_voice_change, name):
                self._clear_voice_reload_guard()
                return False
            worker = asyncio.create_task(
                asyncio.to_thread(self._run_voice_reload, name)
            )
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
            except TimeoutError:
                self.log(string("celune.voice_switch_timeout"), "warning")
                return False
        except BaseException:
            self._clear_voice_reload_guard()
            raise
        finally:
            self._async_runtime_lock.release()
        return self._voice_switch_succeeded(name)

    def _run_voice_reload(self, name: str) -> None:
        """Run a voice reload and release its guard after the worker exits."""
        try:
            self.change_voice(name)
        finally:
            self._clear_voice_reload_guard()

    def _clear_voice_reload_guard(self) -> None:
        """Clear the active voice reload marker."""
        with self._voice_reload_guard:
            self._voice_reload_active = False

    def _voice_switch_succeeded(self, name: str) -> bool:
        """Return whether the requested voice is now the active loaded voice."""
        return self.loaded and self.current_voice == name

    def set_backend(
        self,
        backend_spec: CoreBackendSpec,
    ) -> bool:
        """Request a hot reload into another TTS or VC backend.

        Args:
            backend_spec: The backend name, type, or instance to activate.

        Returns:
            bool: ``True`` when the reload worker was started.
        """
        if not self._prepare_backend_reload(backend_spec):
            return False
        preferred_voice = self.current_voice
        threading.Thread(
            target=self._hot_reload_backend,
            args=(backend_spec, preferred_voice),
            daemon=True,
        ).start()
        return True

    def _prepare_backend_reload(
        self,
        backend_spec: CoreBackendSpec,
    ) -> bool:
        """Prepare runtime state for one backend reload before loading begins."""
        if self._reload_pending or self.cur_state == "reloading":
            self.log(string("celune.reload_already_in_progress"), "warning")
            return False
        if isinstance(backend_spec, str):
            normalized_backend = backend_spec.strip().lower()
            if (
                normalized_backend not in BACKENDS
                and normalized_backend not in VC_BACKENDS
            ):
                self.log(
                    string(
                        "celune.unknown_backend",
                        backend=backend_spec,
                        available=", ".join(
                            tuple(BACKENDS.keys()) + tuple(VC_BACKENDS.keys())
                        ),
                    ),
                    "warning",
                )
                return False

        self.change_input_state_callback(locked=True)
        self.change_voice_lock_state_callback(locked=True)
        self.force_stop_speech()
        self._model_ready.clear()
        self._reload_pending = True
        self._try_play_signal("working")
        return True

    def set_backend_and_wait(
        self,
        backend_spec: CoreBackendSpec,
        timeout: Optional[float] = None,
    ) -> bool:
        """Request a hot backend reload and wait for completion.

        Args:
            backend_spec: The backend name, type, or instance to activate.
            timeout: Optional maximum seconds to wait for the reload to finish. Pass ``None`` to wait until the reload
                completes.

        Returns:
            bool: ``True`` when the requested backend finished loading.
        """
        if not self.set_backend(backend_spec):
            return False

        if not self._model_ready.wait(timeout=timeout):
            self.log(string("celune.backend_switch_timeout"), "warning")
            return False
        return self._backend_reload_succeeded(backend_spec)

    async def set_backend_async(
        self,
        backend_spec: CoreBackendSpec,
        timeout: Optional[float] = None,
    ) -> bool:
        """Request a hot backend reload without blocking the caller's event loop.

        Args:
            backend_spec: Backend specification to load.
            timeout: Maximum time to wait for the reload.

        Returns:
            ``True`` when the backend reload completed successfully.
        """
        await asyncio.to_thread(self._async_runtime_lock.acquire)
        try:
            if not await asyncio.to_thread(self._prepare_backend_reload, backend_spec):
                return False
            preferred_voice = self.current_voice
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._hot_reload_backend,
                        backend_spec,
                        preferred_voice,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                self.log(string("celune.backend_switch_timeout"), "warning")
                return False
        finally:
            self._async_runtime_lock.release()
        return self._backend_reload_succeeded(backend_spec)

    def _backend_reload_succeeded(self, backend_spec: CoreBackendSpec) -> bool:
        """Return whether the requested backend is now the active loaded runtime."""
        target_name = (
            backend_spec.name
            if isinstance(backend_spec, (CeluneBackend, CeluneVCBackend))
            else getattr(backend_spec, "name", str(backend_spec))
        )
        return (
            self.loaded
            and self._active_runtime_backend_name() == str(target_name).lower()
        )

    def set_cevoice(self, bundle: Optional[Union[str, Path]]) -> bool:
        """Request a hot reload into another CEVOICE bundle.

        Args:
            bundle: The CEVOICE bundle name or path to activate.

        Returns:
            bool: ``True`` when the reload worker was started.
        """
        if not self._prepare_cevoice_reload(bundle):
            return False
        threading.Thread(
            target=self._hot_reload_cevoice,
            args=(bundle, None),
            daemon=True,
        ).start()
        return True

    def _prepare_cevoice_reload(self, bundle: Optional[Union[str, Path]]) -> bool:
        """Prepare runtime state for one CEVOICE reload before loading begins."""
        if self._reload_pending or self.cur_state == "reloading":
            self.log(string("celune.reload_already_in_progress"), "warning")
            return False

        if bundle is not None:
            resolved_bundle = resolve_bundle_path(bundle)
            if not resolved_bundle.exists():
                self.log(
                    string("celune.voice_pack_not_found", bundle=bundle), "warning"
                )
                return False

        self.change_input_state_callback(locked=True)
        self.change_voice_lock_state_callback(locked=True)
        self.force_stop_speech()
        self._model_ready.clear()
        self._reload_pending = True
        return True

    def set_cevoice_and_wait(
        self,
        bundle: Optional[Union[str, Path]],
        timeout: Optional[float] = None,
    ) -> bool:
        """Request a hot CEVOICE reload and wait for completion.

        Args:
            bundle: The CEVOICE bundle name or path to activate.
            timeout: Optional maximum seconds to wait for the reload to finish. Pass ``None`` to wait until the reload
                completes.

        Returns:
            bool: ``True`` when the requested CEVOICE pack finished loading.
        """
        if not self.set_cevoice(bundle):
            return False

        if not self._model_ready.wait(timeout=timeout):
            self.log(string("celune.character_switch_timeout"), "warning")
            return False
        return self._cevoice_reload_succeeded(bundle)

    async def set_cevoice_async(
        self,
        bundle: Optional[Union[str, Path]],
        timeout: Optional[float] = None,
    ) -> bool:
        """Request a hot CEVOICE reload without blocking the caller's event loop.

        Args:
            bundle: CEVOICE bundle path or ``None`` for the default bundle.
            timeout: Maximum time to wait for the reload.

        Returns:
            ``True`` when the CEVOICE reload completed successfully.
        """
        await asyncio.to_thread(self._async_runtime_lock.acquire)
        try:
            if not await asyncio.to_thread(self._prepare_cevoice_reload, bundle):
                return False
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._hot_reload_cevoice, bundle, None),
                    timeout=timeout,
                )
            except TimeoutError:
                self.log(string("celune.character_switch_timeout"), "warning")
                return False
        finally:
            self._async_runtime_lock.release()
        return self._cevoice_reload_succeeded(bundle)

    def _cevoice_reload_succeeded(self, bundle: Optional[Union[str, Path]]) -> bool:
        """Return whether the requested CEVOICE bundle is now the active loaded pack."""
        return self.loaded and active_bundle_path() == resolve_bundle_path(bundle)

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

    def _speech_playback_active(self) -> bool:
        """Return whether speech generation or a speech source is active."""
        if self.locked or self.cur_state == "generating":
            return True
        if getattr(self, "_active_speech_generation", None) is not None:
            return True

        return any(
            isinstance(metadata, dict) and metadata.get("kind") == "speech"
            for metadata in self._playback_source_meta.values()
        )

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

    async def wait_until_idle_async(
        self,
        timeout: float = 30.0,
        *,
        wait_for_speech: Optional[bool] = None,
    ) -> bool:
        """Wait until model reload and playback completion without blocking the event loop.

        Args:
            timeout: Maximum time to wait for model and playback readiness.
            wait_for_speech: Whether active speech must finish before returning.

        Returns:
            ``True`` when Celune becomes ready before the timeout.
        """
        ok = await asyncio.to_thread(self._model_ready.wait, timeout)
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
            ok = await asyncio.to_thread(self._playback_done.wait, timeout)
            if not ok:
                self.log(string("celune.playback_idle_timeout"), "warning")
                return False

        with self._say_lock:
            return (not self.locked) and self.loaded

    def setup_extensions(self) -> None:
        """Configure Celune's extension manager."""
        ctx = CeluneContext(
            log=self.log,
            say=self.say,
            think=self.think,
            play=self.play,
            status=self.status_callback,
            set_voice=self.set_voice,
            get_state=lambda: self.cur_state,
            wait_until_ready=self._wait_until_idle,
            backend_override=self.with_backend,
            cevoice_override=self.with_cevoice,
            name=APP_NAME,
            version=__version__,
            log_level=self.log_level,
        )
        self.extension_manager = CeluneExtensionManager(ctx, self._event_dispatcher)
        self.extension_manager.autoload(str(project_root() / "extensions"))

        self.log(
            f"[Core] Loaded extensions: {', '.join(self.extension_manager.list_extensions())}",
            loglevel="verbose",
        )

    def log(
        self,
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Log a message.

        Args:
            msg: The message to emit.
            severity: The message severity level.
            loglevel: The minimum configured log level required to emit the message.
        """
        levels = {"info": 0, "verbose": 1, "debug": 2}
        if levels.get(self.log_level, 0) < levels.get(loglevel, 0):
            return
        self.log_callback(msg, severity)

    def _buffer_startup_log(
        self,
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Buffer startup diagnostics until the runtime banner is emitted."""
        levels = {"info": 0, "verbose": 1, "debug": 2}
        if levels.get(self.log_level, 0) < levels.get(loglevel, 0):
            return
        if not self._startup_banner_emitted:
            self._startup_log_buffer.append((msg, severity, loglevel))
            return
        self._deliver_startup_log(msg, severity, loglevel)

    def _emit_runtime_banner_line(
        self,
        msg: str,
        severity: str = "info",
    ) -> None:
        """Emit one runtime-banner line without entering the startup buffer."""
        self._deliver_startup_log(msg, severity, "info")

    def _deliver_startup_log(
        self,
        msg: str,
        severity: str,
        loglevel: LogLevel,
    ) -> None:
        """Forward one startup log while retaining legacy callback compatibility."""
        try:
            self._startup_log_sink(msg, severity, loglevel=loglevel)
        except TypeError as error:
            if "loglevel" not in str(error):
                raise
            self._startup_log_sink(msg, severity)

    def _flush_startup_logs(self) -> None:
        """Flush buffered constructor diagnostics after the runtime banner."""
        buffered_logs = self._startup_log_buffer
        self._startup_log_buffer = []
        for msg, severity, loglevel in buffered_logs:
            self._deliver_startup_log(msg, severity, loglevel)

    def log_dev(self, msg: str, severity: str = "info") -> None:
        """Log a legacy developer message at verbose level.

        Args:
            msg: The message to emit.
            severity: The message severity level.
        """
        self.log(msg, severity, loglevel="verbose")

    def try_play_signal(self, signal_type: str) -> bool:
        """Public interface for Celune._try_play_signal.

        Args:
            signal_type: The signal type to play.

        Returns:
            bool: ``True`` when the requested signal was played, otherwise ``False``.

        Raises:
            ValueError: An invalid signal was requested.
        """
        return self._try_play_signal(signal_type)

    def _try_play_signal(self, signal_type: str) -> bool:
        """Play a runtime signal only when the playback pipeline can currently accept it."""
        playback_thread = self.playback_thread
        if playback_thread is None or not playback_thread.is_alive():
            return False

        if self.locked and self._playback_done.is_set():
            self._release_pipeline()

        return play_signal(self, signal_type)

    def voice_prompt_supported(self) -> bool:
        """Return whether the active TTS configuration supports voice prompts.

        Returns:
            bool: Whether the currently loaded TTS model supports voice prompting.
        """
        backend = self.backend
        return not (
            backend.name == "qwen3"
            and getattr(backend, "clone_model_id", None) == QWEN3_0_6B_MODEL
        )

    def effective_voice_prompt(self) -> Optional[str]:
        """Return the active voice prompt only when the current model supports it.

        Returns:
            Optional[str]: The current voice prompt if voice prompts are supported, else ``None``.
        """
        if not self.voice_prompt_supported():
            return None
        return self.voice_prompt

    def change_voice(self, voice: str) -> None:
        """Change Celune's voice parameters.

        Args:
            voice: The voice name to load and warm up.

        Raises:
            WarmupError: The newly loaded voice fails warmup.
        """

        self.log(string("celune.reloading", app_name=APP_NAME))
        self._ready_announced = False
        self.status_callback(string("status.reloading"))
        self.progress_callback(None, None)
        self.cur_state = "reloading"
        active_voice = self.current_voice or voice

        try:
            with self._model_lock:
                if self._is_voice_conversion_mode():
                    self.current_voice = voice
                    self.loaded = True
                else:
                    new_model_name = self.backend.model_id_for_voice(voice)

                    # VoxCPM2 uses the same model for all voices, so we don't have to reload every time
                    if new_model_name != self.model_name:
                        if not self._try_play_signal("working"):
                            self.log(
                                "Could not play the working signal.",
                                "warning",
                                loglevel="verbose",
                            )
                        self.log(
                            f"[RELOAD] Unloading model: {self.model_name}",
                            loglevel="verbose",
                        )
                        self.unload_runtime_state(include_normalizer=False)
                        self.log(
                            f"[RELOAD] Loading model: {new_model_name}",
                            loglevel="verbose",
                        )
                        self.model = self.backend.load_model(new_model_name)
                        self.model_name = new_model_name

                        self.log(string("celune.rewarming_up"))
                        if not self._warmup():
                            self._raise_warmup_error("warmup failed after reload")

                        if not self._try_play_signal("readiness"):
                            self.log(
                                "Could not play the readiness signal.",
                                "warning",
                                loglevel="verbose",
                            )

                    self.log(
                        "[RELOAD] The target model is the same as the model currently in use.",
                        loglevel="verbose",
                    )

                    self.current_voice = voice
                    self.loaded = True

            self.voice_changed_callback(voice)
            if active_voice != voice:
                self._emit_event(
                    "voice_changed",
                    VoiceChangedEvent(
                        celune=self,
                        old_voice=active_voice,
                        new_voice=voice,
                    ),
                )
            self.log(string("celune.voice_loaded", voice=voice))
            self.progress_callback(1, 1)
            self.cur_state = "idle"
            self.status_callback(string("status.idle"))
        except Exception as e:
            self.fatal()
            self.log(
                string("celune.reload_error", error=format_error(e, self.log_level)),
                "error",
            )
            self.status_callback(
                string("status.could_not_reload", app_name=APP_NAME),
                "error",
            )
            self.error_callback(string("status.could_not_reload", app_name=APP_NAME))
            self.progress_callback(0, 1)
        finally:
            self._model_ready.set()
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def force_stop_speech(self) -> bool:
        """Forcefully stop Celune from speaking.

        Returns:
            bool: ``True`` when an active utterance was interrupted, otherwise ``False``.
        """
        return force_stop_pipeline(self)

    async def force_stop_speech_async(self) -> bool:
        """Forcefully stop Celune from speaking without blocking an async caller.

        Returns:
            ``True`` when an active utterance was interrupted.
        """
        return await asyncio.to_thread(force_stop_pipeline, self)

    async def enter_sleep_mode_async(self) -> bool:
        """Put Celune to sleep without blocking the caller's event loop.

        Returns:
            ``True`` when Celune enters sleep mode successfully.
        """
        await asyncio.to_thread(self._async_runtime_lock.acquire)
        try:
            return await asyncio.to_thread(self.enter_sleep_mode)
        finally:
            self._async_runtime_lock.release()

    async def wake_from_sleep_async(self) -> bool:
        """Wake Celune without blocking the caller's event loop.

        Returns:
            ``True`` when Celune wakes successfully.
        """
        await asyncio.to_thread(self._async_runtime_lock.acquire)
        try:
            return await asyncio.to_thread(self.wake_from_sleep)
        finally:
            self._async_runtime_lock.release()

    def load(
        self, raise_on_error: bool = False, skip_runtime_check: bool = False
    ) -> bool:
        """Load and initialize Celune.

        Args:
            raise_on_error: Whether Celune should raise for failures to load or signal fatal states.
            skip_runtime_check: Whether Celune should skip runtime checks during startup.

        Returns:
            bool: ``True`` when initialization completed successfully, otherwise ``False``.

        Raises:
            NotAvailableError: If no usable voice or backend is available.
            Exception: If an unexpected loading failure occurs and `raise_on_error` is enabled.
            RuntimeCheckError: If the runtime environment is unsupported.
            BackendError: If backend initialization fails.
        """
        log_runtime_banner(
            self._emit_runtime_banner_line,
            self.vc_backend or self.backend,
            self.backend_mode,
        )
        self._startup_banner_emitted = True
        self._flush_startup_logs()
        self.historical_generated_speech_seconds = saved_output_speech_seconds()

        if self.backend_mode == "ui_test" and self.backend.is_fake:
            self.log(string("celune.test_mode_active", app_name=APP_NAME))
            return True

        if not self.load_available_voices():
            self.fatal()
            self.log(string("celune.no_voices_loaded"), "error")
            self.error_callback(string("celune.no_voices_loaded_short"))
            self.progress_callback(0, 1)
            if raise_on_error:
                raise NotAvailableError("no voices are available")
            return False

        if self.backend.uses_voice_bundles:
            announced_character = announce_default_bundle(self.log)
            character = self.current_character or announced_character
            self.current_character = character

            if self.voice_bundle_is_default:
                self.log(
                    string("celune.current_character_default", character=character)
                )
            else:
                self.log(string("celune.current_character", character=character))

        if self.backend_mode == "normal":
            self.setup_extensions()

        vram_message = validate_vram_preset(self.config)
        if vram_message:
            self.log(vram_message, "warning")

        self.log(
            string(
                "celune.current_vram_preset",
                preset=str(self.config.get("vram", "unknown")).title(),
            )
        )

        self.progress_callback(None, None)
        if self._is_voice_conversion_mode():
            if self.vc_backend is None:
                self.fatal()
                self.log(string("celune.no_vc_backend"), "error")
                self.error_callback(string("celune.no_valid_vc_backend"))
                self.progress_callback(0, 1)
                if raise_on_error:
                    raise NotAvailableError(
                        "requested VC mode, but no VC backend was loaded"
                    )
                return False

            self.vc_backend.preload_models()
            self.model = None
            self.model_name = ""
            self.log(string("celune.ready_for_vc"))
        else:
            self.backend.preload_models()

            self.log(string("celune.all_voices_available"))
            try:
                self.model = self.backend.load_default_model()
                active_voice = self.current_voice or self.voices[0]
                self.model_name = self.backend.model_id_for_voice(active_voice)
            except Exception as e:
                self.fatal()
                self.log(
                    string("celune.default_model_load_failed", app_name=APP_NAME),
                    "error",
                )
                self.log(format_error(e, self.log_level), "error")
                self.error_callback(string("celune.default_model_failed_short"))
                self.progress_callback(0, 1)
                if raise_on_error:
                    raise
                return False

        pipeline_thread = threading.Thread(target=self._run_pipeline_jobs, daemon=True)
        self._generation_thread = None
        self._playback_thread = pipeline_thread
        pipeline_thread.start()

        if not skip_runtime_check and not validate_runtime(
            log=self.log,
            error=self.error_callback,
            set_state=lambda state: setattr(self, "cur_state", state),
            glow_connect_failed=self.glow.connect_failed,
            format_error=format_error,
            log_level=self.log_level,
            backend_name=self._active_runtime_backend_name(),
        ):
            self.fatal()
            self._stop_pipeline_jobs()
            if raise_on_error:
                raise RuntimeCheckError("runtime check failed")
            return False

        warmup_ok = True
        if not self._is_voice_conversion_mode():
            warmup_ok = self._warmup()

        if warmup_ok:
            self.loaded = True
            self._model_ready.set()
            self._release_pipeline()
            self.glow.enter()  # Celune has entered your PC
        else:
            self.fatal()
            self.log(string("celune.warmup_failed"), "error")
            self._stop_pipeline_jobs()
            if raise_on_error:
                raise BackendError("warmup failed")
            return False

        self._start_persona_background_load()

        if self.use_normalization:
            self.load_normalizer()

        if self.backend_mode == "normal":
            self._start_configured_api()

        if persona_enabled(self.config) and self.vision is None:
            self.log(
                string("celune.personas_unavailable", app_name=APP_NAME),
                "warning",
            )

        if self.backend_mode == "normal" and not self._try_play_signal("readiness"):
            self.log(
                "Could not play the readiness signal.",
                "warning",
                loglevel="verbose",
            )

        self._emit_event("ready", ReadyEvent(celune=self))

        return True

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

    def _start_configured_api(self) -> None:
        """Start the API from config without blocking Celune startup."""
        enabled, host, port, token, requests_per_minute = self._api_settings()
        if not enabled or self._api_thread is not None:
            return

        if not is_port_usable(port):
            self.log(string("api.port_in_use", port=port), "warning")
            self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
            return

        try:
            from .api import start_api
        except ModuleNotFoundError as package:
            self.log(
                string(
                    "celune.required_package_missing",
                    package=package.name,
                ),
                "warning",
            )
            self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
            return
        except Exception as e:
            self.log(
                string(
                    "celune.package_import_failed",
                    error=format_error(e, self.log_level),
                ),
                "warning",
            )
            self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
            return

        try:
            self._api_thread = start_api(
                self,
                host=host,
                port=port,
                token=token,
                requests_per_minute=requests_per_minute,
            )
        except Exception as e:
            self.log(
                string("celune.internal_error", error=format_error(e, self.log_level)),
                "warning",
            )
            self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
            return

    def load_normalizer(self) -> None:
        """Load the normalizer LLM."""
        load_epoch = self._normalizer_load_epoch + 1
        self._normalizer_load_epoch = load_epoch

        def _worker():
            loaded_tokenizer: Optional[PreTrainedTokenizerBase] = None
            loaded_llm: Optional[PreTrainedModel] = None

            discard(loaded_tokenizer)
            discard(loaded_llm)
            try:
                loaded_tokenizer, loaded_llm = load_normalizer_components(
                    self.log, self.backend, self.config
                )
                with self._model_lock:
                    if (
                        self._normalizer_load_epoch != load_epoch
                        or self.sleeping
                        or self.exit_requested
                    ):
                        discard(loaded_llm)
                        discard(loaded_tokenizer)
                        gc.collect()
                        if torch.cuda.is_available():
                            with contextlib.suppress(Exception):
                                torch.cuda.synchronize()
                            with contextlib.suppress(Exception):
                                torch.cuda.empty_cache()
                        self.log(
                            "[NORMALIZER] Discarded stale normalizer load.",
                            loglevel="verbose",
                        )
                        return

                    self.tokenizer = loaded_tokenizer
                    self.llm = loaded_llm
                self.log(string("celune.normalizer_loaded"))
                self.progress_callback(1, 1)
            except Exception as e:
                self.log(
                    string(
                        "celune.normalizer_error",
                        error=format_error(e, self.log_level),
                    ),
                    "error",
                )
                self.log(string("celune.normalizer_failed"), "warning")
                self.log(string("celune.normalization_unavailable"), "warning")
                self.progress_callback(0, 1)

        with self._model_lock:
            if self.persona_ready or self.persona_loading:
                return

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self.progress_callback(None, None)
        self.log(
            string(
                "celune.loading_normalizer",
                model_id=NORMALIZER_MODEL_ID,
                device=normalizer_device(self.config),
            )
        )

    def _warmup(
        self,
        fatal_on_failure: bool = True,
        backend: Optional[CeluneBackend] = None,
        model: Optional[PreTrainedModel] = None,
        voice: Optional[str] = None,
    ) -> bool:
        """Warm up Celune's speech capabilities."""
        self.log(string("celune.warmup_start"))
        self.status_callback(string("status.warming_up"))
        self.progress_callback(None, None)
        warmup_text = "A"
        self._last_warmup_error = None
        active_backend = backend if backend is not None else self.backend
        active_model = model if model is not None else self.model
        active_voice = voice if voice is not None else self.current_voice

        forced_error = os.getenv("CELUNE_FORCE_ERROR") in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }

        if forced_error:
            raise WarmupError("forced warmup failure")

        try:
            warmup_start = time.perf_counter()

            with self._model_lock:
                if active_model is None:
                    raise WarmupError("cannot warm up a null model")

                for _, _, _ in active_backend.generate_stream(
                    active_model,
                    text=warmup_text,
                    language=self.language,
                    chunk_size=self.chunk_size,
                    instruct=self.effective_voice_prompt(),
                    voice=active_voice,
                ):
                    pass

            warmup_end = time.perf_counter()
            warmup_took = warmup_end - warmup_start
            self.log(
                f"[WARMUP] done, took {format_number(warmup_took, 2)} seconds",
                loglevel="verbose",
            )

            self.progress_callback(1, 1)
            return True
        except Exception as e:
            self._last_warmup_error = e
            self.log(
                string("celune.warmup_error", error=format_error(e, self.log_level)),
                "error",
            )
            self.progress_callback(0, 1)
            if fatal_on_failure:
                self.fatal()
                self.error_callback(
                    string("celune.warmup_failed_app", app_name=APP_NAME)
                )
            return False

    # as of CeluneNorm 2.0, normalization ACTUALLY works with long inputs
    def normalize(self, text: str) -> Optional[str]:
        """Normalize input text using CeluneNorm.

        Args:
            text: The raw text to normalize before speech generation.

        Returns:
            Optional[str]: The normalized text, the original text for blank input, or ``None`` when normalization is
            unavailable or has failed.
        """

        if not self.use_normalization:
            return None

        if not text or not text.strip():
            return text

        if self.llm is None or self.tokenizer is None:
            return None

        llm = cast(Generative, self.llm)
        tokenizer = cast(NormalizerTokenizer, self.tokenizer)

        def _run_inference() -> Optional[str]:
            inf_start = time.perf_counter()
            try:
                bad_text = text.strip()
                norm_token = "<NORM>"

                # Are we using CeluneNorm?
                norm_token_id = tokenizer.convert_tokens_to_ids(norm_token)
                custom_assert(
                    norm_token_id is not None, ValueError("not a CeluneNorm normalizer")
                )
                assert norm_token_id is not None

                custom_assert(
                    norm_token_id != tokenizer.unk_token_id,
                    ValueError("not a CeluneNorm normalizer"),
                )
                assert norm_token_id != tokenizer.unk_token_id

                prompt = f"{bad_text}{norm_token}"

                tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                device = next(llm.parameters()).device
                inputs = tokens.to(device)
                token_ids = cast(torch.Tensor, tokens["input_ids"])
                len_tokens = token_ids.shape[1]

                self.log(string("celune.tokens_to_normalize", count=len_tokens))
                if len_tokens > 512:
                    self.log(string("celune.input_too_long_to_normalize"), "warning")
                    return None

                with torch.inference_mode():
                    output_ids = llm.generate(
                        **inputs,
                        # CeluneNorm will likely return less, unless you use up your whole context allowance
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_ids = inputs["input_ids"]

                if not isinstance(input_ids, torch.Tensor):
                    self.log(
                        string("celune.normalizer_output_not_tensor"),
                        "warning",
                    )
                    return None

                prompt_len = input_ids.shape[1]
                new_ids = output_ids[0][prompt_len:]

                # CeluneNorm shouldn't do this, but if it does happen, stop Celune from saying nothing
                if new_ids.numel() == 0:
                    self.log(string("celune.normalizer_returned_no_tokens"), "warning")
                    return None

                out = tokenizer.decode(new_ids, skip_special_tokens=True)

                # fix type checker
                if isinstance(out, list):
                    out = out[0] if out else ""

                # too many <NORM>'s can break splitting
                if "<NORM>" in out:
                    out = out.split("<NORM>", 1)[0].strip()

                # are we absolutely sure CeluneNorm did produce something before Celune gets to say it?
                if not out:
                    self.log(string("celune.normalizer_bad_output"), "warning")
                    return None

                inf_total = time.perf_counter() - inf_start
                self.log(string("celune.normalized_text", text=out))
                self.log(
                    string(
                        "celune.normalization_took",
                        seconds=format_number(inf_total, 2),
                    )
                )

                return out

            except Exception as e:
                self.log(
                    string(
                        "celune.normalization_error",
                        error=format_error(e, self.log_level),
                    ),
                    "error",
                )
                return None

        return _run_inference()  # blocks the generation thread, but Celune doesn't mind it since the main thread is up

    def _acquire_pipeline(self, action: str) -> bool:
        """Atomically claim Celune's shared playback pipeline."""
        return acquire_pipeline(self, action)

    def _release_pipeline(self) -> None:
        """Release Celune's shared playback pipeline."""
        release_pipeline(self)

    def _stop_pipeline_jobs(self) -> None:
        """Stop startup pipeline workers after an initialization failure."""
        with self.queue_lock:
            self.text_queue.put(self.sentinel)
            if self._is_voice_conversion_mode():
                self.audio_queue.put(self.sentinel)

        if self._playback_thread is not None:
            self._playback_thread.join()

    def think(
        self,
        text: str,
    ) -> bool:
        """Let Celune reply to an input request.

        Args:
            text: The request that will be sent to Persona for processing.

        Returns:
            bool: ``True`` if Celune processed this smart request, otherwise ``False``.
        """
        if self.test_finished or self.backend_mode == "agent_test":
            return False
        if not mode_allows_persona(self.mode):
            return self.say(text)
        if self.input_mode != "text_to_speech":
            self.log(string("celune.text_input_unavailable_vc"), "warning")
            self.error_callback(string("celune.not_possible"))
            return False

        if self.is_in_tutorial:
            self.log(string("celune.speech_input_disabled_tutorial"), "warning")
            return False

        if self._speech_playback_active():
            self.force_stop_speech()
        self._interrupt_active_agent_for_input(text)

        if self.sleeping:
            self.log(
                string("celune.cannot_think_sleeping", app_name=APP_NAME),
                "warning",
            )
            self.error_callback(string("celune.app_sleeping", app_name=APP_NAME))
            return False

        with self.say_lock:
            self._persona_queue.put(text)
            thread = self._persona_thread
            if thread is not None and thread.is_alive():
                return True

            self.status_callback(string("status.thinking"))
            self.progress_callback(None, None)
            self._ready_announced = False
            thread = threading.Thread(
                target=self._think_worker,
                daemon=True,
            )
            self._persona_thread = thread
            thread.start()
        return True

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

    async def think_async(self, text: str) -> bool:
        """Let Celune reply to one input request without blocking an async caller.

        Args:
            text: Input text for Persona to answer.

        Returns:
            ``True`` when the response was queued successfully.
        """
        return await asyncio.to_thread(self.think, text)

    def _think_worker(self) -> None:
        """Fetch queued Persona responses without blocking Celune's UI thread."""
        current_thread = threading.current_thread()

        try:
            while not self.exit_requested:
                try:
                    text = self._persona_queue.get(timeout=0.1)
                except queue.Empty:
                    with self.say_lock:
                        if not self._persona_queue.empty():
                            continue
                        if self._persona_thread is current_thread:
                            self._persona_thread = None
                        return

                if not self._wait_for_persona_playback():
                    return

                self.status_callback(string("status.thinking"))
                self.cur_state = "thinking"
                self.progress_callback(None, None)

                with self._model_lock:
                    persona_loading = self.persona_loading
                    persona_ready = self.persona_ready
                    vision = self.vision

                route = self.route_input(text, persona_ready=persona_ready)
                if route.failure is not None:
                    can_prepare_persona = (
                        route.failure.kind
                        == AgentClassificationFailureKind.PERSONA_UNAVAILABLE
                        and not persona_ready
                    )
                    if not can_prepare_persona:
                        self._speak_agent_classification_failure(
                            text, route.failure, route
                        )
                        continue
                if route.route == AgentRoute.CLARIFICATION:
                    clarification = route.clarification_prompt
                    if clarification:
                        self.say(clarification)
                    continue
                if route.route != AgentRoute.CONVERSATION:
                    self._run_agent_route(route)
                    continue

                if persona_loading:
                    self.say(text)
                    continue

                if not persona_ready:
                    if vision is None:
                        with self._model_lock:
                            if self.vision is None:
                                self.vision = self._persona_conn()
                            vision = self.vision
                    if vision is None:
                        self.say(text)
                        continue
                    self._start_persona_background_load()
                    self.say(text)
                    continue

                if not think_pipeline(self, text):
                    self.log(string("celune.say_instead"), "warning")
                    self.say(text)
        finally:
            with self.say_lock:
                if self._persona_thread is current_thread:
                    self._persona_thread = None

    @property
    def agent_needle_ready(self) -> bool:
        """Return whether the production Needle selector is loaded and usable."""
        return self._agent_needle_selector is not None

    @property
    def agent_needle_error(self) -> Optional[str]:
        """Return the latest production Needle loading failure, if any."""
        return self._agent_needle_error

    def _load_agent_tool_selector(self) -> AgentToolSelector:
        """Load the verified Needle selector only when an agent task needs it."""
        if self._agent_needle_selector is not None:
            return self._agent_needle_selector
        try:
            selector = NeedleToolSelector.from_pretrained(
                self._agent_tools,
                schemas=self._agent_tool_schemas,
            )
        except Exception as exc:
            self._agent_needle_error = str(exc)
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
                        f"[AGENT] failure_response_generation_failed error={exc}",
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
                f"[AGENT] failure_response_generation_failed error={exc}",
                "warning",
                loglevel="verbose",
            )
        return self.say(string("agent.classifier_unavailable"))

    def _wait_for_persona_playback(self) -> bool:
        """Wait until the shared speech pipeline is available for the next Persona turn."""
        while not self.exit_requested:
            self.playback_done.wait(timeout=0.1)
            with self.say_lock:
                if not self.locked and self.cur_state not in {"generating", "speaking"}:
                    return True
        return False

    def say(
        self,
        text: str,
        save: bool = True,
        display_text: Optional[str] = None,
    ) -> bool:
        """Queue text for Celune to say.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.
            display_text: Optional text to show in logs instead of the synthesis text.

        Returns:
            bool: ``True`` when the text was queued successfully, otherwise ``False``.
        """
        if self.test_finished:
            return False
        self.log(
            f"[ENGINE] say requested text_chars={len(text)} save={save} "
            f"state={self.cur_state} mode={self.input_mode}",
            loglevel="debug",
        )
        if self.input_mode != "text_to_speech":
            self.log(string("celune.text_input_unavailable_vc"), "warning")
            self.error_callback(string("celune.not_possible"))
            self.progress_callback(0, 1)
            return False

        if self._speech_playback_active():
            self.force_stop_speech()

        return say_pipeline(self, text, save=save, display_text=display_text)

    async def say_async(
        self,
        text: str,
        save: bool = True,
        display_text: Optional[str] = None,
    ) -> bool:
        """Queue text for Celune to say without blocking an async caller.

        Args:
            text: Text to synthesize.
            save: Whether to save generated output artifacts.
            display_text: Optional text to show instead of the synthesis text.

        Returns:
            ``True`` when the text was queued successfully.
        """
        if self.input_mode != "text_to_speech":
            self.log(string("celune.text_input_unavailable_vc"), "warning")
            self.error_callback(string("celune.not_possible"))
            self.progress_callback(0, 1)
            return False

        if self._speech_playback_active():
            self.force_stop_speech()

        return await say_pipeline_async(
            self, text, save=save, display_text=display_text
        )

    def say_stream(self, text: str, save: bool = True) -> Optional[SpeechStreamQueue]:
        """Queue text for playback and mirror generated chunks to a queue.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            Optional[queue.Queue]: Queue receiving 48 kHz stereo float32 chunks, or ``None`` when the request could not
            be queued.
        """
        stream_queue: SpeechStreamQueue = queue.Queue(maxsize=2)
        if not queue_speech(self, text, save=save, stream_queue=stream_queue):
            return None
        return stream_queue

    async def say_stream_async(
        self,
        text: str,
        save: bool = True,
    ) -> Optional[SpeechStreamQueue]:
        """Queue text for playback and mirror chunks without blocking an async caller.

        Args:
            text: Text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            A queue receiving generated audio chunks, or ``None`` when queuing fails.
        """
        stream_queue: SpeechStreamQueue = queue.Queue(maxsize=2)
        if not await queue_speech_async(
            self,
            text,
            save=save,
            stream_queue=stream_queue,
        ):
            return None
        return stream_queue

    def submit_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        label: str = "audio input",
        pitch_shift: Optional[int] = None,
        f0_condition: Optional[bool] = None,
        log_playback: bool = True,
        reset_ready_announcement: bool = True,
    ) -> bool:
        """Accept audio input for future non-TTS engine modes.

        Args:
            audio: Decoded mono or stereo input audio.
            sample_rate: Sample rate for the submitted audio.
            label: Human-readable label for the input source.
            pitch_shift: Optional semitone adjustment to apply during VC.
            f0_condition: Optional override enabling Seed-VC singing mode.
            log_playback: Whether playback timing and length info should be logged.
            reset_ready_announcement: Whether this audio input should allow a later ready message once playback
                completes.

        Returns:
            bool: ``True`` when the current mode accepted the audio input.
        """
        if self.test_finished or self.backend_mode == "agent_test":
            return False
        return handle_audio_input(
            self,
            AudioInputRequest(
                audio=np.asarray(audio, dtype=np.float32),
                sample_rate=sample_rate,
                label=label,
                pitch_shift=pitch_shift,
                f0_condition=f0_condition,
                log_playback=log_playback,
                reset_ready_announcement=reset_ready_announcement,
            ),
        )

    def convert_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        label: str = "audio input",
        pitch_shift: Optional[int] = None,
        f0_condition: Optional[bool] = None,
    ) -> Optional[AudioOutput]:
        """Convert submitted audio and return the generated VC output.

        Args:
            audio: Decoded mono or stereo input audio.
            sample_rate: Sample rate for the submitted audio.
            label: Human-readable label for the input source.
            pitch_shift: Optional semitone adjustment overriding the VC default.
            f0_condition: Optional override enabling Seed-VC singing mode.

        Returns:
            Optional[AudioOutput]: The converted audio output, or ``None`` when voice conversion is unavailable.
        """
        if not self._is_voice_conversion_mode():
            self.log(
                string("celune.audio_conversion_unavailable"),
                "warning",
            )
            self.error_callback(string("celune.not_possible"))
            self.progress_callback(0, 1)
            return None

        return convert_audio_input(
            self,
            AudioInputRequest(
                audio=np.asarray(audio, dtype=np.float32),
                sample_rate=sample_rate,
                label=label,
                pitch_shift=pitch_shift,
                f0_condition=f0_condition,
            ),
        )

    def convert_live_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        label: str = "audio input",
        pitch_shift: Optional[int] = None,
        f0_condition: Optional[bool] = None,
    ) -> Optional[AudioOutput]:
        """Convert one low-latency live block through the active VC backend."""
        if not self._is_voice_conversion_mode():
            return self.convert_audio(
                audio,
                sample_rate,
                label=label,
                pitch_shift=pitch_shift,
                f0_condition=f0_condition,
            )

        return convert_audio_input(
            self,
            AudioInputRequest(
                audio=np.asarray(audio, dtype=np.float32),
                sample_rate=sample_rate,
                label=label,
                pitch_shift=pitch_shift,
                f0_condition=f0_condition,
            ),
            live=True,
        )

    def stop_live_audio(self) -> None:
        """Reset any state held by the active live voice-conversion backend."""
        stop_live_audio_input(self)

    def play(self, sound_path: str, keep: bool = False, volume: float = 1.0) -> bool:
        """Play a sound via Celune's pipeline.

        Args:
            sound_path: The path to the audio file to play.
            keep: Whether to prepend this SFX to the next saved utterance.
            volume: How loud should the SFX be played at.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise ``False``.
        """
        if self.test_finished or self.backend_mode == "agent_test":
            return False
        return play_pipeline(self, sound_path, keep=keep, volume=volume)

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
        with self._async_runtime_lock:
            self._exit_requested = True
            self.log(
                f"[ENGINE] close requested state={self.cur_state} loaded={self.loaded} "
                f"sleeping={self.sleeping}",
                loglevel="debug",
            )
            with self._model_lock:
                if self._closed:
                    return
                self._closed = True
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
                if self._pipeline_workers_active():
                    self._abort_backend_operations()
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

    def _pipeline_workers_active(self) -> bool:
        """Return whether a bounded pipeline shutdown left work running."""
        return any(
            worker is not None and worker.is_alive()
            for worker in (self.generation_thread, self.playback_thread)
        )

    def _abort_backend_operations(self) -> None:
        """Abort backend-owned work before pipeline and model teardown begins."""
        for backend in (self.backend, self.vc_backend):
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
