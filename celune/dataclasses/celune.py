"""Grouped Celune runtime state containers and property specs."""

import queue
import threading
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sounddevice as sd
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..config import Config
from ..chroma import AudioRGBGlow
from ..cevoice import CEVoicePersona
from ..backends.tts import CeluneBackend
from ..persona.impl import PersonaClient
from ..backends.vc import CeluneVCBackend
from ..typing.backends import BackendModel
from ..dsp import StreamingPedalboardReverb
from ..extensions.manager import CeluneExtensionManager
from ..constants import JSONSerializable, PipelineStates
from .properties import ConstantPropertySpec, ForwardedPropertySpec
from ..typing.celune import (
    ErrorCallback,
    IdleCallback,
    InputStateCallback,
    MessageCallback,
    ProgressCallback,
    QueueAvailableCallback,
    TTSBackendRecipe,
    VCBackendRecipe,
    VoiceChangedCallback,
    VoiceLockStateCallback,
)


@dataclass
class CeluneCallbackState:
    """Callbacks Celune uses to report state outward."""

    log_callback: MessageCallback
    status_callback: MessageCallback
    error_callback: ErrorCallback
    idle_callback: IdleCallback
    queue_avail_callback: QueueAvailableCallback
    voice_changed_callback: VoiceChangedCallback
    change_input_state_callback: InputStateCallback
    change_voice_lock_state_callback: VoiceLockStateCallback
    progress_callback: ProgressCallback


@dataclass
class CeluneBackendState:
    """Backend selection and configuration state."""

    config: Config
    backend_spec: Optional[TTSBackendRecipe] = None
    backend_kwargs: dict[str, JSONSerializable] = field(default_factory=dict)
    backend: Optional[CeluneBackend] = None
    tts_backend: str = ""
    vc_backend_spec: Optional[VCBackendRecipe] = None
    vc_backend: Optional[CeluneVCBackend] = None
    voice_conversion_backend: str = ""
    vc_pitch_shift: int = 0
    vc_f0_condition: bool = False
    input_mode: str = "text_to_speech"
    chunk_size: int = 0
    language: str = "Auto"
    dev: bool = False
    use_normalization: bool = False


@dataclass
class CeluneModelState:
    """Loaded TTS and normalizer model state."""

    model: Optional[BackendModel] = None
    model_name: str = ""
    llm: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    last_warmup_error: Optional[Exception] = None
    normalizer_load_epoch: int = 0


@dataclass
class CeluneVoiceState:
    """Voice and character-related state."""

    current_voice: Optional[str] = None
    current_character: Optional[str] = None
    current_character_persona: Optional[CEVoicePersona] = None
    voice_bundle_is_default: bool = True
    persona_history: list[dict[str, str]] = field(default_factory=list)
    persona_attachments: list[dict[str, str]] = field(default_factory=list)
    voices: tuple[str, ...] = ()
    voice_prompt: Optional[str] = None


@dataclass
class CelunePipelineState:
    """Queues, worker threads, locks, and playback coordination."""

    text_queue: queue.Queue = field(default_factory=queue.Queue)
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    playback_thread: Optional[threading.Thread] = None
    generation_thread: Optional[threading.Thread] = None
    api_thread: Optional[threading.Thread] = None
    persona_thread: Optional[threading.Thread] = None
    queue_lock: threading.Lock = field(default_factory=threading.Lock)
    utterance_force_stop: threading.Event = field(default_factory=threading.Event)
    next_playback_source_id: int = 0
    playback_source_statuses: dict[int, str] = field(default_factory=dict)
    playback_source_meta: dict[int, dict[str, Union[str, float]]] = field(
        default_factory=dict
    )
    playback_progress_last_emit_at: float = 0.0
    playback_progress_last_source_id: int = 0
    model_ready: threading.Event = field(default_factory=threading.Event)
    playback_done: threading.Event = field(default_factory=threading.Event)
    say_lock: threading.Lock = field(default_factory=threading.Lock)
    wake_lock: threading.Lock = field(default_factory=threading.Lock)
    model_lock: threading.RLock = field(default_factory=threading.RLock)
    exit_requested: bool = False


@dataclass
class CeluneAudioState:
    """Audio output and effect-related state."""

    stream: Optional[sd.OutputStream] = None
    current_sr: Optional[int] = None
    audio_unavailable: bool = False
    can_use_rubberband: bool = True
    speed: float = 1.0
    smart_buffer_generation_speed: Optional[float] = None
    smart_buffer_target_seconds: float = 0.0
    total_generated_speech_seconds: float = 0.0
    historical_generated_speech_seconds: float = 0.0
    reverb: StreamingPedalboardReverb = field(default_factory=StreamingPedalboardReverb)
    recently_saved: Optional[str] = None
    kept_sfx_audio: Optional[npt.NDArray[np.float32]] = None


@dataclass
class CeluneRuntimeState:
    """Top-level lifecycle and runtime integration state."""

    regenerate: bool = False
    locked: bool = True
    loaded: bool = False
    reload_pending: bool = False
    sleeping: bool = False
    last_flavor: Optional[str] = None
    ready_announced: bool = False
    cur_state: str = "init"
    is_in_tutorial: bool = False
    extension_manager: Optional[CeluneExtensionManager] = None
    glow: Optional[AudioRGBGlow] = None
    vision: Optional[PersonaClient] = None


CELUNE_FORWARDED_PROPERTIES = (
    ForwardedPropertySpec("log_callback", "_callbacks", "log_callback"),
    ForwardedPropertySpec("status_callback", "_callbacks", "status_callback"),
    ForwardedPropertySpec("error_callback", "_callbacks", "error_callback"),
    ForwardedPropertySpec("idle_callback", "_callbacks", "idle_callback"),
    ForwardedPropertySpec("queue_avail_callback", "_callbacks", "queue_avail_callback"),
    ForwardedPropertySpec(
        "voice_changed_callback", "_callbacks", "voice_changed_callback"
    ),
    ForwardedPropertySpec(
        "change_input_state_callback",
        "_callbacks",
        "change_input_state_callback",
    ),
    ForwardedPropertySpec(
        "change_voice_lock_state_callback",
        "_callbacks",
        "change_voice_lock_state_callback",
    ),
    ForwardedPropertySpec("progress_callback", "_callbacks", "progress_callback"),
    ForwardedPropertySpec("config", "_backend_state", "config"),
    ForwardedPropertySpec("_backend_spec", "_backend_state", "backend_spec"),
    ForwardedPropertySpec("_backend_kwargs", "_backend_state", "backend_kwargs"),
    ForwardedPropertySpec("backend", "_backend_state", "backend"),
    ForwardedPropertySpec("tts_backend", "_backend_state", "tts_backend"),
    ForwardedPropertySpec("_vc_backend_spec", "_backend_state", "vc_backend_spec"),
    ForwardedPropertySpec("vc_backend", "_backend_state", "vc_backend"),
    ForwardedPropertySpec(
        "voice_conversion_backend",
        "_backend_state",
        "voice_conversion_backend",
    ),
    ForwardedPropertySpec("vc_pitch_shift", "_backend_state", "vc_pitch_shift"),
    ForwardedPropertySpec("vc_f0_condition", "_backend_state", "vc_f0_condition"),
    ForwardedPropertySpec("input_mode", "_backend_state", "input_mode"),
    ForwardedPropertySpec("chunk_size", "_backend_state", "chunk_size"),
    ForwardedPropertySpec("language", "_backend_state", "language"),
    ForwardedPropertySpec("dev", "_backend_state", "dev"),
    ForwardedPropertySpec("use_normalization", "_backend_state", "use_normalization"),
    ForwardedPropertySpec("model", "_model_state", "model"),
    ForwardedPropertySpec("model_name", "_model_state", "model_name"),
    ForwardedPropertySpec("llm", "_model_state", "llm"),
    ForwardedPropertySpec("tokenizer", "_model_state", "tokenizer"),
    ForwardedPropertySpec("_last_warmup_error", "_model_state", "last_warmup_error"),
    ForwardedPropertySpec(
        "_normalizer_load_epoch",
        "_model_state",
        "normalizer_load_epoch",
    ),
    ForwardedPropertySpec("current_voice", "_voice_state", "current_voice"),
    ForwardedPropertySpec("current_character", "_voice_state", "current_character"),
    ForwardedPropertySpec(
        "current_character_persona",
        "_voice_state",
        "current_character_persona",
    ),
    ForwardedPropertySpec(
        "voice_bundle_is_default",
        "_voice_state",
        "voice_bundle_is_default",
    ),
    ForwardedPropertySpec("persona_history", "_voice_state", "persona_history"),
    ForwardedPropertySpec("persona_attachments", "_voice_state", "persona_attachments"),
    ForwardedPropertySpec("voices", "_voice_state", "voices"),
    ForwardedPropertySpec("voice_prompt", "_voice_state", "voice_prompt"),
    ForwardedPropertySpec("text_queue", "_pipeline_state", "text_queue"),
    ForwardedPropertySpec("audio_queue", "_pipeline_state", "audio_queue"),
    ForwardedPropertySpec("_playback_thread", "_pipeline_state", "playback_thread"),
    ForwardedPropertySpec("_generation_thread", "_pipeline_state", "generation_thread"),
    ForwardedPropertySpec("_api_thread", "_pipeline_state", "api_thread"),
    ForwardedPropertySpec("_persona_thread", "_pipeline_state", "persona_thread"),
    ForwardedPropertySpec("_queue_lock", "_pipeline_state", "queue_lock"),
    ForwardedPropertySpec(
        "_utterance_force_stop",
        "_pipeline_state",
        "utterance_force_stop",
    ),
    ForwardedPropertySpec(
        "_next_playback_source_id",
        "_pipeline_state",
        "next_playback_source_id",
    ),
    ForwardedPropertySpec(
        "_playback_source_statuses",
        "_pipeline_state",
        "playback_source_statuses",
    ),
    ForwardedPropertySpec(
        "_playback_source_meta", "_pipeline_state", "playback_source_meta"
    ),
    ForwardedPropertySpec(
        "_playback_progress_last_emit_at",
        "_pipeline_state",
        "playback_progress_last_emit_at",
    ),
    ForwardedPropertySpec(
        "_playback_progress_last_source_id",
        "_pipeline_state",
        "playback_progress_last_source_id",
    ),
    ForwardedPropertySpec("_model_ready", "_pipeline_state", "model_ready"),
    ForwardedPropertySpec("_playback_done", "_pipeline_state", "playback_done"),
    ForwardedPropertySpec("_say_lock", "_pipeline_state", "say_lock"),
    ForwardedPropertySpec("_wake_lock", "_pipeline_state", "wake_lock"),
    ForwardedPropertySpec("_model_lock", "_pipeline_state", "model_lock"),
    ForwardedPropertySpec("_exit_requested", "_pipeline_state", "exit_requested"),
    ForwardedPropertySpec("_stream", "_audio_state", "stream"),
    ForwardedPropertySpec("_current_sr", "_audio_state", "current_sr"),
    ForwardedPropertySpec("_audio_unavailable", "_audio_state", "audio_unavailable"),
    ForwardedPropertySpec("can_use_rubberband", "_audio_state", "can_use_rubberband"),
    ForwardedPropertySpec("speed", "_audio_state", "speed"),
    ForwardedPropertySpec(
        "smart_buffer_generation_speed",
        "_audio_state",
        "smart_buffer_generation_speed",
    ),
    ForwardedPropertySpec(
        "smart_buffer_target_seconds",
        "_audio_state",
        "smart_buffer_target_seconds",
    ),
    ForwardedPropertySpec(
        "total_generated_speech_seconds",
        "_audio_state",
        "total_generated_speech_seconds",
    ),
    ForwardedPropertySpec(
        "historical_generated_speech_seconds",
        "_audio_state",
        "historical_generated_speech_seconds",
    ),
    ForwardedPropertySpec("reverb", "_audio_state", "reverb"),
    ForwardedPropertySpec("recently_saved", "_audio_state", "recently_saved"),
    ForwardedPropertySpec("kept_sfx_audio", "_audio_state", "kept_sfx_audio"),
    ForwardedPropertySpec("regenerate", "_runtime_state", "regenerate"),
    ForwardedPropertySpec("locked", "_runtime_state", "locked"),
    ForwardedPropertySpec("loaded", "_runtime_state", "loaded"),
    ForwardedPropertySpec("sleeping", "_runtime_state", "sleeping"),
    ForwardedPropertySpec("_last_flavor", "_runtime_state", "last_flavor"),
    ForwardedPropertySpec("_ready_announced", "_runtime_state", "ready_announced"),
    ForwardedPropertySpec("_reload_pending", "_runtime_state", "reload_pending"),
    ForwardedPropertySpec("cur_state", "_runtime_state", "cur_state"),
    ForwardedPropertySpec("is_in_tutorial", "_runtime_state", "is_in_tutorial"),
    ForwardedPropertySpec("extension_manager", "_runtime_state", "extension_manager"),
    ForwardedPropertySpec("glow", "_runtime_state", "glow"),
    ForwardedPropertySpec("vision", "_runtime_state", "vision"),
    ForwardedPropertySpec("stream", "_audio_state", "stream"),
    ForwardedPropertySpec("say_lock", "_pipeline_state", "say_lock", read_only=True),
    ForwardedPropertySpec(
        "utterance_force_stop",
        "_pipeline_state",
        "utterance_force_stop",
        read_only=True,
    ),
    ForwardedPropertySpec(
        "queue_lock", "_pipeline_state", "queue_lock", read_only=True
    ),
    ForwardedPropertySpec(
        "playback_done", "_pipeline_state", "playback_done", read_only=True
    ),
    ForwardedPropertySpec(
        "model_ready", "_pipeline_state", "model_ready", read_only=True
    ),
    ForwardedPropertySpec(
        "generation_thread",
        "_pipeline_state",
        "generation_thread",
        read_only=True,
    ),
    ForwardedPropertySpec(
        "playback_thread",
        "_pipeline_state",
        "playback_thread",
        read_only=True,
    ),
    ForwardedPropertySpec(
        "exit_requested", "_pipeline_state", "exit_requested", read_only=True
    ),
    ForwardedPropertySpec(
        "model_lock", "_pipeline_state", "model_lock", read_only=True
    ),
    ForwardedPropertySpec(
        "audio_unavailable",
        "_audio_state",
        "audio_unavailable",
        read_only=True,
    ),
    ForwardedPropertySpec("current_sr", "_audio_state", "current_sr"),
)

CELUNE_CONSTANT_PROPERTIES = (
    ConstantPropertySpec(
        "force_stop_marker",
        PipelineStates.UTTERANCE_FORCE_END,
    ),
    ConstantPropertySpec(
        "utterance_done",
        PipelineStates.UTTERANCE_END,
    ),
    ConstantPropertySpec(
        "sentinel",
        PipelineStates.TERMINATE,
    ),
)
