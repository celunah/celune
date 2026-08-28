# SPDX-License-Identifier: Apache-2.0
"""Core Celune protocols and callback types."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Union, Optional, Protocol

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .aliases import LogLevel
from .modes import BackendMode, OperationMode
from .common import JSON, Config, JSONSerializable

if TYPE_CHECKING:
    import queue
    import threading

    import sounddevice as sd

    from ..audio.dsp import StreamingPedalboardReverb
    from .locks import ComponentLockOwner, ComponentBusyResult
    from ..locks import ComponentLockManager
    from ..chroma import AudioRGBGlow
    from .aliases import AudioChunk, AudioChunks
    from ..cevoice import CEVoicePersona
    from ..constants import PipelineStates
    from ..backends.vc import CeluneVCBackend
    from ..backends.tts import BackendModel, CeluneBackend
    from ..persona.impl import PersonaClient
    from ..persona.memory import PersonaMemoryStore
    from ..persona.emotion import PersonaEmotionAnalyzer
    from ..extensions.manager import CeluneExtensionManager


type GenerationKwarg = Union[torch.Tensor, int, bool, None]


class SupportsClose(Protocol):
    """Protocol for objects that can be closed."""

    def close(self) -> None:
        """Release any resources owned by the object.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class SupportsUnload(Protocol):
    """Protocol for objects that can unload their runtime state."""

    def unload(self) -> None:
        """Unload any optional runtime state owned by the object.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class Generative(Protocol):
    """Protocol for normalization-capable language models."""

    def generate(self, **kwargs: GenerationKwarg) -> torch.Tensor:
        """Generate token IDs from the provided model inputs.

        Args:
            kwargs: Backend-specific generation keyword arguments.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")

    def device(self) -> Union[torch.device, str]:
        """Return the device used by the generative model.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over the model parameters.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


type ReleasableObject = Union[
    SupportsClose,
    SupportsUnload,
    PreTrainedModel,
    PreTrainedTokenizerBase,
]


class NormalizerTokenizer(Protocol):
    """Tokenizer behavior CeluneNorm uses during normalization."""

    unk_token_id: Optional[int]
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]

    def convert_tokens_to_ids(self, tokens: str) -> Optional[int]:
        """Convert one token to its integer ID.

        Args:
            tokens: Token text to resolve into an integer ID.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> BatchEncoding:
        """Tokenize text for model input."""
        raise NotImplementedError("protocol not defined")

    def decode(
        self,
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> Union[str, list[str]]:
        """Decode generated token IDs.

        Args:
            token_ids: Generated token IDs to decode.
            skip_special_tokens: Whether special tokens should be omitted.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class MessageCallback(Protocol):
    """Callback accepting a message, severity, and log-level threshold."""

    def __call__(
        self,
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Handle a message emitted by Celune."""
        raise NotImplementedError("protocol not defined")


class InputStateCallback(Protocol):
    """Callback accepting an input lock state."""

    def __call__(self, locked: bool) -> None:
        """Handle input lock-state changes."""
        raise NotImplementedError("protocol not defined")


class VoiceLockStateCallback(Protocol):
    """Callback accepting a voice lock state."""

    def __call__(self, locked: bool) -> None:
        """Handle voice lock-state changes."""
        raise NotImplementedError("protocol not defined")


class ProgressCallback(Protocol):
    """Callback accepting progress and total values."""

    def __call__(self, progress: Optional[float], total: Optional[float]) -> None:
        """Handle a progress update emitted by Celune."""
        raise NotImplementedError("protocol not defined")


class CaptionCallback(Protocol):
    """Callback accepting the active speech caption, or ``None`` when finished."""

    def __call__(self, caption: Optional[str]) -> None:
        """Handle a speech caption lifecycle update."""
        raise NotImplementedError("protocol not defined")


class CaptionTimingCallback(Protocol):
    """Callback receiving generated speech for optional caption timing."""

    def __call__(
        self,
        caption: str,
        audio: AudioChunk,
        sample_rate: int,
        timing_text: Optional[str] = None,
    ) -> None:
        """Analyze generated speech to refine caption timing."""
        raise NotImplementedError("protocol not defined")


type ErrorCallback = Callable[[str], None]
type IdleCallback = Callable[[], None]
type QueueAvailableCallback = Callable[[], None]
type VoiceChangedCallback = Callable[[str], None]
type TTSBackendRecipe = Union[str, type[CeluneBackend]]
type VCBackendRecipe = Union[str, type[CeluneVCBackend]]
type TTSBackendSpec = Union[TTSBackendRecipe, CeluneBackend]
type VCBackendSpec = Union[VCBackendRecipe, CeluneVCBackend]
type CoreBackendSpec = Union[TTSBackendSpec, VCBackendSpec]


class CeluneStateAccessors:
    """Typed attribute surface exposed by ``Celune`` via forwarded properties."""

    log_callback: MessageCallback
    status_callback: MessageCallback
    error_callback: Callable[[str], None]
    idle_callback: Callable[[], None]
    queue_avail_callback: Callable[[], None]
    voice_changed_callback: Callable[[str], None]
    change_input_state_callback: InputStateCallback
    change_voice_lock_state_callback: VoiceLockStateCallback
    progress_callback: ProgressCallback
    caption_progress_callback: ProgressCallback
    caption_callback: CaptionCallback
    caption_timing_callback: CaptionTimingCallback
    config: Config
    backend_mode: BackendMode
    _backend_spec: Optional[TTSBackendRecipe]
    _backend_kwargs: dict[str, JSONSerializable]
    backend: CeluneBackend
    tts_backend: str
    _vc_backend_spec: Optional[VCBackendRecipe]
    vc_backend: Optional[CeluneVCBackend]
    voice_conversion_backend: str
    vc_pitch_shift: int
    vc_f0_condition: bool
    mode: OperationMode
    input_mode: str
    chunk_size: int
    language: str
    log_level: LogLevel
    use_normalization: bool
    model: Optional[BackendModel]
    model_name: str
    llm: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizerBase]
    _last_warmup_error: Optional[Exception]
    _normalizer_load_epoch: int
    current_voice: Optional[str]
    current_character: Optional[str]
    current_character_persona: Optional[CEVoicePersona]
    voice_bundle_is_default: bool
    persona_history: list[dict[str, str]]
    persona_session_summary: str
    persona_attachments: list[dict[str, str]]
    voices: tuple[str, ...]
    voice_prompt: Optional[str]
    text_queue: queue.Queue
    audio_queue: queue.Queue
    _playback_thread: Optional[threading.Thread]
    _generation_thread: Optional[threading.Thread]
    _api_thread: Optional[threading.Thread]
    _persona_thread: Optional[threading.Thread]
    _wake_background_thread: Optional[threading.Thread]
    _wake_background_lock: threading.Lock
    _persona_queue: queue.Queue
    _queue_lock: threading.Lock
    _utterance_force_stop: threading.Event
    _speech_generation: int
    _playback_generation: int
    _next_playback_source_id: int
    _playback_source_statuses: dict[int, str]
    _playback_source_meta: dict[int, dict[str, Union[str, float]]]
    _playback_progress_last_emit_at: float
    _playback_progress_last_source_id: int
    _model_ready: threading.Event
    _playback_done: threading.Event
    _say_lock: threading.Lock
    _wake_lock: threading.Lock
    _model_lock: threading.RLock
    _component_locks: ComponentLockManager
    _pipeline_lock_owner: Optional[ComponentLockOwner]
    _last_component_busy: Optional[ComponentBusyResult]
    _exit_requested: bool
    _stream: Optional[sd.OutputStream]
    _current_sr: Optional[int]
    _audio_unavailable: bool
    can_use_rubberband: bool
    speed: float
    smart_buffer_generation_speed: Optional[float]
    smart_buffer_target_seconds: float
    total_generated_speech_seconds: float
    historical_generated_speech_seconds: float
    reverb: StreamingPedalboardReverb
    recently_saved: Optional[str]
    kept_sfx_audio: Optional[AudioChunks]
    regenerate: bool
    locked: bool
    loaded: bool
    _reload_pending: bool
    sleeping: bool
    _last_flavor: Optional[str]
    _ready_announced: bool
    _closed: bool
    is_in_tutorial: bool
    extension_manager: Optional[CeluneExtensionManager]
    glow: AudioRGBGlow
    vision: Optional[PersonaClient]
    persona_emotion_analyzer: Optional[PersonaEmotionAnalyzer]
    persona_memory_store: Optional[PersonaMemoryStore]
    persona_ready: bool
    persona_loading: bool
    test_finished: bool
    test_result: Optional[JSON]
    _persona_load_thread: Optional[threading.Thread]
    _active_speech_generation: Optional[int]
    _webui_callbacks_wrapped: bool
    stream: Optional[sd.OutputStream]
    say_lock: threading.Lock
    utterance_force_stop: threading.Event
    queue_lock: threading.Lock
    force_stop_marker: PipelineStates
    playback_done: threading.Event
    model_ready: threading.Event
    utterance_done: PipelineStates
    sentinel: PipelineStates
    generation_thread: Optional[threading.Thread]
    playback_thread: Optional[threading.Thread]
    exit_requested: bool
    model_lock: threading.RLock
    component_locks: ComponentLockManager
    last_component_busy: Optional[ComponentBusyResult]
    audio_unavailable: bool
    current_sr: Optional[int]

    @property
    def cur_state(self) -> str:
        """Return the current runtime-state label.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("typing surface only")

    @cur_state.setter
    def cur_state(self, value: str) -> None:
        """Store the current runtime-state label.

        Args:
            value: The runtime-state label to store.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("typing surface only")

    @property
    def persona_queue(self) -> queue.Queue:
        """Return the queue receiving Persona input text.

        Returns:
            queue.Queue: Celune's current Persona queue.
        """
        return self._persona_queue

    @property
    def speech_generation(self) -> int:
        """Return the current speech-generation counter.

        Returns:
            int: Celune's current speech-generation counter.
        """
        return self._speech_generation


class _BundleWithPath(Protocol):  # noqa: PYI046
    """Protocol for bundle-like objects that expose a path."""

    @property
    def path(self) -> Union[str, Path]:
        """Return the bundle path."""
