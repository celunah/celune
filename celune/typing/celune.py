"""Core Celune protocols and callback types."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .common import Config, JSONSerializable

if TYPE_CHECKING:
    import queue
    import threading

    import numpy as np
    import numpy.typing as npt
    import sounddevice as sd

    from ..backends import BackendModel, CeluneBackend
    from ..cevoice import CEVoicePersona
    from ..chroma import AudioRGBGlow
    from ..constants import PipelineStates
    from ..dsp import StreamingPedalboardReverb
    from ..extensions.manager import CeluneExtensionManager
    from ..persona.impl import PersonaClient


class SupportsClose(Protocol):
    """Protocol for objects that can be closed."""

    def close(self) -> None:
        """Release any resources owned by the object."""
        raise NotImplementedError("protocol not defined")


class SupportsUnload(Protocol):
    """Protocol for objects that can unload their runtime state."""

    def unload(self) -> None:
        """Unload any optional runtime state owned by the object."""
        raise NotImplementedError("protocol not defined")


class Generative(Protocol):
    """Protocol for normalization-capable language models."""

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Generate token IDs from the provided model inputs.

        Args:
            kwargs: Backend-specific generation keyword arguments.
        """
        raise NotImplementedError("protocol not defined")

    def device(self) -> Union[torch.device, str]:
        """Return the device used by the generative model."""
        raise NotImplementedError("protocol not defined")

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over the model parameters."""
        raise NotImplementedError("protocol not defined")


ReleasableObject = Union[
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
        """
        raise NotImplementedError("protocol not defined")


class MessageCallback(Protocol):
    """Callback accepting a message and optional severity."""

    def __call__(self, msg: str, severity: str = "info") -> None:
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


ErrorCallback = Callable[[str], None]
IdleCallback = Callable[[], None]
QueueAvailableCallback = Callable[[], None]
VoiceChangedCallback = Callable[[str], None]


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
    config: Config
    _backend_spec: Optional[Union[str, type["CeluneBackend"]]]
    _backend_kwargs: dict[str, JSONSerializable]
    backend: "CeluneBackend"
    tts_backend: str
    chunk_size: int
    language: str
    dev: bool
    use_normalization: bool
    model: Optional["BackendModel"]
    model_name: str
    llm: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizerBase]
    _last_warmup_error: Optional[Exception]
    _normalizer_load_epoch: int
    current_voice: Optional[str]
    current_character: Optional[str]
    current_character_persona: Optional["CEVoicePersona"]
    voice_bundle_is_default: bool
    persona_history: list[dict[str, str]]
    persona_attachments: list[dict[str, str]]
    voices: tuple[str, ...]
    voice_prompt: Optional[str]
    text_queue: "queue.Queue"
    audio_queue: "queue.Queue"
    _playback_thread: Optional["threading.Thread"]
    _generation_thread: Optional["threading.Thread"]
    _api_thread: Optional["threading.Thread"]
    _persona_thread: Optional["threading.Thread"]
    _queue_lock: "threading.Lock"
    _utterance_force_stop: "threading.Event"
    _next_playback_source_id: int
    _playback_source_statuses: dict[int, str]
    _playback_source_meta: dict[int, dict[str, Union[str, float]]]
    _playback_progress_last_emit_at: float
    _playback_progress_last_source_id: int
    _model_ready: "threading.Event"
    _playback_done: "threading.Event"
    _say_lock: "threading.Lock"
    _wake_lock: "threading.Lock"
    _model_lock: "threading.RLock"
    _exit_requested: bool
    _stream: Optional["sd.OutputStream"]
    _current_sr: Optional[int]
    _audio_unavailable: bool
    can_use_rubberband: bool
    speed: float
    reverb: "StreamingPedalboardReverb"
    recently_saved: Optional[str]
    kept_sfx_audio: Optional["npt.NDArray[np.float32]"]
    regenerate: bool
    locked: bool
    loaded: bool
    sleeping: bool
    _last_flavor: Optional[str]
    _ready_announced: bool
    cur_state: str
    is_in_tutorial: bool
    extension_manager: Optional["CeluneExtensionManager"]
    glow: "AudioRGBGlow"
    vision: Optional["PersonaClient"]
    stream: Optional["sd.OutputStream"]
    say_lock: "threading.Lock"
    utterance_force_stop: "threading.Event"
    queue_lock: "threading.Lock"
    force_stop_marker: "PipelineStates"
    playback_done: "threading.Event"
    model_ready: "threading.Event"
    utterance_done: "PipelineStates"
    sentinel: "PipelineStates"
    generation_thread: Optional["threading.Thread"]
    playback_thread: Optional["threading.Thread"]
    exit_requested: bool
    model_lock: "threading.RLock"
    audio_unavailable: bool
    current_sr: Optional[int]
