# SPDX-License-Identifier: Apache-2.0
"""Celune's public engine facade."""

from pathlib import Path
from collections.abc import Mapping, Callable
from typing import Never, Union, ClassVar, Optional, final

import numpy as np

from .config import (
    Config,
)
from . import vc as core_vc
from .typing.agent import (
    AgentTool,
    AgentTaskConfig,
    AgentToolSchema,
    AgentToolSelector,
    AgentClassificationResult,
)
from .typing.celune import (
    CaptionCallback,
    CoreBackendSpec,
    MessageCallback,
    ProgressCallback,
    InputStateCallback,
    CeluneStateAccessors,
    VoiceLockStateCallback,
    CaptionProgressCallback,
)
from .dataclasses.celune import (
    CELUNE_CONSTANT_PROPERTIES,
    CELUNE_FORWARDED_PROPERTIES,
)
from .dataclasses.events import (
    StateChangedEvent,
)
from .typing.aliases import LogLevel
from .core import agent as core_agent
from .core import sleep as core_sleep
from .core import voice as core_voice
from .dataclasses.properties import (
    bind_constant_properties,
    bind_forwarded_properties,
)
from .typing.modes import BackendMode
from .core import reload as core_reload
from .core import speech as core_speech
from .core import loading as core_loading
from .agent.routing import AgentInputRouter
from .core import callbacks as core_callbacks
from .core import lifecycle as core_lifecycle
from .typing.pipeline import SpeechStreamQueue
from .typing.common import JSON, JSONSerializable
from .typing.events import EventName, EventPayload
from .core import conversation as core_conversation
from .constants import DEFAULT_TARGET_CHUNK_LENGTH_SECONDS
from .dataclasses.pipeline import AudioOutput, CaptionPlaybackState

npt = np.typing


def tts_quantization_enabled(
    config: Optional[Mapping[str, JSONSerializable]],
    backend_spec: CoreBackendSpec,
) -> bool:
    """Resolve the default TTS quantization policy, excluding LuxTTS."""
    return core_lifecycle.tts_quantization_enabled(config, backend_spec)


@final
class Celune(CeluneStateAccessors):
    """The character engine for Celune."""

    _normalizer_load_epoch: int
    _next_playback_source_id: int
    _playback_source_statuses: dict[int, str]
    _playback_source_meta: dict[int, dict[str, Union[str, float]]]
    _playback_caption_states: dict[int, CaptionPlaybackState]

    def __init_subclass__(cls, **kwargs: Never) -> Never:
        raise TypeError(f"{__class__.__name__} is final and cannot be subclassed")

    _instance: ClassVar[Optional["Celune"]] = None

    @property
    def next_playback_source_id(self) -> int:
        """Return the most recently allocated playback source identifier."""
        return getattr(self, "_next_playback_source_id", 0)

    @next_playback_source_id.setter
    def next_playback_source_id(self, value: int) -> None:
        """Store the most recently allocated playback source identifier."""
        self._next_playback_source_id = value

    @property
    def playback_source_statuses(self) -> dict[int, str]:
        """Return playback status labels keyed by source identifier."""
        statuses = getattr(self, "_playback_source_statuses", None)
        if statuses is None:
            statuses = {}
            self._playback_source_statuses = statuses
        return statuses

    @playback_source_statuses.setter
    def playback_source_statuses(self, value: dict[int, str]) -> None:
        """Replace the playback status map."""
        self._playback_source_statuses = value

    @property
    def playback_source_meta(self) -> dict[int, dict[str, Union[str, float]]]:
        """Return metadata for queued playback sources."""
        metadata = getattr(self, "_playback_source_meta", None)
        if metadata is None:
            metadata = {}
            self._playback_source_meta = metadata
        return metadata

    @playback_source_meta.setter
    def playback_source_meta(
        self,
        value: dict[int, dict[str, Union[str, float]]],
    ) -> None:
        """Replace the playback source metadata map."""
        self._playback_source_meta = value

    @property
    def playback_caption_states(self) -> dict[int, CaptionPlaybackState]:
        """Return caption timing state keyed by playback source identifier."""
        states = getattr(self, "_playback_caption_states", None)
        if states is None:
            states = {}
            self._playback_caption_states = states
        return states

    @playback_caption_states.setter
    def playback_caption_states(
        self,
        value: dict[int, CaptionPlaybackState],
    ) -> None:
        """Replace the caption timing state map."""
        self._playback_caption_states = value

    @property
    def agent_tools(self) -> tuple[AgentTool, ...]:
        """Return the tools registered with this engine's local agent runtime."""
        return self._agent_tools

    @agent_tools.setter
    def agent_tools(self, tools: tuple[AgentTool, ...]) -> None:
        """Replace the registered agent tools for controlled runtime setup."""
        self._agent_tools = tools

    @property
    def agent_tool_schemas(self) -> Mapping[str, AgentToolSchema]:
        """Return the validated schemas advertised by the local agent runtime."""
        return self._agent_tool_schemas

    @agent_tool_schemas.setter
    def agent_tool_schemas(self, schemas: Mapping[str, AgentToolSchema]) -> None:
        """Replace the registered schemas during controlled runtime setup."""
        self._agent_tool_schemas = schemas

    @property
    def agent_router(self) -> AgentInputRouter:
        """Return the input router bound to this engine's agent runtime."""
        return self._agent_router

    @agent_router.setter
    def agent_router(self, router: AgentInputRouter) -> None:
        """Replace the router when the engine installs a test runtime."""
        self._agent_router = router

    @property
    def agent_needle_selector(self) -> Optional[AgentToolSelector]:
        """Return the optional external tool selector configured for this engine."""
        return self._agent_needle_selector

    @agent_needle_selector.setter
    def agent_needle_selector(
        self,
        selector: Optional[AgentToolSelector],
    ) -> None:
        """Replace the optional selector during controlled runtime setup."""
        self._agent_needle_selector = selector

    def __init__(
        self,
        config: Config,
        backend: Optional[CoreBackendSpec] = None,
        tts_backend: Optional[CoreBackendSpec] = None,
        vc_backend: Optional[CoreBackendSpec] = None,
        vc_f0_condition: Optional[bool] = None,
        input_mode: Optional[str] = None,
        chunk_size: int = 0,  # defaulted to 0 because not all backends use this
        target_chunk_length: float = DEFAULT_TARGET_CHUNK_LENGTH_SECONDS,
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
        core_lifecycle.initialize(
            self,
            config,
            backend,
            tts_backend,
            vc_backend,
            vc_f0_condition,
            input_mode,
            chunk_size,
            target_chunk_length,
            language,
            log_callback,
            status_callback,
            error_callback,
            idle_callback,
            queue_avail_callback,
            voice_changed_callback,
            change_input_state_callback,
            change_voice_lock_state_callback,
            progress_callback,
            caption_progress_callback,
            caption_callback,
            log_level,
            agent_tool_selector,
            backend_mode,
            startup_callback,
        )

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
        event = self.update_cur_state(value)
        if event is not None:
            self.emit_event("state_changed", event)

    def update_cur_state(self, value: str) -> Optional[StateChangedEvent]:
        """Store a runtime state and return its event without dispatching it.

        Args:
            value: The new runtime-state label to store.

        Returns:
            Optional[StateChangedEvent]: The transition event, if the state changed.
        """
        if self.test_finished and value != "stopped":
            return None
        old_state = self._runtime_state.cur_state
        self._runtime_state.cur_state = value
        if old_state == value:
            return None
        self.log(
            f"[STATE] transition old={old_state} new={value}",
            loglevel="debug",
        )
        return StateChangedEvent(
            celune=self,
            old_state=old_state,
            new_state=value,
        )

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
        return core_lifecycle.finish_test_mode(
            self, mode, success, task_state=task_state, detail=detail, checks=checks
        )

    def emit_event(self, event_name: EventName, event: EventPayload) -> None:
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

    def is_voice_conversion_mode(self) -> bool:
        """Return whether this Celune instance is in voice-conversion mode.

        Returns:
            bool: Whether this Celune instance is currently in voice conversion mode.
        """
        return core_lifecycle.is_voice_conversion_mode(self)

    def unload_runtime_state(
        self,
        include_normalizer: bool = False,
        include_vc: bool = True,
        close_backends: bool = False,
        release_cuda_cache: bool = True,
    ) -> None:
        """Unload runtime models while reserving the lifecycle resource."""
        return core_lifecycle.unload_runtime_state(
            self,
            include_normalizer=include_normalizer,
            include_vc=include_vc,
            close_backends=close_backends,
            release_cuda_cache=release_cuda_cache,
        )

    def raise_warmup_error(self, message: str) -> None:
        """Raise a Celune warmup error while preserving its original cause."""
        return core_reload.raise_warmup_error(self, message)

    def wait_until_idle(
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
        return core_reload.wait_until_idle(
            self,
            timeout,
            wait_for_speech=wait_for_speech,
        )

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
        return core_reload.with_backend(self, backend_spec, timeout)

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
        return core_reload.with_cevoice(self, bundle, timeout)

    def api_settings(self) -> tuple[bool, str, int, Optional[str], int, str]:
        """Resolve API access, authentication, port, and rate-limit settings."""
        return core_lifecycle.api_settings(self)

    # as of CeluneNorm 2.0, normalization ACTUALLY works with long inputs

    def stop_live_audio(self) -> None:
        """Reset any state held by the active live voice-conversion backend."""
        return core_speech.stop_live_audio(self)

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
        return core_speech.play(
            self, sound_path, keep=keep, volume=volume, on_started=on_started
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
        return core_speech.play_audio(self, audio, sample_rate, label, keep=keep)

    def close(self) -> None:
        """Shut off Celune and release loaded runtime state."""
        return core_lifecycle.close(self)

    def fatal(self, *, semi_fatal: bool = False) -> None:
        """Mark Celune state as fatal and prevent further operations.

        Args:
            semi_fatal: Whether the failure came from configuration and permits
                opening the settings manager.
        """
        return core_lifecycle.fatal(self, semi_fatal=semi_fatal)

    def run_pipeline_jobs(self) -> None:
        """
        Start Celune's async speech pipeline inside one engine thread.
        """
        return core_lifecycle.run_pipeline_jobs(self)

    def change_voice(self, voice: "str") -> "None":
        """
        Change Celune's voice parameters.

        Args:
            voice: The voice name to load and warm up.

        Raises:
            WarmupError: The newly loaded voice fails warmup.
        """
        return core_voice.change_voice(self, voice)

    def effective_voice_prompt(self) -> "Optional[str]":
        """
        Return the active voice prompt only when the current model supports it.
        """
        return core_voice.effective_voice_prompt(self)

    def force_stop_speech(self) -> "bool":
        """
        Forcefully stop Celune from speaking.

        Returns:
            bool: ``True`` when an active utterance was interrupted, otherwise ``False``.
        """
        return core_voice.force_stop_speech(self)

    def load_available_voices(self) -> "bool":
        """
        Load the active voice set appropriate for the selected backend.

        Returns:
            bool: ``True`` when at least one voice is available.
        """
        return core_voice.load_available_voices(self)

    def load_voice_bundle(self, bundle: "Optional[Union[str, Path]]" = None) -> "bool":
        """
        Select and load a CEVOICE bundle into Celune's active voice set.

        Args:
            bundle: A built-in bundle name, explicit bundle path, or ``None`` to use Celune's default
                bundle.

        Returns:
            bool: ``True`` when a CEVOICE bundle was loaded, otherwise ``False``.
        """
        return core_voice.load_voice_bundle(self, bundle)

    def log(
        self, msg: "str", severity: "str" = "info", *, loglevel: "LogLevel" = "info"
    ) -> "None":
        """
        Log a message.

        Args:
            msg: The message to emit.
            severity: The message severity level.
            loglevel: The minimum configured log level required to emit the message.
        """
        return core_callbacks.log(self, msg, severity, loglevel=loglevel)

    def log_dev(self, msg: "str", severity: "str" = "info") -> "None":
        """
        Log a legacy developer message at verbose level.

        Args:
            msg: The message to emit.
            severity: The message severity level.
        """
        return core_callbacks.log_dev(self, msg, severity)

    def reset_persona_conversation(self) -> "None":
        """
        Clear Persona conversation context when its character changes.
        """
        return core_voice.reset_persona_conversation(self)

    def set_backend(self, backend_spec: "CoreBackendSpec") -> "bool":
        """
        Request a hot reload into another TTS or VC backend.

        Args:
            backend_spec: The backend name, type, or instance to activate.

        Returns:
            bool: ``True`` when the reload worker was started.
        """
        return core_voice.set_backend(self, backend_spec)

    def set_backend_and_wait(
        self, backend_spec: "CoreBackendSpec", timeout: "Optional[float]" = None
    ) -> "bool":
        """
        Request a hot backend reload and wait for completion.

        Args:
            backend_spec: The backend name, type, or instance to activate.
            timeout: Optional maximum seconds to wait for the reload to finish. Pass ``None`` to wait until
                the reload completes.

        Returns:
            bool: ``True`` when the requested backend finished loading.
        """
        return core_voice.set_backend_and_wait(self, backend_spec, timeout)

    async def set_backend_async(
        self, backend_spec: "CoreBackendSpec", timeout: "Optional[float]" = None
    ) -> "bool":
        """
        Request a hot backend reload without blocking the caller's event loop.

        Args:
            backend_spec: Backend specification to load.
            timeout: Maximum time to wait for the reload.
        """
        return await core_voice.set_backend_async(self, backend_spec, timeout)

    def set_cevoice(self, bundle: "Optional[Union[str, Path]]") -> "bool":
        """
        Request a hot reload into another CEVOICE bundle.

        Args:
            bundle: The CEVOICE bundle name or path to activate.

        Returns:
            bool: ``True`` when the reload worker was started.
        """
        return core_voice.set_cevoice(self, bundle)

    def set_cevoice_and_wait(
        self, bundle: "Optional[Union[str, Path]]", timeout: "Optional[float]" = None
    ) -> "bool":
        """
        Request a hot CEVOICE reload and wait for completion.

        Args:
            bundle: The CEVOICE bundle name or path to activate.
            timeout: Optional maximum seconds to wait for the reload to finish. Pass ``None`` to wait until
                the reload completes.

        Returns:
            bool: ``True`` when the requested CEVOICE pack finished loading.
        """
        return core_voice.set_cevoice_and_wait(self, bundle, timeout)

    async def set_cevoice_async(
        self, bundle: "Optional[Union[str, Path]]", timeout: "Optional[float]" = None
    ) -> "bool":
        """
        Request a hot CEVOICE reload without blocking the caller's event loop.

        Args:
            bundle: CEVOICE bundle path or ``None`` for the default bundle.
            timeout: Maximum time to wait for the reload.
        """
        return await core_voice.set_cevoice_async(self, bundle, timeout)

    def set_voice(self, name: "str") -> "bool":
        """
        Extension method for changing Celune's voice.

        Args:
            name: The voice name to load.

        Returns:
            bool: ``True`` when the reload thread was started, otherwise ``False``.
        """
        return core_voice.set_voice(self, name)

    def set_voice_and_wait(self, name: "str", timeout: "float" = 30.0) -> "bool":
        """
        Change Celune's voice and wait until the reload finishes.

        Args:
            name: The voice name to load.
            timeout: How long to wait before considering the reload a failure.

        Returns:
            bool: ``True`` when the requested voice finished loading, otherwise ``False``.
        """
        return core_voice.set_voice_and_wait(self, name, timeout)

    async def set_voice_async(self, name: "str", timeout: "float" = 30.0) -> "bool":
        """
        Change Celune's voice without blocking the caller's event loop.

        Args:
            name: Voice name to load.
            timeout: Maximum time to wait for the reload.

        Raises:
            Exception: If the asynchronous reload worker raises unexpectedly.
        """
        return await core_voice.set_voice_async(self, name, timeout)

    def set_voices(self, voices: "tuple[str, ...]") -> "None":
        """
        Configure Celune's voice information.

        Args:
            voices: The list of available voice names.
        """
        return core_voice.set_voices(self, voices)

    def setup_extensions(self) -> "None":
        """
        Configure Celune's extension manager.
        """
        return core_voice.setup_extensions(self)

    def sleep(self, seconds: "float") -> "None":
        """
        Pause the current extension worker for a number of seconds.

        Args:
            seconds: Number of seconds to pause the current thread.
        """
        return core_voice.sleep(self, seconds)

    def status(self, msg: "str", severity: "str" = "info") -> "None":
        """
        Update Celune's status display.

        Args:
            msg: The status message to show.
            severity: The status severity level.
        """
        return core_callbacks.status(self, msg, severity)

    def try_play_signal(self, signal_type: "str") -> "bool":
        """
        Public interface for Celune._try_play_signal.

        Args:
            signal_type: The signal type to play.

        Returns:
            bool: ``True`` when the requested signal was played, otherwise ``False``.

        Raises:
            ValueError: An invalid signal was requested.
        """
        return core_voice.try_play_signal(self, signal_type)

    def voice_prompt_supported(self) -> "bool":
        """
        Return whether the active TTS configuration supports voice prompts.

        Returns:
            bool: Whether the currently loaded TTS model supports voice prompting.
        """
        return core_voice.voice_prompt_supported(self)

    async def wait_until_idle_async(
        self, timeout: "float" = 30.0, *, wait_for_speech: "Optional[bool]" = None
    ) -> "bool":
        """
        Wait until model reload and playback completion without blocking the event loop.

        Args:
            timeout: Maximum time to wait for model and playback readiness.
            wait_for_speech: Whether active speech must finish before returning.
        """
        return await core_voice.wait_until_idle_async(
            self, timeout, wait_for_speech=wait_for_speech
        )

    async def enter_sleep_mode_async(self) -> "bool":
        """
        Put Celune to sleep without blocking the caller's event loop.
        """
        return await core_loading.enter_sleep_mode_async(self)

    async def force_stop_speech_async(self) -> "bool":
        """
        Forcefully stop Celune from speaking without blocking an async caller.
        """
        return await core_loading.force_stop_speech_async(self)

    def load(
        self, raise_on_error: "bool" = False, skip_runtime_check: "bool" = False
    ) -> "bool":
        """
        Load and initialize Celune.

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
        return core_loading.load(self, raise_on_error, skip_runtime_check)

    def load_normalizer(self) -> "None":
        """
        Load the normalizer LLM.
        """
        return core_loading.load_normalizer(self)

    def normalize(self, text: "str") -> "Optional[str]":
        """
        Normalize input text using CeluneNorm.

        Args:
            text: The raw text to normalize before speech generation.
        """
        return core_loading.normalize(self, text)

    async def wake_from_sleep_async(self) -> "bool":
        """
        Wake Celune without blocking the caller's event loop.
        """
        return await core_loading.wake_from_sleep_async(self)

    def enter_sleep_mode(self) -> "bool":
        """
        Put Celune to sleep and unload models according to configuration.

        Returns:
            bool: Whether Celune was put to sleep.
        """
        return core_sleep.enter_sleep_mode(self)

    def sleep_enabled(self) -> "bool":
        """
        Return whether automatic sleep mode is enabled.

        Returns:
            bool: Whether automatic sleep mode is enabled.
        """
        return core_sleep.sleep_enabled(self)

    def sleep_timeout_seconds(self) -> "float":
        """
        Return the configured idle timeout in seconds.

        Returns:
            float: The configured idle timeout in seconds.
        """
        return core_sleep.sleep_timeout_seconds(self)

    def unload_normalizer_state(self, release_cuda_cache: "bool" = True) -> "None":
        """
        Unload only CeluneNorm components and release unused memory.

        Args:
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """
        return core_sleep.unload_normalizer_state(self, release_cuda_cache)

    def wake_from_sleep(self) -> "bool":
        """
        Wake Celune and reload anything unloaded by sleep mode.

        Returns:
            bool: Whether Celune was woken up from sleep.

        Raises:
            NotAvailableError: Celune has no valid model ID to reload after waking up.
            WarmupError: Celune cannot warm up after waking up.
        """
        return core_sleep.wake_from_sleep(self)

    @property
    def agent_needle_error(self) -> "Optional[str]":
        """
        Return the latest production Needle loading failure, if any.
        """
        return core_agent.agent_needle_error(self)

    @property
    def agent_needle_ready(self) -> "bool":
        """
        Return whether the production Needle selector is loaded and usable.
        """
        return core_agent.agent_needle_ready(self)

    def classify_input(
        self, text: "str", *, persona_ready: "Optional[bool]" = None
    ) -> "AgentClassificationResult":
        """
        Classify one input without creating a task or starting execution.

        Args:
            text: The text or transcription to classify.
            persona_ready: Whether the loaded Persona model may resolve ambiguity.

        Returns:
            AgentClassificationResult: The typed conversation-first classification.
        """
        return core_agent.classify_input(self, text, persona_ready=persona_ready)

    def route_input(
        self, text: "str", *, persona_ready: "Optional[bool]" = None
    ) -> "AgentClassificationResult":
        """
        Route one input through the active task or ordinary Persona conversation.

        Args:
            text: The text or transcription to route.
            persona_ready: Whether the loaded Persona model may resolve ambiguity.

        Returns:
            AgentClassificationResult: The typed route selected for the input.
        """
        return core_agent.route_input(self, text, persona_ready=persona_ready)

    def run_agent_route(self, route: "AgentClassificationResult") -> "bool":
        """
        Consume a routed task through the shared agent runtime.
        """
        return core_agent.run_agent_route(self, route)

    def say(
        self,
        text: "str",
        save: "bool" = True,
        display_text: "Optional[str]" = None,
    ) -> "bool":
        """
        Queue text for Celune to say.

        Args:
            text: The input text to queue for synthesis.
            save: Whether to save generated output artifacts.
            display_text: Optional text to show in logs instead of the synthesis text.

        Returns:
            bool: ``True`` when the text was queued successfully, otherwise ``False``.

        Raises:
            Exception: Re-raised after releasing the pipeline if queueing fails.
        """
        return core_speech.say(self, text, save, display_text)

    async def say_async(
        self,
        text: "str",
        save: "bool" = True,
        display_text: "Optional[str]" = None,
    ) -> "bool":
        """
        Queue text for Celune to say without blocking an async caller.

        Args:
            text: The text to synthesize.
            save: Whether the generated utterance should be persisted to disk.
            display_text: Optional UI-facing text to associate with the request.

        Returns:
            bool: ``True`` when the request was queued successfully, otherwise ``False``.
        """
        return await core_speech.say_async(self, text, save, display_text)

    def say_stream(
        self, text: "str", save: "bool" = True
    ) -> "Optional[SpeechStreamQueue]":
        """
        Queue text for playback and mirror generated chunks to a queue.
        """
        return core_speech.say_stream(self, text, save)

    async def say_stream_async(
        self, text: "str", save: "bool" = True
    ) -> "Optional[SpeechStreamQueue]":
        """
        Queue text for playback and mirror chunks without blocking an async caller.
        """
        return await core_speech.say_stream_async(self, text, save)

    def submit_audio(
        self,
        audio: "np.ndarray",
        sample_rate: "int",
        label: "str" = "audio input",
        f0_condition: "Optional[bool]" = None,
        log_playback: "bool" = True,
        reset_ready_announcement: "bool" = True,
    ) -> "bool":
        """
        Accept audio input for the active engine mode.
        """
        return core_speech.submit_audio(
            self,
            audio,
            sample_rate,
            label,
            f0_condition,
            log_playback,
            reset_ready_announcement,
        )

    def release_pipeline(self, playback_idle: "bool" = True) -> "None":
        """
        Release Celune's shared playback pipeline.

        Args:
            playback_idle: Whether playback should be marked fully idle now.
        """
        return core_speech.release_pipeline(self, playback_idle)

    def think(self: "Celune", text: "str") -> "bool":
        """
        Let Celune reply to an input request.

        Args:
            text: The request that will be sent to Persona for processing.

        Returns:
            bool: ``True`` if Celune processed this smart request, otherwise ``False``.
        """
        return core_conversation.think(self, text)

    async def think_async(self: "Celune", text: "str") -> "bool":
        """
        Let Celune reply to one input request without blocking an async caller.

        Args:
            text: Input text for Persona to answer.

        Returns:
            bool: ``True`` when the response was queued successfully.
        """
        return await core_conversation.think_async(self, text)

    def convert_audio(
        self,
        audio: "np.ndarray",
        sample_rate: "int",
        label: "str" = "audio input",
        f0_condition: "Optional[bool]" = None,
    ) -> "Optional[AudioOutput]":
        """
        Convert submitted audio and return the generated VC output.
        """
        return core_vc.convert_audio(self, audio, sample_rate, label, f0_condition)

    def convert_live_audio(
        self,
        audio: "np.ndarray",
        sample_rate: "int",
        label: "str" = "audio input",
        f0_condition: "Optional[bool]" = None,
    ) -> "Optional[AudioOutput]":
        """
        Convert one low-latency live block through the active VC backend.
        """
        return core_vc.convert_live_audio(self, audio, sample_rate, label, f0_condition)


def agent_task_config(config: Config) -> AgentTaskConfig:
    """Build the agent task limits from the nested ``agent`` settings."""
    return core_lifecycle.agent_task_config(config)
