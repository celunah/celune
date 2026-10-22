# SPDX-License-Identifier: Apache-2.0
"""CEVOICE and backend handler methods."""

from __future__ import annotations
import time
import asyncio
import threading
from pathlib import Path
from typing import Union, Optional

from .lifecycle import config_str
from . import reload as core_reload
from . import lifecycle as core_lifecycle

from .. import __version__
from ..constants import APP_NAME
from ..paths import project_root
from ..config import config_value
from ..i18n import string, tagged_string
from ..utils import format_error_message
from ..threads import run_in_daemon_thread
from ..extensions.base import CeluneContext
from ..pipeline.playback import play_signal
from ..registry.tts import QWEN3_06B_MODEL_ID
from ..backends.tts import BACKENDS, CeluneBackend
from ..dataclasses.events import VoiceChangedEvent
from ..backends.vc import VC_BACKENDS, CeluneVCBackend
from ..extensions.manager import CeluneExtensionManager
from ..typing.celune import BundleWithPath, CoreBackendSpec
from ..pipeline.playback import force_stop_speech as force_stop_pipeline
from ..cevoice import (
    default_loader,
    active_bundle_path,
    resolve_bundle_path,
    select_voice_bundle,
    bundle_character_name,
    persona_metadata_from_manifest,
    bundle_matches_default_pack_checksum,
)


def set_voices(self, voices: tuple[str, ...]) -> None:
    """Configure Celune's voice information.

    Args:
        voices: The list of available voice names.
    """
    self.voices = voices


def reset_persona_conversation(self) -> None:
    """Clear Persona conversation context when its character changes."""
    self.persona_history.clear()
    self.persona_session_summary = ""
    self.persona_attachments.clear()


def _bundle_path_string(bundle: Optional[BundleWithPath]) -> Optional[str]:
    """Return one bundle path as a string when it is available."""
    return None if bundle is None else str(bundle.path)


def load_voice_bundle(self, bundle: Optional[Union[str, Path]] = None) -> bool:
    """Select and load a CEVOICE bundle into Celune's active voice set.

    Args:
        bundle: A built-in bundle name, explicit bundle path, or ``None`` to use Celune's default bundle.

    Returns:
        bool: ``True`` when a CEVOICE bundle was loaded, otherwise ``False``.
    """
    previous_loader = default_loader()
    previous_bundle_path = (
        _bundle_path_string(previous_loader.bundle)
        if previous_loader is not None
        else None
    )
    previous_character = self.current_character
    select_voice_bundle(bundle)
    loader = default_loader()
    if loader is None:
        if previous_bundle_path is not None or previous_character is not None:
            self.reset_persona_conversation()
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
        core_reload.emit_character_event_transition(
            self,
            previous_character,
            previous_bundle_path,
            None,
            None,
            True,
        )
        return bool(voices)

    new_bundle_path = _bundle_path_string(loader.bundle)
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
        preferred_voice if preferred_voice in voices else voices[0] if voices else None
    )
    if (
        previous_bundle_path != new_bundle_path
        or previous_character != self.current_character
    ):
        self.reset_persona_conversation()
    core_reload.emit_character_event_transition(
        self,
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
            config_str(config_value(self.config, "voice_bundle"))
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
    if not _prepare_voice_change(self, name):
        return False

    threading.Thread(
        target=self.change_voice,
        args=(name,),
        daemon=True,
    ).start()
    return True


def _prepare_voice_change(self, name: str) -> bool:
    """Wait for speech to drain before preparing one voice switch."""
    if getattr(self.backend, "singing", False):
        self.log(string("celune.singing_voice_change_unavailable"), "warning")
        return False

    if name not in self.voices:
        # this voice was not found in the current CEVOICE/CECHAR pack
        self.log(string("celune.unknown_voice", voice=name), "warning")
        return False

    self.change_input_state_callback(locked=True)
    wait_for_speech = speech_playback_active(self)
    previous_state = self.cur_state
    self.cur_state = "reloading"

    if not self._model_ready.is_set():
        self.log(string("celune.waiting_for_models"))
    if not core_reload.wait_until_idle(self, wait_for_speech=wait_for_speech):
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
    if not set_voice(self, name):
        return False

    if not self._model_ready.wait(timeout=timeout):
        self.log(string("celune.voice_switch_timeout"), "warning")
        return False
    return _voice_switch_succeeded(self, name)


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
    await run_in_daemon_thread(self._async_runtime_lock.acquire)
    try:
        with self._voice_reload_guard:
            if self._voice_reload_active:
                self.log(string("celune.reload_already_in_progress"), "warning")
                return False
            self._voice_reload_active = True
        if not await run_in_daemon_thread(self._prepare_voice_change, name):
            _clear_voice_reload_guard(self)
            return False
        worker = asyncio.create_task(
            run_in_daemon_thread(_run_voice_reload, self, name)
        )
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
        except TimeoutError:
            self.log(string("celune.voice_switch_timeout"), "warning")
            return False
    except BaseException:
        _clear_voice_reload_guard(self)
        raise
    finally:
        self._async_runtime_lock.release()
    return _voice_switch_succeeded(self, name)


def _run_voice_reload(self, name: str) -> None:
    """Run a voice reload and release its guard after the worker exits."""
    try:
        self.change_voice(name)
    finally:
        _clear_voice_reload_guard(self)


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
    if not _prepare_backend_reload(self, backend_spec):
        return False
    preferred_voice = self.current_voice
    threading.Thread(
        target=core_reload.reload_backend,
        args=(self, backend_spec, preferred_voice),
        daemon=True,
    ).start()
    return True


def _prepare_backend_reload(
    self,
    backend_spec: CoreBackendSpec,
) -> bool:
    """Prepare runtime state for one backend reload before loading begins."""
    with self.say_lock:
        if self._closed or self.exit_requested:
            return False
        reload_already_pending = self._reload_pending or self.cur_state == "reloading"

    if reload_already_pending:
        self.log(string("celune.reload_already_in_progress"), "warning")
        return False

    if isinstance(backend_spec, str):
        normalized_backend = backend_spec.strip().lower()
        if normalized_backend not in BACKENDS and normalized_backend not in VC_BACKENDS:
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

    with self.say_lock:
        if self._closed or self.exit_requested:
            return False
        reload_already_pending = self._reload_pending or self.cur_state == "reloading"
        if not reload_already_pending:
            self._reload_pending = True
            self._model_ready.clear()

    if reload_already_pending:
        self.log(string("celune.reload_already_in_progress"), "warning")
        return False

    self.change_input_state_callback(locked=True)
    self.change_voice_lock_state_callback(locked=True)
    self.force_stop_speech()
    try_play_signal(self, "working")
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
    if not set_backend(self, backend_spec):
        return False

    if not self._model_ready.wait(timeout=timeout):
        self.log(string("celune.backend_switch_timeout"), "warning")
        return False
    return _backend_reload_succeeded(self, backend_spec)


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
    await run_in_daemon_thread(self._async_runtime_lock.acquire)
    try:
        if not await run_in_daemon_thread(
            _prepare_backend_reload,
            self,
            backend_spec,
        ):
            return False
        preferred_voice = self.current_voice
        try:
            await asyncio.wait_for(
                run_in_daemon_thread(
                    core_reload.reload_backend,
                    self,
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
    return _backend_reload_succeeded(self, backend_spec)


def _backend_reload_succeeded(self, backend_spec: CoreBackendSpec) -> bool:
    """Return whether the requested backend is now the active loaded runtime."""
    target_name = (
        backend_spec.name
        if isinstance(backend_spec, (CeluneBackend, CeluneVCBackend))
        else getattr(backend_spec, "name", str(backend_spec))
    )
    return (
        self.loaded
        and core_lifecycle.active_runtime_backend_name(self) == str(target_name).lower()
    )


def set_cevoice(self, bundle: Optional[Union[str, Path]]) -> bool:
    """Request a hot reload into another CEVOICE bundle.

    Args:
        bundle: The CEVOICE bundle name or path to activate.

    Returns:
        bool: ``True`` when the reload worker was started.
    """
    if not _prepare_cevoice_reload(self, bundle):
        return False
    threading.Thread(
        target=core_reload.reload_cevoice,
        args=(self, bundle, None),
        daemon=True,
    ).start()
    return True


def _prepare_cevoice_reload(self, bundle: Optional[Union[str, Path]]) -> bool:
    """Prepare runtime state for one CEVOICE reload before loading begins."""
    with self.say_lock:
        if self._closed or self.exit_requested:
            return False
        reload_already_pending = self._reload_pending or self.cur_state == "reloading"

    if reload_already_pending:
        self.log(string("celune.reload_already_in_progress"), "warning")
        return False

    if bundle is not None:
        resolved_bundle = resolve_bundle_path(bundle)
        if not resolved_bundle.exists():
            self.log(string("celune.voice_pack_not_found", bundle=bundle), "warning")
            return False

    with self.say_lock:
        if self._closed or self.exit_requested:
            return False
        reload_already_pending = self._reload_pending or self.cur_state == "reloading"
        if not reload_already_pending:
            self._reload_pending = True
            self._model_ready.clear()

    if reload_already_pending:
        self.log(string("celune.reload_already_in_progress"), "warning")
        return False

    self.change_input_state_callback(locked=True)
    self.change_voice_lock_state_callback(locked=True)
    self.force_stop_speech()
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
    if not set_cevoice(self, bundle):
        return False

    if not self._model_ready.wait(timeout=timeout):
        self.log(string("celune.character_switch_timeout"), "warning")
        return False
    return _cevoice_reload_succeeded(self, bundle)


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
    await run_in_daemon_thread(self._async_runtime_lock.acquire)
    try:
        if not await run_in_daemon_thread(
            _prepare_cevoice_reload,
            self,
            bundle,
        ):
            return False
        try:
            await asyncio.wait_for(
                run_in_daemon_thread(
                    core_reload.reload_cevoice,
                    self,
                    bundle,
                    None,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            self.log(string("celune.character_switch_timeout"), "warning")
            return False
    finally:
        self._async_runtime_lock.release()
    return _cevoice_reload_succeeded(self, bundle)


def _cevoice_reload_succeeded(self, bundle: Optional[Union[str, Path]]) -> bool:
    """Return whether the requested CEVOICE bundle is now the active loaded pack."""
    return self.loaded and active_bundle_path() == resolve_bundle_path(bundle)


def speech_playback_active(self) -> bool:
    """Return whether speech generation or a speech source is active."""
    if self.locked or self.cur_state == "generating":
        return True
    if getattr(self, "_active_speech_generation", None) is not None:
        return True

    return any(
        isinstance(metadata, dict) and metadata.get("kind") == "speech"
        for metadata in self.playback_source_meta.values()
    )


def any_playback_active(self) -> bool:
    """Return whether any registered audio source is still draining."""
    return bool(getattr(self, "playback_source_meta", {}))


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
    ok = await run_in_daemon_thread(self._model_ready.wait, timeout)
    if not ok:
        self.log(string("celune.ready_wait_timeout"), "warning")
        self.log(string("celune.ready_wait_reason"), "warning")
        self.log(string("celune.ready_wait_not_fatal"), "warning")
        return False

    if not self.loaded:
        self.log(string("celune.model_unloaded_while_waiting"), "warning")
        return False

    if wait_for_speech is None:
        wait_for_speech = speech_playback_active(self)

    if wait_for_speech:
        ok = await run_in_daemon_thread(self._playback_done.wait, timeout)
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
        wait_until_ready=lambda timeout=30.0: core_reload.wait_until_idle(
            self, timeout=timeout
        ),
        backend_override=self.with_backend,
        cevoice_override=self.with_cevoice,
        name=APP_NAME,
        version=__version__,
        log_level=self.log_level,
    )
    self.extension_manager = CeluneExtensionManager(
        ctx,
        self._event_dispatcher,
        core=self,
    )
    self.extension_manager.autoload(str(project_root() / "extensions"))

    self.log(
        f"[Core] Loaded extensions: {', '.join(self.extension_manager.list_extensions())}",
        loglevel="verbose",
    )


def sleep(self, seconds: float) -> None:
    """Pause the current extension worker for a number of seconds.

    Args:
        seconds: Number of seconds to pause the current thread.
    """
    if self.exit_requested:
        return
    time.sleep(seconds)


def try_play_signal(self, signal_type: str) -> bool:
    """Play a runtime signal when the playback pipeline can accept it.

    Args:
        signal_type: The signal type to play.

    Returns:
        bool: ``True`` when the requested signal was played, otherwise ``False``.

    Raises:
        ValueError: An invalid signal was requested.
    """
    if self._closed or self.exit_requested:
        return False
    playback_thread = self.playback_thread
    if playback_thread is None or not playback_thread.is_alive():
        return False

    if self.locked and self._playback_done.is_set():
        self.release_pipeline()

    return play_signal(self, signal_type)


def voice_prompt_supported(self) -> bool:
    """Return whether the active TTS configuration supports voice prompts.

    Returns:
        bool: Whether the currently loaded TTS model supports voice prompting.
    """
    backend = self.backend
    return not (
        backend.name == "qwen3"
        and getattr(backend, "clone_model_id", None) == QWEN3_06B_MODEL_ID
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
    if self._closed or self.exit_requested:
        return

    self.log(string("celune.reloading", app_name=APP_NAME))
    self._ready_announced = False
    self.status_callback(string("status.reloading"))
    self.progress_callback(None, None)
    self.cur_state = "reloading"
    active_voice = self.current_voice or voice

    try:
        with self._model_lock:
            if self.is_voice_conversion_mode():
                self.current_voice = voice
                self.loaded = True
            else:
                new_model_name = self.backend.model_id_for_voice(voice)

                # VoxCPM2 uses the same model for all voices, so we don't have to reload every time
                if new_model_name != self.model_name:
                    if not try_play_signal(self, "working"):
                        self.log(
                            "Could not play the working signal.",
                            "warning",
                            loglevel="verbose",
                        )
                    self.log(
                        f"[RELOAD] Unloading model: {self.model_name}",
                        loglevel="verbose",
                    )
                    core_lifecycle.unload_runtime_state(
                        self,
                        include_normalizer=False,
                    )
                    self.log(
                        f"[RELOAD] Loading model: {new_model_name}",
                        loglevel="verbose",
                    )
                    self.model = self.backend.load_model(new_model_name)
                    self.model_name = new_model_name

                    self.log(string("celune.rewarming_up"))
                    from .loading import warmup

                    if not warmup(self):
                        core_reload.raise_warmup_error(
                            self,
                            "warmup failed after reload",
                        )

                    if not try_play_signal(self, "readiness"):
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

        self.log(string("celune.voice_loaded", voice=voice))
        self.progress_callback(1, 1)
        self.cur_state = "idle"
        self.status_callback(string("status.idle"))
        self._model_ready.set()
        self.voice_changed_callback(voice)
        if active_voice != voice:
            self.emit_event(
                "voice_changed",
                VoiceChangedEvent(
                    celune=self,
                    old_voice=active_voice,
                    new_voice=voice,
                ),
            )
    except Exception as error:
        self.fatal()
        self.log(
            format_error_message(
                tagged_string("celune.reload_error", "RELOAD ERROR"),
                error,
                self.log_level,
            ),
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
