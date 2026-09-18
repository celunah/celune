# SPDX-License-Identifier: Apache-2.0
"""Celune core speech helpers."""

from __future__ import annotations

import os
import pathlib
import queue
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import soundfile as sf
from iso639 import Lang
from iso639.exceptions import DeprecatedLanguageValue, InvalidLanguageValue

from .audio.dsp import pitch_shift_audio, resample_audio, split
from .cevoice import default_loader
from .constants import APP_NAME, BASE_SR
from .dataclasses.pipeline import (
    AudioInputRequest,
    AudioOutput,
    SpeechRequest,
    VoiceConversionRequest,
)
from .i18n import string
from .persona.impl import compact_persona_history
from .conversation import (
    _classify_persona_memories,
    _store_persona_memories,
)
from .pipeline import (
    _format_stat_duration,
    _invalidate_speech_work,
    _run_in_daemon_thread,
    close_stream,
)
from .playback import (
    acquire_pipeline,
    acquire_pipeline_result,
    _clear_playback_source_status,
    _download_youtube_sfx,
    _is_youtube_sfx_url,
    _next_playback_source_id,
    _playback_source_meta,
    _queue_playback_chunk,
    _queue_playback_done,
    _register_overlay_playback_state,
    _register_playback_source,
    _set_playback_source_status,
    release_pipeline,
)
from .typing.aliases import AudioChunk
from .typing.pipeline import SpeechStreamQueue
from .utils import (
    detect_language,
    format_error_message,
    is_april_fools,
    normalize_special_characters,
    rng_replace,
)
from .vc import normalize_vc_audio
from .binding import install_class_functions, install_module_functions

if TYPE_CHECKING:
    from .celune import Celune

__all__ = (
    "_finish_speech_readiness",
    "_prepare_speech_readiness",
    "_queue_speech_after_ready",
    "close",
    "convert_audio_input",
    "deliver_persona_response",
    "finish_streaming_sfx_audio",
    "handle_audio_input",
    "play",
    "prepare_playback_audio",
    "queue_sfx_audio",
    "queue_speech",
    "queue_speech_async",
    "queue_streaming_sfx_audio",
    "say",
    "say_async",
    "say_stream",
    "say_stream_async",
    "stop_live_audio_input",
    "submit_audio",
)


def deliver_persona_response(engine: Celune, request: str, response: str) -> bool:
    """Store and speak one response produced by a routed Persona task."""
    spoken_text = response.strip()
    if not spoken_text:
        return False

    _store_persona_memories(engine, request)
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
    _classify_persona_memories(engine, request)
    return queued


def say(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say.

    Args:
        engine: The Celune engine that should speak the text.
        text: The input text to queue for synthesis.
        save: Whether to save generated output artifacts.
        display_text: Optional text to show in logs instead of the synthesis text.

    Returns:
        bool: ``True`` when the text was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if queueing fails.
    """
    if getattr(engine, "test_finished", False):
        return False
    engine.log(
        f"[ENGINE] say requested text_chars={len(text)} save={save} "
        f"state={engine.cur_state} mode={engine.input_mode}",
        loglevel="debug",
    )
    if engine.input_mode != "text_to_speech":
        engine.log(string("celune.text_input_unavailable_vc"), "warning")
        engine.error_callback(string("celune.not_possible"))
        engine.progress_callback(0, 1)
        return False

    return queue_speech(
        engine, text, save=save, stream_queue=None, display_text=display_text
    )


async def say_async(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say without blocking an async caller.

    Args:
        engine: Runtime that owns the speech queues.
        text: The text to synthesize.
        save: Whether the generated utterance should be persisted to disk.
        display_text: Optional UI-facing text to associate with the request.

    Returns:
        bool: ``True`` when the request was queued successfully, otherwise ``False``.
    """
    if getattr(engine, "test_finished", False):
        return False
    if engine.input_mode != "text_to_speech":
        engine.log(string("celune.text_input_unavailable_vc"), "warning")
        engine.error_callback(string("celune.not_possible"))
        engine.progress_callback(0, 1)
        return False

    return await queue_speech_async(
        engine,
        text,
        save=save,
        stream_queue=None,
        display_text=display_text,
    )


def say_stream(
    engine: Celune,
    text: str,
    save: bool = True,
) -> Optional[SpeechStreamQueue]:
    """Queue text for playback and mirror generated chunks to a queue."""
    stream_queue: SpeechStreamQueue = queue.Queue(maxsize=2)
    if not queue_speech(engine, text, save=save, stream_queue=stream_queue):
        return None
    return stream_queue


async def say_stream_async(
    engine: Celune,
    text: str,
    save: bool = True,
) -> Optional[SpeechStreamQueue]:
    """Queue text for playback and mirror chunks without blocking an async caller."""
    stream_queue: SpeechStreamQueue = queue.Queue(maxsize=2)
    if not await queue_speech_async(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
    ):
        return None
    return stream_queue


def submit_audio(
    engine: Celune,
    audio: np.ndarray,
    sample_rate: int,
    label: str = "audio input",
    pitch_shift: Optional[int] = None,
    f0_condition: Optional[bool] = None,
    log_playback: bool = True,
    reset_ready_announcement: bool = True,
) -> bool:
    """Accept audio input for the active engine mode."""
    if getattr(engine, "test_finished", False) or engine.backend_mode == "agent_test":
        return False
    return handle_audio_input(
        engine,
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


def handle_audio_input(engine: Celune, request: AudioInputRequest) -> bool:
    """Accept engine-level audio input and route it according to the active mode.

    Args:
        engine: The Celune engine receiving the audio input.
        request: The submitted audio input request.

    Returns:
        bool: ``True`` when the request was accepted, otherwise ``False``.
    """
    audio = np.asarray(request.audio, dtype=np.float32)
    if getattr(engine, "input_mode", "text_to_speech") == "voice_conversion":
        output = convert_audio_input(engine, request)
        if output is None:
            return False
        return queue_sfx_audio(
            engine,
            output.audio,
            output.sample_rate,
            output.label,
            status_label_key="pipeline.revoicing_label",
            log_length=request.log_playback,
            reset_ready_announcement=request.reset_ready_announcement,
        )

    engine.log(
        "Audio input was accepted but ignored in text-to-speech-only mode "
        f"(label={request.label!r}, sample_rate={request.sample_rate}, "
        f"shape={audio.shape!r})",
        loglevel="verbose",
    )
    return True


def convert_audio_input(
    engine: Celune,
    request: AudioInputRequest,
    *,
    live: bool = False,
) -> Optional[AudioOutput]:
    """Run one VC conversion request and return the converted audio output.

    Args:
        engine: The Celune engine receiving the audio input.
        request: The submitted audio input request.

    Returns:
        Optional[AudioOutput]: The converted audio output, or ``None`` when voice conversion is unavailable.
    """
    backend = getattr(engine, "vc_backend", None)
    if backend is None:
        engine.log(string("pipeline.vc_backend_unconfigured"), "warning")
        engine.error_callback(string("pipeline.vc_backend_unconfigured"))
        engine.progress_callback(0, 1)
        return None

    target_references: tuple[pathlib.Path, ...] = ()
    current_voice = getattr(engine, "current_voice", None)
    if isinstance(current_voice, str) and current_voice.strip():
        loader = default_loader()
        if loader is not None:
            try:
                target_references = (loader.materialize(current_voice, "wav"),)
            except Exception as e:
                engine.log(
                    format_error_message(
                        string("pipeline.vc_reference_load_failed"),
                        e,
                        getattr(engine, "log_level", "info"),
                    ),
                    "warning",
                )
                target_references = ()

    conversion_request = VoiceConversionRequest(
        source_audio=normalize_vc_audio(request.audio),
        sample_rate=request.sample_rate,
        target_voice=getattr(engine, "current_voice", None),
        target_character=getattr(engine, "current_character", None),
        target_references=target_references,
        label=request.label,
        pitch_shift=0,
        f0_condition=(
            request.f0_condition
            if isinstance(request.f0_condition, bool)
            else getattr(engine, "vc_f0_condition", False)
        ),
    )
    converter = getattr(backend, "convert_live", None) if live else None
    typed_converter = cast(
        Callable[[VoiceConversionRequest], AudioOutput],
        converter,
    )
    output = (
        typed_converter(conversion_request)
        if callable(converter)
        else backend.convert(conversion_request)
    )
    resolved_pitch_shift = (
        request.pitch_shift
        if isinstance(request.pitch_shift, int)
        else getattr(engine, "vc_pitch_shift", 0)
    )
    if resolved_pitch_shift == 0:
        return output

    return AudioOutput(
        audio=pitch_shift_audio(
            np.asarray(output.audio, dtype=np.float32),
            output.sample_rate,
            resolved_pitch_shift,
        ),
        sample_rate=output.sample_rate,
        label=output.label,
    )


def stop_live_audio_input(engine: Celune) -> None:
    """Reset a backend's live voice-conversion session when capture stops."""
    backend = getattr(engine, "vc_backend", None)
    stop_live = getattr(backend, "stop_live", None)
    if callable(stop_live):
        stop_live()


def queue_speech(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say and optionally mirror audio chunks.

    Args:
        engine: The Celune engine that should speak the text.
        text: The input text to queue for synthesis.
        save: Whether to save generated output artifacts.
        stream_queue: Optional queue receiving generated 48 kHz float32 chunks.
        display_text: Optional text to show in logs instead of the synthesis text.

    Returns:
        bool: ``True`` when the text was queued successfully, otherwise ``False``.

    Raises:
        Exception: An exception was caught and subsequently raised to propagate it to Celune.
    """
    if not _prepare_speech_readiness(engine):
        return False

    if not _wait_for_model_ready(engine):
        return False
    if not _finish_speech_readiness(engine):
        return False

    return _queue_speech_after_ready(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
    )


def _prepare_speech_readiness(engine: Celune) -> bool:
    """Run the pre-wait checks shared by synchronous and async speech queueing."""
    if getattr(engine, "test_finished", False):
        return False
    if engine.is_in_tutorial:
        engine.log(string("celune.speech_input_disabled_tutorial"), "warning")
        return False

    if getattr(engine, "sleeping", False):
        engine.log(
            string("pipeline.cannot_speak_sleeping", app_name=APP_NAME),
            "warning",
        )
        engine.error_callback(string("celune.app_sleeping", app_name=APP_NAME))
        engine.progress_callback(0, 1)
        return False

    if _speech_reload_blocks(engine):
        return False

    if not engine.model_ready.is_set():
        engine.status_callback(string("status.waiting_for_model"))
        engine.progress_callback(None, None)
        engine.log(string("pipeline.speak_waiting_reload"), "info")

    return True


def _speech_reload_blocks(engine: Celune) -> bool:
    """Reject speech when model lifecycle work is still in progress."""
    with engine.say_lock:
        reload_pending = (
            bool(getattr(engine, "_reload_pending", False))
            or getattr(engine, "cur_state", None) == "reloading"
        )
    if not reload_pending:
        return False

    acquisition = acquire_pipeline_result(engine, "speak")
    if acquisition.acquired:
        release_pipeline(engine)
        return False
    engine.progress_callback(0, 1)
    return True


def _wait_for_model_ready(engine: Celune) -> bool:
    """Wait for readiness while observing a reload boundary."""
    while not engine.model_ready.wait(timeout=0.1):
        if _speech_reload_blocks(engine):
            return False
        if engine.exit_requested:
            return False
    return True


def _finish_speech_readiness(engine: Celune) -> bool:
    """Run the post-wait model checks shared by speech queueing paths."""
    if not engine.loaded and not getattr(engine.backend, "is_fake", False):
        engine.log(string("ui.core_engine_not_loaded"), "warning")
        engine.error_callback(string("pipeline.not_ready_app", app_name=APP_NAME))
        engine.progress_callback(0, 1)
        return False

    return True


def _queue_speech_after_ready(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue one speech request after reload readiness is satisfied."""

    speech_text = normalize_special_characters(text, for_tts=True)
    preserved_display_text = display_text if display_text is not None else text
    language_meta = detect_language(
        speech_text,
        list(engine.backend.supported_languages),
    )
    requested_language = engine.language
    backend_name = str(getattr(engine.backend, "name", "")).strip().lower()
    if (
        not isinstance(requested_language, str)
        or not requested_language.strip()
        or requested_language.strip().lower() == "auto"
    ):
        # Celune usually handles language detection, but Qwen has its own handler
        requested_language = (
            "Auto" if backend_name == "qwen3" else language_meta["language"]
        )

    if not language_meta["supported"]:
        # "zh-cn" has to be clipped to just "zh" to be a valid language code
        try:
            language = Lang(language_meta["language"][:2]).name
        except (InvalidLanguageValue, DeprecatedLanguageValue):
            language = language_meta["language"]

        engine.log(
            string("pipeline.received_unsupported_language", language=language),
            "warning",
        )
        engine.log(
            string("pipeline.may_not_say_properly", app_name=APP_NAME), "warning"
        )

    if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
        "1",
        "true",
        "on",
        "yes",
        "enabled",
    }:
        engine.log(string("pipeline.april_fools"))
        speech_text = rng_replace(
            speech_text,
            targets=["celune"],
            replacements=["celine"],
        )

    if not acquire_pipeline(engine, "speak"):
        engine.progress_callback(0, 1)
        return False

    try:
        if not engine.loaded and not engine.backend.is_fake:
            engine.log(string("ui.core_engine_not_loaded"), "warning")
            engine.error_callback(string("pipeline.not_ready_app", app_name=APP_NAME))
            release_pipeline(engine)
            engine.progress_callback(0, 1)
            return False

        engine.cur_state = "generating"
        with engine.queue_lock:
            engine._speech_generation = getattr(engine, "_speech_generation", 0) + 1
            engine.utterance_force_stop.clear()
            engine.text_queue.put(
                SpeechRequest(
                    speech_text,
                    display_text=preserved_display_text,
                    language=requested_language,
                    save=save,
                    stream_queue=stream_queue,
                    normalize=engine.use_normalization,
                    generation=engine.speech_generation,
                )
            )
            engine.log(
                "[QUEUE] speech "
                f"generation={engine.speech_generation} text_chars={len(speech_text)} "
                f"language={requested_language} stream_queue={stream_queue is not None} "
                f"text_queue={engine.text_queue.qsize()}",
                loglevel="debug",
            )
        engine.status_callback(string("status.generating"))
        engine.progress_callback(None, None)
        return True
    except Exception:
        release_pipeline(engine)
        raise


async def queue_speech_async(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say without blocking the caller's event loop.

    Args:
        engine: Runtime that owns the speech queues.
        text: The text to synthesize.
        save: Whether the generated utterance should be persisted to disk.
        stream_queue: Optional queue receiving generated playback chunks.
        display_text: Optional UI-facing text to associate with the request.

    Returns:
        bool: ``True`` when the request was queued successfully, otherwise ``False``.
    """
    if not _prepare_speech_readiness(engine):
        return False

    if not await _run_in_daemon_thread(lambda: _wait_for_model_ready(engine)):
        return False

    if not _finish_speech_readiness(engine):
        return False

    return _queue_speech_after_ready(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
    )


def queue_sfx_audio(
    engine: Celune,
    audio: AudioChunk,
    sample_rate: int,
    label: str,
    keep: bool = False,
    volume: float = 1.0,
    status_label_key: str = "pipeline.playing_label",
    log_length: bool = True,
    reset_ready_announcement: bool = True,
) -> bool:
    """Queue decoded SFX audio through Celune's playback pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.
        label: Human-readable label for logs and status.
        keep: Whether to prepend this SFX to the next saved utterance.
        volume: Gain multiplier applied before the clip is queued for playback.
        status_label_key: Localization key used for the surfaced playback status.
        log_length: Whether to log the prepared playback sample rate and length.
        reset_ready_announcement: Whether this source should trigger a later ready announcement.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup fails.
    """
    try:
        audio = prepare_playback_audio(audio, sample_rate)
        playback_sample_rate = BASE_SR
        audio_len = len(audio) / playback_sample_rate
        if log_length:
            engine.log(f"{playback_sample_rate} Hz, {_format_stat_duration(audio_len)}")

        if keep:
            engine.kept_sfx_audio = [chunk.copy() for chunk in split(audio, BASE_SR, 1)]

        source_id = _next_playback_source_id(engine)
        _register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        _register_playback_source(engine, source_id, kind="sfx", base_gain=volume)
        engine.cur_state = "speaking"
        playback_generation = getattr(engine, "_playback_generation", 0)
        _set_playback_source_status(
            engine,
            source_id,
            string(status_label_key, label=label),
        )
        # push the smallest possible chunks for responsive stopping
        for chunk in split(audio, playback_sample_rate, 1):
            if not _queue_playback_chunk(
                engine,
                source_id,
                chunk,
                playback_sample_rate,
                generation=playback_generation,
            ):
                _clear_playback_source_status(engine, source_id)
                return False
        if not _queue_playback_done(
            engine,
            source_id,
            generation=playback_generation,
        ):
            _clear_playback_source_status(engine, source_id)
            return False
        return True
    except Exception:
        engine.playback_done.set()
        raise


def queue_streaming_sfx_audio(
    engine: Celune,
    audio: AudioChunk,
    sample_rate: int,
    label: str,
    *,
    source_id: Optional[int] = None,
    generation: Optional[int] = None,
    volume: float = 1.0,
    status_label_key: str = "pipeline.playing_label",
    log_length: bool = False,
    reset_ready_announcement: bool = False,
) -> Optional[int]:
    """Queue one SFX segment onto a persistent playback source.

    Args:
        engine: The Celune engine that should play the sound.
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.
        label: Human-readable label for logs and status.
        source_id: Existing playback source to append to, or ``None`` to create one.
        generation: Playback generation captured by the producer session.
        volume: Gain multiplier applied before the clip is queued for playback.
        status_label_key: Localization key used for the surfaced playback status.
        log_length: Whether to log the prepared playback sample rate and length.
        reset_ready_announcement: Whether a newly created source should reset readiness.

    Returns:
        Optional[int]: The persistent playback source id, or ``None`` when the
            producer belongs to a cancelled playback generation.
    """
    audio = prepare_playback_audio(audio, sample_rate)
    playback_sample_rate = BASE_SR
    audio_len = len(audio) / playback_sample_rate
    if log_length:
        engine.log(f"{playback_sample_rate} Hz, {_format_stat_duration(audio_len)}")

    active_generation = getattr(engine, "_playback_generation", 0)
    if generation is not None and generation != active_generation:
        return None
    source_generation = active_generation if generation is None else generation

    meta = _playback_source_meta(engine)
    if source_id is None or source_id not in meta:
        source_id = _next_playback_source_id(engine)
        _register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        _register_playback_source(engine, source_id, kind="sfx", base_gain=volume)
        engine.cur_state = "speaking"
    elif float(meta[source_id].get("generation", 0.0)) != float(active_generation):
        return None

    _set_playback_source_status(
        engine,
        source_id,
        string(status_label_key, label=label),
    )

    if len(audio) > 0 and not _queue_playback_chunk(
        engine,
        source_id,
        audio,
        playback_sample_rate,
        generation=source_generation,
    ):
        return None

    return source_id


def finish_streaming_sfx_audio(engine: Celune, source_id: Optional[int]) -> None:
    """Mark one persistent SFX playback source as complete.

    Args:
        engine: Runtime that owns the playback queues.
        source_id: Persistent playback source to finish.
    """
    if source_id is None:
        return
    source_meta = _playback_source_meta(engine).get(source_id)
    if not isinstance(source_meta, dict):
        return
    _queue_playback_done(
        engine,
        source_id,
        generation=int(float(source_meta.get("generation", 0.0))),
    )


def prepare_playback_audio(
    audio: AudioChunk,
    sample_rate: int,
) -> AudioChunk:
    """Normalize audio to Celune's shared playback format.

    Args:
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.

    Returns:
        AudioChunk: Audio resampled into Celune's playback format.
    """
    return resample_audio(np.asarray(audio, dtype=np.float32), sample_rate)


def play(
    engine: Celune,
    sound_path: str,
    keep: bool = False,
    volume: float = 1.0,
    on_started: Optional[Callable[[], None]] = None,
) -> bool:
    """Play a sound via Celune's pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        sound_path: The path to the audio file to play.
        keep: Whether to prepend this SFX to the next saved utterance.
        volume: How loud should the SFX be played at, limited to half of max volume to protect headphone users.
        on_started: Optional callback invoked after the audio is decoded and immediately before it is queued.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup fails.
    """
    downloaded_from_url = _is_youtube_sfx_url(sound_path)
    if downloaded_from_url:
        downloaded_info = _download_youtube_sfx(engine, sound_path)
        if downloaded_info is None:
            return False
        downloaded, playback_label = downloaded_info
        sound_path = str(downloaded)
    else:
        playback_label = sound_path

    if not os.path.exists(sound_path):
        engine.log(
            string("pipeline.cannot_find_sound", app_name=APP_NAME, path=sound_path),
            "warning",
        )
        return False

    supported_formats = ("wav", "flac", "ogg", "mp3", "aiff")

    if not any(sound_path.endswith(audio_format) for audio_format in supported_formats):
        engine.log(
            string("pipeline.sfx_format_unsupported", app_name=APP_NAME),
            "warning",
        )
        engine.log(
            string(
                "pipeline.supported_formats",
                formats=", ".join(supported_formats),
            ),
            "warning",
        )
        return False

    audio, sr = sf.read(sound_path, dtype="float32")

    if on_started is not None:
        on_started()

    queued = queue_sfx_audio(
        engine,
        np.asarray(audio, dtype=np.float32),
        sr,
        playback_label,
        keep,
        volume=volume * 0.5,
    )
    if queued and downloaded_from_url:
        engine.status_callback(
            string("pipeline.playing_label", label=playback_label),
        )
    return queued


def close(engine: Celune) -> None:
    """Shut off Celune and exit.

    Args:
        engine: The Celune engine to shut down.
    """
    if not getattr(engine, "test_finished", False):
        engine.log(string("pipeline.exiting"))
    engine._exit_requested = True
    _invalidate_speech_work(engine)
    close_stream(engine, abort=True)

    engine.text_queue.put(engine.sentinel)
    engine.audio_queue.put(engine.sentinel)

    if engine.generation_thread is not None:
        engine.generation_thread.join(timeout=0.5)

    if engine.playback_thread is not None:
        engine.playback_thread.join(timeout=0.5)

    manager = getattr(engine, "component_locks", None)
    try:
        close_stream(engine, abort=True)
        engine.glow.leave()
        engine.glow.finished.wait(timeout=5)
    finally:
        if manager is not None:
            manager.release_all()


def install(target):
    """Install extracted definitions in the original module."""
    install_module_functions(target, {name: globals()[name] for name in __all__})


def install_engine(target):
    """Install general speech entrypoints on ``Celune``."""
    install_class_functions(
        target,
        {
            name: globals()[name]
            for name in (
                "say",
                "say_async",
                "say_stream",
                "say_stream_async",
                "submit_audio",
            )
        },
    )
