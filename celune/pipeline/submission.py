# SPDX-License-Identifier: Apache-2.0
"""Speech request submission and text preparation."""

from __future__ import annotations

import os
import re
import time
import queue
import pathlib
from typing import TYPE_CHECKING, Union, Optional, cast
from collections.abc import Callable

import numpy as np
import soundfile as sf
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue

from ..vc import VC_PLAYBACK_GAIN, normalize_vc_audio
from ..cesn import contains_cesn_note_event
from ..i18n import string, tagged_string
from ..utils import (
    rng_replace,
    cesn_to_text,
    is_april_fools,
    detect_language,
    format_error_message,
    normalize_special_characters,
)
from .output import (
    is_youtube_sfx_url,
    download_youtube_sfx,
    format_stat_duration,
)
from ..cevoice import default_loader
from .captions import (
    playback_source_meta,
    next_playback_source_id,
    register_playback_source,
    set_playback_source_status,
    clear_playback_source_status,
    register_overlay_playback_state,
)
from .buffering import (
    acquire_pipeline,
    release_pipeline,
    queue_playback_done,
    queue_playback_chunk,
    acquire_pipeline_result,
    flush_buffered_speech_chunks,
)
from ..audio.dsp import split, resample_audio, is_silent_utterance
from ..constants import BASE_SR, APP_NAME
from ..exceptions import NotAvailableError
from ..typing.aliases import AudioChunk, AudioChunks
from ..typing.pipeline import SpeechStreamQueue
from ..core.concurrency import run_in_daemon_thread
from ..dataclasses.pipeline import (
    AudioOutput,
    SpeechTiming,
    SpeechRequest,
    AudioInputRequest,
    PreparedSpeechAudio,
    VoiceConversionRequest,
    CaptionAlignmentSection,
)

if TYPE_CHECKING:
    from ..celune import Celune
    from .captions import CaptionAlignmentWorker


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
    if engine.exit_requested or getattr(engine, "test_finished", False):
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
        engine,
        text,
        save=save,
        stream_queue=None,
        display_text=display_text,
    )


def _resolve_tutorial_display_sections(
    sections: tuple[str, ...],
    display_sections: Optional[tuple[str, ...]],
) -> tuple[str, ...]:
    """Return caption passages and validate their one-to-one section mapping.

    Args:
        sections: Spoken tutorial passages.
        display_sections: Optional passages shown by the caption UI.

    Returns:
        tuple[str, ...]: Caption passages aligned by section with speech.

    Raises:
        NotAvailableError: If display passages do not match the spoken passages.
    """
    visible_sections = sections if display_sections is None else display_sections
    if len(visible_sections) != len(sections) or any(
        not section.split() for section in visible_sections
    ):
        raise NotAvailableError("tutorial display sections do not match their speech")
    return visible_sections


def prepare_tutorial_sections(
    engine: Celune,
    sections: tuple[str, ...],
    is_active: Callable[[], bool],
    timeout: float,
    display_sections: Optional[tuple[str, ...]] = None,
) -> Optional[tuple[AudioChunk, ...]]:
    """Generate tutorial speech passages separately without normalization.

    Args:
        engine: Runtime that owns tutorial speech generation.
        sections: Passage text sent to the TTS backend.
        is_active: Callback that reports whether the tutorial is still active.
        timeout: Maximum time to wait for all passages to be generated.
        display_sections: Optional caption text corresponding to each passage.

    Returns:
        Optional[tuple[AudioChunk, ...]]: Prepared passage audio, or ``None`` if
        the tutorial was canceled.
    """
    visible_sections = _resolve_tutorial_display_sections(sections, display_sections)
    transcript = " ".join(sections)
    display_transcript = " ".join(visible_sections)
    result_queue: queue.Queue[Union[PreparedSpeechAudio, Exception]] = queue.Queue(
        maxsize=1
    )
    if not sections or not _queue_speech_request(
        engine,
        transcript,
        save=False,
        display_text=display_transcript,
        allow_tutorial=True,
        normalize=False,
        synthesis_sections=sections,
        audio_capture_queue=result_queue,
    ):
        raise NotAvailableError("tutorial speech preparation was rejected")

    deadline = time.monotonic() + timeout
    while is_active() and time.monotonic() < deadline:
        try:
            result = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(result, Exception):
            raise result
        if len(result.sections) != len(sections) or any(
            not audio.size for audio in result.sections
        ):
            raise NotAvailableError("tutorial speech preparation returned no audio")
        return result.sections
    if is_active():
        raise TimeoutError("tutorial speech preparation timed out")
    return None


def play_tutorial_sections(
    engine: Celune,
    sections: tuple[str, ...],
    audio_sections: tuple[AudioChunk, ...],
    is_active: Callable[[], bool],
    on_section_start: Callable[[int], None],
    timeout: float,
    display_sections: Optional[tuple[str, ...]] = None,
) -> bool:
    """Stitch tutorial passages and align visible text with spoken passages.

    Args:
        engine: Runtime that owns tutorial playback.
        sections: Passage text sent to the TTS backend.
        audio_sections: Generated audio for each spoken passage.
        is_active: Callback that reports whether the tutorial is still active.
        on_section_start: Callback invoked when playback reaches each passage.
        timeout: Maximum time to wait for playback to complete.
        display_sections: Optional caption text corresponding to each passage.

    Returns:
        bool: ``True`` when stitched tutorial playback completes.
    """
    if not sections or len(sections) != len(audio_sections):
        raise NotAvailableError("tutorial speech sections do not match their audio")
    visible_sections = _resolve_tutorial_display_sections(sections, display_sections)
    normalized_audio = tuple(
        np.asarray(audio, dtype=np.float32) for audio in audio_sections
    )
    if any(
        audio.ndim != normalized_audio[0].ndim
        or audio.shape[1:] != normalized_audio[0].shape[1:]
        for audio in normalized_audio
    ):
        raise NotAvailableError("tutorial speech sections have different audio layouts")

    pause_samples = BASE_SR // 2
    combined_parts: list[AudioChunk] = []
    section_starts: list[int] = []
    alignment_sections: list[CaptionAlignmentSection] = []
    total_frames = 0
    transcript_word_start = 0
    display_word_start = 0
    silence_shape = (pause_samples,) + normalized_audio[0].shape[1:]
    silence = np.zeros(silence_shape, dtype=np.float32)
    for index, section_audio in enumerate(normalized_audio):
        section_starts.append(total_frames)
        transcript_words = normalize_special_characters(
            sections[index],
            for_tts=True,
        ).split()
        display_words = visible_sections[index].split()
        combined_parts.append(section_audio)
        alignment_sections.append(
            CaptionAlignmentSection(
                audio_start_frame=total_frames,
                audio_end_frame=total_frames + len(section_audio),
                transcript_word_start=transcript_word_start,
                transcript_word_end=transcript_word_start + len(transcript_words),
                display_word_start=display_word_start,
                display_word_end=display_word_start + len(display_words),
            )
        )
        transcript_word_start += len(transcript_words)
        display_word_start += len(display_words)
        total_frames += len(section_audio)
        if index < len(audio_sections) - 1:
            combined_parts.append(silence)
            total_frames += pause_samples

    transcript = " ".join(sections)
    display_transcript = " ".join(visible_sections)
    combined_audio = np.concatenate(combined_parts).astype(np.float32, copy=False)
    source_queue: queue.Queue[Union[int, Exception]] = queue.Queue(maxsize=1)
    if not _queue_speech_request(
        engine,
        transcript,
        save=False,
        display_text=display_transcript,
        allow_tutorial=True,
        normalize=False,
        prepared_audio=combined_audio,
        playback_source_queue=source_queue,
        caption_alignment_sections=tuple(alignment_sections),
    ):
        raise NotAvailableError("tutorial playback was rejected")

    source_id: Optional[int] = None
    deadline = time.monotonic() + timeout
    while is_active() and time.monotonic() < deadline:
        try:
            source = source_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(source, Exception):
            raise source
        source_id = source
        break
    if source_id is None:
        if is_active():
            raise TimeoutError("tutorial playback did not start")
        return False

    next_section = 0
    deadline = time.monotonic() + timeout
    source_meta = getattr(engine, "playback_source_meta", {})
    while is_active() and time.monotonic() < deadline:
        metadata = source_meta.get(source_id, {})
        played_frames = int(float(metadata.get("played_frames", 0.0)))
        while (
            next_section < len(section_starts)
            and played_frames >= section_starts[next_section]
        ):
            on_section_start(next_section)
            next_section += 1
        if engine.playback_done.wait(timeout=0.02):
            return True
    return False


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
    if engine.exit_requested or getattr(engine, "test_finished", False):
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
    if engine.exit_requested:
        return None
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
    if engine.exit_requested:
        return None
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
            volume=VC_PLAYBACK_GAIN,
            status_label_key="pipeline.revoicing_label",
            log_length=request.log_playback,
            reset_ready_announcement=request.reset_ready_announcement,
        )

    engine.log(
        f"{string('pipeline.audio_ignored_tts_only')} "
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
    return output


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
    return _queue_speech_request(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
        allow_tutorial=False,
    )


def _queue_speech_request(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
    *,
    allow_tutorial: bool,
    normalize: Optional[bool] = None,
    synthesis_sections: tuple[str, ...] = (),
    audio_capture_queue: Optional[
        queue.Queue[Union[PreparedSpeechAudio, Exception]]
    ] = None,
    prepared_audio: Optional[AudioChunk] = None,
    playback_source_queue: Optional[queue.Queue[Union[int, Exception]]] = None,
    caption_alignment_sections: tuple[CaptionAlignmentSection, ...] = (),
) -> bool:
    """Wait for readiness and queue one speech request.

    Args:
        engine: Runtime that owns the speech pipeline.
        text: Text to synthesize.
        save: Whether generated output artifacts should be saved.
        stream_queue: Optional queue that receives generated audio chunks.
        display_text: Optional text to associate with the request in the UI.
        allow_tutorial: Whether this internal request may speak during tutorial mode.

    Returns:
        bool: ``True`` when the request was queued successfully.
    """
    if getattr(engine.backend, "singing", False) and not contains_cesn_note_event(text):
        engine.log(string("celune.singing_backend_requires_cesn"), "warning")
        return False

    if engine.exit_requested or not _prepare_speech_readiness(
        engine,
        allow_tutorial=allow_tutorial,
    ):
        return False

    if not _wait_for_model_ready(engine) or engine.exit_requested:
        return False
    if not _finish_speech_readiness(engine):
        return False

    return _queue_speech_after_ready(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
        normalize=normalize,
        synthesis_sections=synthesis_sections,
        audio_capture_queue=audio_capture_queue,
        prepared_audio=prepared_audio,
        playback_source_queue=playback_source_queue,
        caption_alignment_sections=caption_alignment_sections,
    )


def _prepare_speech_readiness(
    engine: Celune,
    *,
    allow_tutorial: bool = False,
) -> bool:
    """Run the pre-wait checks shared by synchronous and async speech queueing.

    Args:
        engine: Runtime that owns the speech pipeline.
        allow_tutorial: Whether this internal request may speak during tutorial mode.

    Returns:
        bool: ``True`` when speech may proceed.
    """
    if getattr(engine, "test_finished", False):
        return False
    if engine.is_in_tutorial and not allow_tutorial:
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
    *,
    normalize: Optional[bool] = None,
    synthesis_sections: tuple[str, ...] = (),
    audio_capture_queue: Optional[
        queue.Queue[Union[PreparedSpeechAudio, Exception]]
    ] = None,
    prepared_audio: Optional[AudioChunk] = None,
    playback_source_queue: Optional[queue.Queue[Union[int, Exception]]] = None,
    caption_alignment_sections: tuple[CaptionAlignmentSection, ...] = (),
) -> bool:
    """Queue one speech request after reload readiness is satisfied."""

    if engine.exit_requested:
        return False

    singing_backend = bool(getattr(engine.backend, "singing", False))
    speech_text = (
        text if singing_backend else normalize_special_characters(text, for_tts=True)
    )
    preserved_display_text = display_text if display_text is not None else text
    if singing_backend and contains_cesn_note_event(preserved_display_text):
        preserved_display_text = cesn_to_text(preserved_display_text)
    supported_languages = list(engine.backend.supported_languages)
    language_meta = (
        None if singing_backend else detect_language(speech_text, supported_languages)
    )
    requested_language = engine.language
    backend_name = str(getattr(engine.backend, "name", "")).strip().lower()
    if (
        not isinstance(requested_language, str)
        or not requested_language.strip()
        or requested_language.strip().lower() == "auto"
    ):
        # Celune usually handles language detection, but Qwen has its own handler
        if backend_name == "qwen3":
            requested_language = "Auto"
        elif language_meta is not None:
            requested_language = language_meta["language"]
        elif supported_languages:
            requested_language = supported_languages[0]
        else:
            requested_language = "en"

    if language_meta is not None and not language_meta["supported"]:
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

    if (
        not singing_backend
        and is_april_fools()
        and os.getenv("CELUNE_DISABLE_APRIL_FOOLS")
        not in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }
    ):
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
        request_rejected = False
        with engine.queue_lock:
            if engine.exit_requested:
                request_rejected = True
            else:
                engine._speech_generation = getattr(engine, "_speech_generation", 0) + 1
                engine.utterance_force_stop.clear()
                engine.text_queue.put(
                    SpeechRequest(
                        speech_text,
                        display_text=preserved_display_text,
                        language=requested_language,
                        save=save,
                        stream_queue=stream_queue,
                        normalize=(
                            engine.use_normalization if normalize is None else normalize
                        ),
                        generation=engine.speech_generation,
                        synthesis_sections=tuple(
                            section
                            if singing_backend
                            else normalize_special_characters(section, for_tts=True)
                            for section in synthesis_sections
                        ),
                        audio_capture_queue=audio_capture_queue,
                        prepared_audio=prepared_audio,
                        playback_source_queue=playback_source_queue,
                        caption_alignment_sections=caption_alignment_sections,
                    )
                )
                engine.log(
                    "[QUEUE] speech "
                    f"generation={engine.speech_generation} text_chars={len(speech_text)} "
                    f"language={requested_language} stream_queue={stream_queue is not None} "
                    f"text_queue={engine.text_queue.qsize()}",
                    loglevel="debug",
                )
        if request_rejected:
            release_pipeline(engine)
            return False
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
    if engine.exit_requested or not _prepare_speech_readiness(engine):
        return False

    if (
        not await run_in_daemon_thread(lambda: _wait_for_model_ready(engine))
        or engine.exit_requested
    ):
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
    if (
        engine.test_finished
        or engine.backend_mode == "agent_test"
        or engine.exit_requested
    ):
        return False

    try:
        audio = prepare_playback_audio(audio, sample_rate)
        if engine.exit_requested:
            return False
        playback_sample_rate = BASE_SR
        audio_len = len(audio) / playback_sample_rate
        if log_length:
            engine.log(f"{playback_sample_rate} Hz, {format_stat_duration(audio_len)}")

        if keep:
            engine.kept_sfx_audio = [chunk.copy() for chunk in split(audio, BASE_SR, 1)]

        source_id = next_playback_source_id(engine)
        register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        register_playback_source(engine, source_id, kind="sfx", base_gain=volume)
        engine.cur_state = "speaking"
        playback_generation = getattr(engine, "_playback_generation", 0)
        set_playback_source_status(
            engine,
            source_id,
            string(status_label_key, label=label),
        )
        # push the smallest possible chunks for responsive stopping
        for chunk in split(audio, playback_sample_rate, 1):
            if not queue_playback_chunk(
                engine,
                source_id,
                chunk,
                playback_sample_rate,
                generation=playback_generation,
            ):
                clear_playback_source_status(engine, source_id)
                return False
        if engine.exit_requested:
            clear_playback_source_status(engine, source_id)
            return False
        if not queue_playback_done(
            engine,
            source_id,
            generation=playback_generation,
        ):
            clear_playback_source_status(engine, source_id)
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
    low_latency: bool = False,
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
        low_latency: Whether playback should skip contention reserve buffering.
        reset_ready_announcement: Whether a newly created source should reset readiness.

    Returns:
        Optional[int]: The persistent playback source id, or ``None`` when the
            producer belongs to a canceled playback generation.
    """
    audio = prepare_playback_audio(audio, sample_rate)
    playback_sample_rate = BASE_SR
    audio_len = len(audio) / playback_sample_rate
    if log_length:
        engine.log(f"{playback_sample_rate} Hz, {format_stat_duration(audio_len)}")

    active_generation = getattr(engine, "_playback_generation", 0)
    if generation is not None and generation != active_generation:
        return None
    source_generation = active_generation if generation is None else generation

    meta = playback_source_meta(engine)
    if source_id is None or source_id not in meta:
        source_id = next_playback_source_id(engine)
        register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        register_playback_source(
            engine,
            source_id,
            kind="sfx",
            base_gain=volume,
            low_latency=low_latency,
        )
        engine.cur_state = "speaking"
    elif float(meta[source_id].get("generation", 0.0)) != float(active_generation):
        return None

    set_playback_source_status(
        engine,
        source_id,
        string(status_label_key, label=label),
    )

    if len(audio) > 0 and not queue_playback_chunk(
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
    source_meta = playback_source_meta(engine).get(source_id)
    if not isinstance(source_meta, dict):
        return
    queue_playback_done(
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
    if (
        engine.test_finished
        or engine.backend_mode == "agent_test"
        or engine.exit_requested
    ):
        return False

    downloaded_from_url = is_youtube_sfx_url(sound_path)
    if downloaded_from_url:
        downloaded_info = download_youtube_sfx(engine, sound_path)
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


def finish_tutorial_audio_capture(
    engine: Celune,
    audio_capture_queue: queue.Queue[Union[PreparedSpeechAudio, Exception]],
    speech_len: float,
    full_audio: AudioChunks,
    prepared_sections: list[AudioChunk],
    silent_retry_count: int,
    max_silent_retries: int,
) -> bool:
    """Finish captured tutorial generation and report whether it should retry."""
    engine.reverb.reset()
    full_audio_array = np.concatenate(full_audio) if full_audio else None
    is_silent = False
    silence_tier = 0
    if full_audio_array is not None:
        is_silent, silence_tier = is_silent_utterance(full_audio_array)
    if is_silent and silence_tier == 2 and silent_retry_count < max_silent_retries:
        engine.regenerate = True
        engine.log(
            string(
                "pipeline.silent_regenerating",
                retry_count=silent_retry_count + 1,
                max_retries=max_silent_retries,
            ),
            "warning",
        )
        return True
    if is_silent and silence_tier == 2:
        engine.log(
            string(
                "pipeline.silent_regeneration_limit_reached",
                max_retries=max_silent_retries,
            ),
            "warning",
        )
    if is_silent and silence_tier == 1:
        engine.log(string("pipeline.may_be_silent"), "warning")
    engine.total_generated_speech_seconds += speech_len
    audio_capture_queue.put(PreparedSpeechAudio(tuple(prepared_sections)))
    release_pipeline(engine)
    return False


def play_prepared_speech(
    engine: Celune,
    item: SpeechRequest,
    caption_worker: Optional[CaptionAlignmentWorker],
) -> None:
    """Queue prepared speech through the shared playback and caption path."""
    source_id: Optional[int] = None
    try:
        audio = item.prepared_audio
        if audio is None or not audio.size:
            raise NotAvailableError("prepared speech has no audio")
        audio = np.asarray(audio, dtype=np.float32)
        source_id = next_playback_source_id(engine)
        caption_enabled = (
            engine.config.get("captions") is True and caption_worker is not None
        )
        words = tuple(item.display_text.split())
        register_playback_source(
            engine,
            source_id,
            kind="speech",
            async_caption_audio=caption_enabled,
            caption_word_total=len(words) if caption_enabled else 0,
        )
        engine.log(f"[GEN] {item.display_text}")
        speech_timing = SpeechTiming(start_time=time.monotonic())
        if caption_enabled and caption_worker is not None:
            language_resolver = getattr(
                engine.backend,
                "resolve_generation_language",
                None,
            )
            language = (
                language_resolver(item.language)
                if callable(language_resolver)
                else item.language
            )
            caption_worker.submit_chunk(
                engine,
                source_id,
                [audio],
                speech_timing,
                item.stream_queue,
                item.display_text,
                item.text,
                words,
                language if isinstance(language, str) else None,
                0,
                len(words),
                tuple(item.text.split()),
                [False],
                [False],
                BASE_SR,
                item.caption_alignment_sections,
            )
            caption_worker.submit_done(
                engine,
                source_id,
                item.stream_queue,
                release_pipeline_when_finished=False,
                analysis_audio=audio.copy(),
            )
            release_pipeline(engine, playback_idle=False)
        else:
            flush_buffered_speech_chunks(
                engine,
                source_id,
                [audio],
                speech_timing,
                False,
                item.stream_queue,
            )
            queue_playback_done(
                engine,
                source_id,
                release_pipeline_when_finished=True,
                analysis_audio=audio.copy(),
            )
        if item.playback_source_queue is not None:
            item.playback_source_queue.put(source_id)
    except Exception as error:
        if source_id is not None and caption_worker is not None:
            caption_worker.cancel_source(source_id)
        if item.playback_source_queue is not None:
            item.playback_source_queue.put(error)
        engine.log(
            format_error_message(
                tagged_string("pipeline.gen_error", "GEN ERROR"),
                error,
                engine.log_level,
            ),
            "error",
        )
        engine.cur_state = "error"
        release_pipeline(engine)
        engine.progress_callback(0, 1)
        engine.error_callback(string("pipeline.could_not_generate", app_name=APP_NAME))


def split_text(engine: Celune, text: str) -> list[str]:
    """Adaptively split text into chunks. Short text is unaffected, while long text is chunked effectively.

    Args:
        engine: The Celune engine to report output back to.
        text: The input text to split.

    Returns:
        list[str]: The generated text chunks.
    """
    text = text.strip()
    if not text:
        return []

    chunk_length = 150
    max_length = 400

    # detect sentences
    unit_checker = re.compile(r"\S.*?(?:[.!?]+[\"')\]]*(?=\s+|$)|$)", re.DOTALL)

    # detected quoted text with a boundary
    quote_checker = re.compile(r'"[^"]*[.!?]"')

    if len(text) <= max_length and not quote_checker.search(text):
        # input is short, return as is
        return [text]

    def split_long_unit(value: str) -> list[str]:
        pieces = [piece.strip() for piece in value.splitlines() if piece.strip()]
        if not pieces:
            pieces = value.split()

        unit_chunks = []
        unit_current = ""

        for piece in pieces:
            if len(piece) > max_length:
                if unit_current:
                    unit_chunks.append(unit_current)
                    unit_current = ""
                unit_chunks.extend(split_words(piece))
                continue

            if (unit_current and len(unit_current) + 1 + len(piece) > max_length) or (
                unit_current and len(unit_current) >= chunk_length
            ):
                unit_chunks.append(unit_current)
                unit_current = piece
            elif unit_current:
                unit_current = f"{unit_current} {piece}"
            else:
                unit_current = piece

        if unit_current:
            unit_chunks.append(unit_current)

        return unit_chunks

    def split_words(value: str) -> list[str]:
        word_chunks = []
        word_current = ""

        for word in value.split():
            if word_current and len(word_current) + 1 + len(word) > max_length:
                word_chunks.append(word_current)
                word_current = word
            elif word_current:
                word_current = f"{word_current} {word}"
            else:
                word_current = word

        if word_current:
            word_chunks.append(word_current)

        return word_chunks

    def split_sentences(value: str) -> list[str]:
        units = []
        for rmatch in unit_checker.finditer(value):
            unit = rmatch.group(0).strip()
            if len(unit) > max_length:
                units.extend(split_long_unit(unit))
            elif unit:
                units.append(unit)
        return units

    def split_units(value: str) -> list[str]:
        units = []
        start = 0

        for qmatch in quote_checker.finditer(value):
            units.extend(split_sentences(value[start : qmatch.start()]))
            units.append(qmatch.group(0).strip())
            start = qmatch.end()

        units.extend(split_sentences(value[start:]))
        return [unit for unit in units if unit]

    all_units = split_units(text)
    if not all_units:
        return []

    chunks = []
    current = ""

    for u in all_units:
        if quote_checker.fullmatch(u):
            if current:
                chunks.append(current)
                current = ""
            chunks.append(u)
            continue

        if (current and len(current) + 1 + len(u) > max_length) or (
            current and len(current) >= chunk_length
        ):
            chunks.append(current)
            current = u
        elif current:
            current = f"{current} {u}"
        else:
            current = u

    if current:
        chunks.append(current)

    engine.log(f"Chunks: {len(chunks)}")
    return chunks
