# SPDX-License-Identifier: Apache-2.0
"""Speech generation workers and request processing."""

from __future__ import annotations
import queue
import pathlib
import datetime
import collections.abc
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, cast

import torch
import numpy as np
import pyrubberband as rb

from .timing import (
    monotonic_time,
)

from ..i18n import string, tagged_string
from ..paths import (
    outputs_dir,
)
from ..threads import run_in_daemon_thread
from ..typing.common import JSONSerializable
from ..typing.aliases import AudioChunk, AudioChunks
from ..constants import (
    BASE_SR,
    APP_NAME,
)
from ..persona.conversation import effective_voice_prompt
from ..vram import is_cuda_out_of_memory, release_cuda_after_oom
from ..exceptions import CESNError, BackendError, NotAvailableError
from ..utils import (
    format_number,
    format_error_message,
)
from ..dataclasses.pipeline import (
    SpeechTiming,
    SpeechRequest,
)
from ..audio.dsp import (
    soften,
    to_48khz,
    is_silent_utterance,
)
from ..metadata import (
    write_celune_flac,
    celune_metadata_payload,
)
from ..singing import format_cesn_error, prepare_synthesis_chunks
from .submission import (
    split_text,
    play_prepared_speech,
    finish_tutorial_audio_capture,
)
from .output import (
    format_stat_duration,
    _output_filename_prefix,
    _output_filename_fragment,
)
from .buffering import (
    release_pipeline,
    queue_playback_done,
    queue_playback_chunk,
    remember_smart_buffer_speed,
    smart_buffer_target_seconds,
    flush_buffered_speech_chunks,
)
from .captions import (
    playback_source_meta,
    caption_tracking_modes,
    next_playback_source_id,
    playback_caption_states,
    register_playback_source,
    caption_chunk_start_frame,
    caption_chunk_word_ranges,
    record_caption_playback_segment,
    build_caption_alignment_sections,
    record_caption_playback_audio_chunk,
)

if TYPE_CHECKING:
    from .captions import CaptionAlignmentWorker

    from ..celune import Celune

_SHORT_INPUT_VOCODER_ERROR = (
    "Calculated padded input size per channel: (6). Kernel size: (7)."
)

_EMPTY_INPUT_VOCODER_ERROR = (
    "min(): Expected reduction dim to be specified for input.numel() == 0"
)


def _is_short_input_generation_error(error: BaseException) -> bool:
    """Return whether a backend rejected the known too-short input shape."""
    message = str(error)
    return (
        _SHORT_INPUT_VOCODER_ERROR in message
        or _EMPTY_INPUT_VOCODER_ERROR in message
        or "LuxTTS produced no acoustic frames" in message
    )


_MAX_SILENT_UTTERANCE_RETRIES = 3


def _recover_quantized_tts(engine: Celune, language: Optional[str]) -> None:
    """Reload a failed quantized TTS runtime in BF16.

    Args:
        engine: Celune runtime that owns the active TTS backend.
        language: Language requested by the failed generation.

    Raises:
        BackendError: The active voice cannot be resolved or BF16 recovery fails.
    """
    backend = engine.backend
    try:
        backend.disable_runtime_quantization()
        backend.unload_model()
        active_voice = engine.current_voice or backend.default_voice
        if active_voice is None:
            raise BackendError(
                "cannot recover a quantized model without an active voice",
                error_code="tts_quantization_recovery_failed",
            )
        model_id = backend.model_id_for_voice(active_voice)
        model = backend.load_model(model_id, lang=language)
        backend.model = model
        engine.model = model
        engine.model_name = model_id
    except Exception as error:
        engine.fatal()
        if isinstance(error, BackendError):
            raise
        raise BackendError(
            "quantized TTS recovery failed",
            error_code="tts_quantization_recovery_failed",
            error_type=type(error).__name__,
        ) from error


def _process_generation_request(
    engine: Celune,
    item: SpeechRequest,
    caption_worker: Optional[CaptionAlignmentWorker],
    reload_language_model: collections.abc.Callable[[Optional[str]], None],
) -> None:
    """Process one queued speech request on a blocking worker thread."""
    if item.prepared_audio is not None:
        play_prepared_speech(engine, item, caption_worker)
        return

    text = item.text
    display_text = item.display_text
    request_language = item.language
    save_output = item.save
    stream_queue = item.stream_queue
    capture_tutorial_audio = item.audio_capture_queue is not None
    kept_sfx_audio = engine.kept_sfx_audio
    engine.kept_sfx_audio = None

    if engine.exit_requested:
        if stream_queue is not None:
            stream_queue.put(NotAvailableError("stream queue interrupted"))
            stream_queue.put(None)
        if item.audio_capture_queue is not None:
            item.audio_capture_queue.put(
                NotAvailableError("tutorial speech preparation was interrupted")
            )
        release_pipeline(engine)
        return

    fatal_model_reload = False
    while True:
        source_id = next_playback_source_id(engine)
        prepared_sections: list[AudioChunk] = []
        try:
            engine.model_ready.wait()

            if not engine.loaded and not engine.backend.is_fake:
                engine.log(string("ui.core_engine_not_loaded"), "warning")
                engine.locked = False
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("model is not ready")
                    )
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("model is not ready"))
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            start_time = monotonic_time()
            engine.log(f"[GEN] {display_text}")
            engine.log(
                "[GEN] start "
                f"generation={item.generation} text_chars={len(text)} "
                f"backend={getattr(engine.backend, 'name', type(engine.backend).__name__)} "
                f"model={getattr(engine, 'model_name', '')} language={request_language}",
                loglevel="debug",
            )
            speech_len = 0.0
            buffered_speech_len = 0.0
            buffer_target_seconds = smart_buffer_target_seconds(
                engine,
                0.0,
                0.0,
            )
            engine.smart_buffer_target_seconds = buffer_target_seconds
            speech_timing = SpeechTiming(start_time)
            pushed_audio = False
            stream_frame_count = 0
            accepted_stream_frame_count = 0
            caption_language: Optional[str] = None
            caption_alignment_failed: list[bool] = [False]
            pushed_audio_state: list[bool] = [False]
            caption_audio_frames = 0
            caption_audio_sections: list[tuple[int, int, str]] = []

            # these generation parameters are fixed and do not change
            # this only applies to Qwen3-TTS, other backends discard this
            generation_params: collections.abc.Mapping[str, JSONSerializable] = {
                "temperature": 0.15,
                "top_k": 20,
                "top_p": 0.7,
                "repetition_penalty": 1.1,
            }

            if capture_tutorial_audio:
                chunks = list(item.synthesis_sections)
            elif getattr(
                engine.backend, "singing", False
            ) and not text.lstrip().startswith("["):
                message = string("cesn.no_notation_found")
                engine.progress_callback(0, 1)
                engine.log(message, "warning")
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(NotAvailableError(message))
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError(message))
                    stream_queue.put(None)
                release_pipeline(engine)
                engine.change_input_state_callback(locked=False)
                engine.change_voice_lock_state_callback(locked=len(engine.voices) < 2)
                break
            else:
                chunks = prepare_synthesis_chunks(engine, text, split_text)
            if not chunks:
                engine.progress_callback(0, 1)
                engine.error_callback(string("pipeline.nothing_to_say"))
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech has no sections")
                    )
                release_pipeline(engine)
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("nothing to say"))
                    stream_queue.put(None)
                break
            if capture_tutorial_audio:
                engine.progress_callback(0, len(chunks))

            buffer: AudioChunks = []
            full_audio: AudioChunks = []
            generated_text_parts: list[str] = []
            request_generation = item.generation
            caption_word_total = len(display_text.split())
            singing_caption_enabled, caption_alignment_enabled = caption_tracking_modes(
                engine,
                tutorial_audio=capture_tutorial_audio,
                caption_worker_available=caption_worker is not None,
            )
            if not capture_tutorial_audio:
                register_playback_source(
                    engine,
                    source_id,
                    kind="speech",
                    async_caption_audio=caption_alignment_enabled,
                    caption_word_total=(
                        caption_word_total if caption_alignment_enabled else 0
                    ),
                    track_caption_sections=singing_caption_enabled,
                )
            caption_word_ranges = caption_chunk_word_ranges(display_text, chunks)

            for chunk_index, chunk_text in enumerate(chunks):
                if engine.exit_requested:
                    break

                if engine.utterance_force_stop.is_set():
                    break

                chunk_word_start, chunk_word_end = caption_word_ranges[chunk_index]
                chunk_start_frames = caption_chunk_start_frame(
                    engine,
                    source_id,
                    caption_audio_frames if caption_alignment_enabled else 0,
                    use_queued_total=(
                        not caption_alignment_enabled and not singing_caption_enabled
                    ),
                )

                if item.normalize and not getattr(engine.backend, "singing", False):
                    engine.status_callback(string("status.normalizing"))
                    engine.progress_callback(None, None)
                    normalized = engine.normalize(chunk_text)
                    if normalized is not None:
                        if normalized == chunk_text:
                            engine.log(
                                "This input is already normalized.",
                                "warning",
                                loglevel="verbose",
                            )
                        else:
                            differences = sum(
                                x != y for x, y in zip(normalized, chunk_text)
                            ) + abs(len(normalized) - len(chunk_text))

                            if differences > max(5, int(len(chunk_text) * 0.05)):
                                chunk_text = normalized

                if not getattr(engine.backend, "singing", False):
                    generated_text_parts.append(chunk_text)
                is_first_chunk = chunk_index == 0
                last_timing: Optional[dict] = None

                with engine.model_lock:
                    resolve_generation_language = getattr(
                        engine.backend,
                        "resolve_generation_language",
                        None,
                    )
                    if callable(resolve_generation_language):
                        target_language = resolve_generation_language(request_language)
                    else:
                        target_language = request_language
                    caption_language = (
                        target_language if isinstance(target_language, str) else None
                    )

                    try:
                        reload_language_model(caption_language)
                    except Exception:
                        fatal_model_reload = engine.model is None
                        raise

                    for (
                        audio_chunk,
                        sr,  # 24 kHz if Qwen3 or Celune Mini, 48 kHz if VoxCPM2
                        timing,
                    ) in engine.backend.generate_stream(
                        engine.model,
                        text=chunk_text,
                        language=target_language,
                        chunk_size=engine.chunk_size,
                        instruct=effective_voice_prompt(engine),
                        voice=engine.current_voice,
                        temperature=generation_params["temperature"],
                        top_k=generation_params["top_k"],
                        top_p=generation_params["top_p"],
                        repetition_penalty=generation_params["repetition_penalty"],
                    ):
                        stream_frame_count += 1
                        if timing is not None:
                            last_timing = timing
                        if engine.exit_requested:
                            break

                        if (
                            engine.utterance_force_stop.is_set()
                            or request_generation
                            != getattr(engine, "_speech_generation", request_generation)
                        ):
                            break

                        first_chunk_time = None
                        if timing is not None:
                            raw_first_chunk_time = timing.get("first_chunk_time")
                            if isinstance(raw_first_chunk_time, float):
                                first_chunk_time = raw_first_chunk_time

                        speech_timing.mark_first_chunk(first_chunk_time)

                        if isinstance(audio_chunk, torch.Tensor):
                            audio_chunk = audio_chunk.cpu().numpy()

                        audio_chunk = to_48khz(
                            np.asarray(audio_chunk, dtype=np.float32), sr
                        )
                        has_audio = bool(audio_chunk.size and np.any(audio_chunk))
                        if (
                            stream_frame_count <= 3
                            or audio_chunk.size == 0
                            or not has_audio
                        ):
                            engine.log(
                                "[STREAM] core received "
                                f"frame={stream_frame_count} samples={audio_chunk.size} "
                                f"sample_rate={sr} nonzero={has_audio}",
                                loglevel="debug",
                            )
                        if not has_audio:
                            engine.log(
                                f"[STREAM] core discarded frame={stream_frame_count}",
                                loglevel="debug",
                            )
                            continue
                        accepted_stream_frame_count += 1

                        if engine.speed != 1.0 and engine.can_use_rubberband:
                            try:
                                audio_chunk = rb.time_stretch(
                                    audio_chunk, BASE_SR, engine.speed
                                )
                            except RuntimeError:
                                engine.log(
                                    string("pipeline.rubber_band_unavailable"),
                                    "warning",
                                )
                                engine.can_use_rubberband = False
                            else:
                                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                        if engine.reverb.strength > 0.0:
                            audio_chunk = engine.reverb.process(audio_chunk, BASE_SR)
                            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                        if is_first_chunk:
                            audio_chunk = soften(audio_chunk, BASE_SR, end=False)
                            is_first_chunk = False

                        if (
                            engine.exit_requested
                            or engine.utterance_force_stop.is_set()
                            or request_generation
                            != getattr(engine, "_speech_generation", request_generation)
                        ):
                            break

                        buffer.append(audio_chunk)
                        full_audio.append(audio_chunk)
                        if caption_alignment_enabled:
                            caption_audio_frames += len(audio_chunk)
                        section_final = False
                        if singing_caption_enabled:
                            section_final = record_caption_playback_audio_chunk(
                                engine,
                                source_id,
                                len(audio_chunk),
                                timing,
                            )
                        chunk_dur = len(audio_chunk) / BASE_SR
                        speech_len += chunk_dur
                        buffered_speech_len += chunk_dur
                        generation_elapsed = max(
                            monotonic_time() - start_time,
                            1e-6,
                        )
                        buffer_target_seconds = smart_buffer_target_seconds(
                            engine,
                            speech_len,
                            generation_elapsed,
                        )
                        engine.smart_buffer_target_seconds = buffer_target_seconds

                        if (
                            not capture_tutorial_audio
                            and not caption_alignment_enabled
                            and (
                                buffer_target_seconds <= 0.0
                                or buffered_speech_len >= buffer_target_seconds
                                or (singing_caption_enabled and section_final)
                            )
                        ):
                            pushed_audio = flush_buffered_speech_chunks(
                                engine,
                                source_id,
                                buffer,
                                speech_timing,
                                pushed_audio,
                                stream_queue,
                                caption_text=None,
                            )
                            buffered_speech_len = 0.0

                    if (
                        not engine.exit_requested
                        and not engine.utterance_force_stop.is_set()
                        and last_timing is not None
                        and last_timing.get("is_final")
                        and bool(last_timing.get("missing_eos"))
                    ):
                        engine.log(
                            string("pipeline.token_limit_reached"),
                            "warning",
                        )

                if capture_tutorial_audio:
                    if engine.reverb.strength > 0.0:
                        tail = engine.reverb.flush()
                        if len(tail) > 0:
                            buffer.append(np.asarray(tail, dtype=np.float32))
                            full_audio.append(np.asarray(tail, dtype=np.float32))
                        engine.reverb.reset()
                    prepared_sections.append(
                        np.concatenate(buffer).astype(np.float32, copy=False)
                        if buffer
                        else np.empty(0, dtype=np.float32)
                    )
                    buffer.clear()
                    if (
                        not engine.exit_requested
                        and not engine.utterance_force_stop.is_set()
                        and request_generation
                        == getattr(engine, "_speech_generation", request_generation)
                    ):
                        engine.progress_callback(chunk_index + 1, len(chunks))
                elif (
                    not engine.exit_requested
                    and not engine.utterance_force_stop.is_set()
                    and request_generation
                    == getattr(engine, "_speech_generation", request_generation)
                ):
                    word_start_frames: tuple[int, ...] = ()
                    if caption_alignment_enabled and caption_worker is not None:
                        if (
                            caption_audio_frames > chunk_start_frames
                            and chunk_text.split()
                        ):
                            caption_audio_sections.append(
                                (chunk_start_frames, caption_audio_frames, chunk_text)
                            )
                    else:
                        pushed_audio = flush_buffered_speech_chunks(
                            engine,
                            source_id,
                            buffer,
                            speech_timing,
                            pushed_audio,
                            stream_queue,
                            caption_text=None,
                        )
                    buffered_speech_len = 0.0
                    if (
                        (not caption_alignment_enabled or caption_worker is None)
                        and not singing_caption_enabled
                        and source_id in playback_caption_states(engine)
                    ):
                        with engine.queue_lock:
                            source_meta = playback_source_meta(engine).get(source_id)
                            chunk_end_frames = (
                                int(float(source_meta.get("total_frames", 0.0)))
                                if isinstance(source_meta, dict)
                                else chunk_start_frames
                            )
                        record_caption_playback_segment(
                            engine,
                            source_id,
                            chunk_start_frames,
                            chunk_end_frames,
                            chunk_word_start,
                            chunk_word_end,
                            tuple(chunk_text.split()),
                            word_start_frames,
                        )

            timing_text = (
                " ".join(generated_text_parts) if generated_text_parts else text
            )
            if generated_text_parts:
                text = timing_text

            generation_time = max(monotonic_time() - start_time, 1e-6)

            engine.log(
                "[GEN] stream complete "
                f"generation={request_generation} seconds={speech_len:.3f} "
                f"elapsed={generation_time:.3f} buffered={buffered_speech_len:.3f}",
                loglevel="debug",
            )

            engine.log(
                f"[STREAM] core totals received={stream_frame_count} "
                f"accepted={accepted_stream_frame_count}",
                loglevel="debug",
            )

            if engine.exit_requested:
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech preparation was interrupted")
                    )
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                elif stream_queue is not None:
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            if engine.utterance_force_stop.is_set() or request_generation != getattr(
                engine, "_speech_generation", request_generation
            ):
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech preparation was interrupted")
                    )
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                elif stream_queue is not None:
                    stream_queue.put(None)
                engine.reverb.reset()
                break

            generation_speed = speech_len / generation_time
            engine.log(
                f"{format_stat_duration(speech_len)}, "
                f"{format_number(generation_time, 2)}s, "
                f"{format_number(generation_speed, 2)}x"
            )
            remember_smart_buffer_speed(engine, generation_speed)
            engine.smart_buffer_target_seconds = smart_buffer_target_seconds(
                engine,
                speech_len,
                generation_time,
            )
            engine.log(f"TTFC {format_number(speech_timing.ttfc_ms(), 1)}ms")

            if buffer and not capture_tutorial_audio and not caption_alignment_enabled:
                flush_buffered_speech_chunks(
                    engine,
                    source_id,
                    buffer,
                    speech_timing,
                    pushed_audio,
                    stream_queue,
                    caption_text=None,
                )

            engine.log("[GEN] done")

            saved_path = None
            analysis_audio = None
            if capture_tutorial_audio:
                audio_capture_queue = item.audio_capture_queue
                if audio_capture_queue is None:
                    raise NotAvailableError("tutorial capture queue is unavailable")
                if finish_tutorial_audio_capture(
                    engine,
                    audio_capture_queue,
                    speech_len,
                    full_audio,
                    prepared_sections,
                    item.silent_retry_count,
                    _MAX_SILENT_UTTERANCE_RETRIES,
                ):
                    item = replace(
                        item,
                        silent_retry_count=item.silent_retry_count + 1,
                    )
                    continue
                break

            if not engine.exit_requested:
                if engine.reverb.strength > 0.0:
                    tail = engine.reverb.flush()
                    if len(tail) > 0:
                        if caption_alignment_enabled and caption_worker is not None:
                            buffer.append(np.asarray(tail, dtype=np.float32))
                            full_audio.append(tail)
                            caption_audio_frames += len(tail)
                        else:
                            queued_tail = queue_playback_chunk(
                                engine,
                                source_id,
                                tail,
                                BASE_SR,
                            )
                            if queued_tail and stream_queue is not None:
                                stream_queue.put(tail.copy())
                            if queued_tail:
                                buffer.append(tail)
                                full_audio.append(tail)

                engine.reverb.reset()
                is_silent = False
                silence_tier = 0
                full_audio_array = np.concatenate(full_audio) if full_audio else None
                if full_audio_array is not None:
                    is_silent, silence_tier = is_silent_utterance(full_audio_array)

                if is_silent and silence_tier == 2:
                    if item.silent_retry_count < _MAX_SILENT_UTTERANCE_RETRIES:
                        engine.regenerate = True
                        if caption_alignment_enabled and caption_worker is not None:
                            caption_worker.submit_done(
                                engine,
                                source_id,
                                stream_queue,
                                release_pipeline_when_finished=False,
                                notify_idle_when_finished=False,
                                finish_stream=False,
                            )
                        else:
                            queue_playback_done(
                                engine,
                                source_id,
                                release_pipeline_when_finished=False,
                                notify_idle_when_finished=False,
                            )
                        item = replace(
                            item,
                            silent_retry_count=item.silent_retry_count + 1,
                        )
                        engine.log(
                            string(
                                "pipeline.silent_regenerating",
                                retry_count=item.silent_retry_count,
                                max_retries=_MAX_SILENT_UTTERANCE_RETRIES,
                            ),
                            "warning",
                        )
                        continue
                    engine.log(
                        string(
                            "pipeline.silent_regeneration_limit_reached",
                            max_retries=_MAX_SILENT_UTTERANCE_RETRIES,
                        ),
                        "warning",
                    )
                if is_silent and silence_tier == 1:
                    engine.log(string("pipeline.may_be_silent"), "warning")

                if caption_alignment_enabled and caption_worker is not None and buffer:
                    if full_audio_array is not None:
                        buffer.clear()
                        buffer.append(full_audio_array)
                        full_audio.clear()
                    if caption_audio_sections:
                        transcript, alignment_sections = (
                            build_caption_alignment_sections(
                                display_text,
                                tuple(caption_audio_sections),
                            )
                        )
                        caption_worker.submit_chunk(
                            engine,
                            source_id,
                            buffer,
                            speech_timing,
                            stream_queue,
                            display_text,
                            transcript,
                            tuple(display_text.split()),
                            caption_language,
                            0,
                            caption_word_total,
                            tuple(transcript.split()),
                            caption_alignment_failed,
                            pushed_audio_state,
                            BASE_SR,
                            alignment_sections,
                        )
                    else:
                        caption_worker.submit_audio(
                            engine,
                            source_id,
                            buffer,
                            speech_timing,
                            stream_queue,
                            pushed_audio_state,
                        )

                engine.total_generated_speech_seconds += speech_len

                if save_output and full_audio_array is not None:
                    wav = full_audio_array
                    analysis_audio = wav.copy()
                    if kept_sfx_audio is not None:
                        wav = np.concatenate([*kept_sfx_audio, wav])
                    timestamp = datetime.datetime.now(datetime.UTC).strftime(
                        "%Y%m%d%H%M%S"
                    )

                    singing_backend = bool(getattr(engine.backend, "singing", False))
                    first_words = _output_filename_fragment(
                        text,
                        singing_backend=singing_backend,
                    )
                    filename_prefix = _output_filename_prefix(
                        singing_backend=singing_backend
                    )

                    output_dir = outputs_dir()
                    if not output_dir.exists():
                        engine.log(string("pipeline.outputs_path_creating"), "warning")
                        try:
                            output_dir.mkdir(parents=True)
                        except OSError as e:
                            engine.log(
                                format_error_message(
                                    string("pipeline.outputs_create_failed"),
                                    e,
                                    engine.log_level,
                                ),
                                "warning",
                            )

                    if output_dir.exists():
                        file_name = f"{filename_prefix}_{timestamp}_{first_words}.flac"
                        saved_path = str(pathlib.Path("outputs") / file_name)
                        actual_saved_path = str(output_dir / file_name)
                        sample_rate = BASE_SR
                        subtype = "PCM_24"
                        metadata = celune_metadata_payload(
                            engine,
                            text=text,
                            display_text=display_text,
                            generation_params=generation_params,
                            sample_rate=sample_rate,
                            subtype=subtype,
                            included_kept_sfx=kept_sfx_audio is not None,
                        )
                        try:
                            write_celune_flac(
                                engine,
                                actual_saved_path,
                                wav,
                                sample_rate,
                                subtype=subtype,
                                metadata=metadata,
                            )
                        except Exception as e:
                            engine.log(
                                format_error_message(
                                    string("pipeline.flac_save_failed"),
                                    e,
                                    engine.log_level,
                                ),
                                "warning",
                            )
                            saved_path = None

                engine.recently_saved = saved_path
                if caption_alignment_enabled and caption_worker is not None:
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        saved_path=saved_path,
                        analysis_audio=analysis_audio,
                    )
                    release_pipeline(engine, playback_idle=False)
                else:
                    queue_playback_done(
                        engine,
                        source_id,
                        release_pipeline_when_finished=True,
                        saved_path=saved_path,
                        analysis_audio=analysis_audio,
                    )
                    if stream_queue is not None:
                        stream_queue.put(None)
            break
        except Exception as original_error:
            if engine.exit_requested:
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                release_pipeline(engine)
                break

            if caption_worker is not None and not capture_tutorial_audio:
                caption_worker.cancel_source(source_id)
                caption_worker.submit_done(
                    engine,
                    source_id,
                    stream_queue,
                    release_pipeline_when_finished=False,
                    notify_idle_when_finished=False,
                    finish_stream=False,
                )

            cuda_oom = is_cuda_out_of_memory(original_error)
            recovery_error: Optional[Exception] = None
            if (
                not cuda_oom
                and not isinstance(original_error, CESNError)
                and getattr(engine.backend, "quantization_active", False)
            ):
                try:
                    _recover_quantized_tts(engine, request_language)
                except Exception as caught_recovery_error:
                    recovery_error = caught_recovery_error

            error = recovery_error or original_error
            cesn_error = isinstance(error, CESNError)
            if item.audio_capture_queue is not None:
                item.audio_capture_queue.put(error)
            short_input_error = _is_short_input_generation_error(error)
            input_too_short_message = string("pipeline.input_too_short")
            if cuda_oom:
                release_cuda_after_oom()
                engine.log(
                    format_error_message(
                        tagged_string("pipeline.gen_error", "GEN ERROR"),
                        error,
                        engine.log_level,
                    ),
                    "warning",
                )
                engine.status_callback(
                    string("pipeline.generation_oom", app_name=APP_NAME), "warning"
                )
            elif short_input_error:
                engine.log(input_too_short_message, "warning")
            elif isinstance(error, CESNError):
                engine.log(format_cesn_error(text, error), "error")
            else:
                engine.log(
                    format_error_message(
                        tagged_string("pipeline.gen_error", "GEN ERROR"),
                        error,
                        engine.log_level,
                    ),
                    "error",
                )
            if stream_queue is not None and caption_worker is not None:
                caption_worker.submit_stream_error(stream_queue, error)
            elif stream_queue is not None:
                stream_queue.put(error)
                stream_queue.put(None)
            engine.cur_state = (
                "idle" if short_input_error or cuda_oom or cesn_error else "error"
            )
            release_pipeline(engine)
            if cesn_error:
                engine.status_callback(string("status.idle"))
            if fatal_model_reload:
                engine.fatal()
            engine.progress_callback(0, 1)
            if short_input_error:
                engine.status_callback(input_too_short_message, "warning")
            elif cesn_error:
                engine.change_input_state_callback(locked=False)
                engine.change_voice_lock_state_callback(locked=len(engine.voices) < 2)
            else:
                engine.error_callback(
                    string("pipeline.could_not_generate", app_name=APP_NAME)
                )
            break


async def generation_worker_job(
    engine: Celune,
    reload_language_model: collections.abc.Callable[[Optional[str]], None],
) -> None:
    """Generate audio tokens and send them to the audio pipeline as an async job.

    Args:
        engine: Runtime that owns the generation queue and playback state.
    """
    caption_worker = None
    if engine.config.get("captions") is True:
        from .captions import CaptionAlignmentWorker

        caption_worker = CaptionAlignmentWorker(engine.log, engine.log_level)

    while True:
        item = await run_in_daemon_thread(engine.text_queue.get)
        engine.regenerate = False

        if item is engine.sentinel:
            if caption_worker is not None:
                await run_in_daemon_thread(
                    lambda: caption_worker.close(
                        lambda: engine.audio_queue.put(engine.sentinel)
                    )
                )
            else:
                try:
                    engine.audio_queue.put_nowait(engine.sentinel)
                except queue.Full:
                    await run_in_daemon_thread(
                        lambda: engine.audio_queue.put(engine.sentinel)
                    )
            break

        request = cast(SpeechRequest, item)
        if request.generation != getattr(
            engine, "_speech_generation", request.generation
        ):
            if request.stream_queue is not None:
                request.stream_queue.put(None)
            if request.audio_capture_queue is not None:
                request.audio_capture_queue.put(
                    NotAvailableError("speech generation was superseded")
                )
            if request.playback_source_queue is not None:
                request.playback_source_queue.put(
                    NotAvailableError("prepared playback was superseded")
                )
            continue

        engine.utterance_force_stop.clear()
        engine._active_speech_generation = request.generation
        try:
            await run_in_daemon_thread(
                lambda request=request: _process_generation_request(
                    engine,
                    request,
                    caption_worker,
                    reload_language_model,
                )
            )
        finally:
            engine._active_speech_generation = None
