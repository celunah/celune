# pylint: disable=W0212, W0718, R0912, R0914, R0915
"""Speech pipeline helpers for Celune."""

import os
import re
import time
import queue
import random
import contextlib
from typing import TYPE_CHECKING

import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyrubberband as rb

from .dsp import _resample_audio, _soften_onset, _split, _to_48khz
from .exceptions import NotAvailableError
from .utils import format_number

if TYPE_CHECKING:
    from .celune import Celune


def clear_queue(q: queue.Queue) -> None:
    """Drain all pending items from a queue."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def close_stream(engine: "Celune", abort: bool = False) -> None:
    """Close the current audio stream if one exists."""
    if engine.stream is None:
        return

    with contextlib.suppress(Exception):
        if abort:
            engine.stream.abort()
        else:
            engine.stream.stop()

    with contextlib.suppress(Exception):
        engine.stream.close()

    engine._stream = None
    engine._current_sr = None


def force_stop_speech(engine: "Celune") -> bool:
    """Forcefully stop Celune from speaking."""
    with engine.say_lock:
        is_active = engine.locked or (engine.cur_state in {"generating", "speaking"})

    if not is_active:
        engine.utterance_force_stop.clear()
        return False

    engine.log("Forcefully stopping speech.")
    engine.utterance_force_stop.set()

    with engine.queue_lock:
        clear_queue(engine.text_queue)
        clear_queue(engine.audio_queue)
        engine.audio_queue.put(engine.force_stop_marker)

    return True


def acquire_pipeline(engine: "Celune", action: str) -> bool:
    """Atomically claim Celune's shared playback pipeline."""
    with engine.say_lock:
        if engine.dev:
            engine.log(f"[LOCK] acquire requested by {action}, locked={engine.locked}")
        if engine.locked:
            engine.log(f"Tried to {action} while Celune was busy.", "warning")
            engine.error_callback("Celune is currently busy")
            return False

        engine.locked = True
        engine.playback_done.clear()
        if engine.dev:
            engine.log(f"[LOCK] acquired by {action}")
        return True


def release_pipeline(engine: "Celune") -> None:
    """Release Celune's shared playback pipeline."""
    with engine.say_lock:
        engine.locked = False
        engine.playback_done.set()
        engine.cur_state = "idle"
        if engine.dev:
            engine.log("[LOCK] released")


def say(engine: "Celune", text: str) -> bool:
    """Queue text for Celune to say."""
    if not engine.model_ready.is_set():
        engine.status_callback("Waiting for model")
        engine.log("Speak request is waiting for model reload to finish.", "info")

    engine.model_ready.wait()

    if not engine.loaded:
        engine.log("Model became unavailable before speaking.", "warning")
        engine.error_callback("Celune is not currently ready")
        return False

    normalized = None
    if engine.use_normalization:
        engine.status_callback("Normalizing")
        normalized = engine.normalize(text)

    if not acquire_pipeline(engine, "speak"):
        return False

    try:
        if not engine.loaded:
            engine.log("Model became unavailable before queueing speech.", "warning")
            engine.error_callback("Celune is not currently ready")
            release_pipeline(engine)
            return False

        engine.cur_state = "generating"
        engine.text_queue.put(normalized if normalized is not None else text)
        engine.status_callback("Generating")
        return True
    except Exception:
        release_pipeline(engine)
        raise


def play(engine: "Celune", sound_path: str) -> bool:
    """Play a sound via Celune's pipeline."""
    if not os.path.exists(sound_path):
        engine.log(f"Celune cannot find {sound_path}.", "warning")
        return False

    if not acquire_pipeline(engine, "play"):
        return False

    try:
        audio, sr = sf.read(sound_path, dtype="float32")
        audio_len = len(audio) / sr
        engine.log(
            f"Sample rate: {sr} Hz, length: {format_number(audio_len, 2)} seconds"
        )

        audio = _resample_audio(audio, sr)

        engine.cur_state = "speaking"
        for chunk in _split(audio, sr, engine.chunk_size):
            engine.audio_queue.put((chunk, 48000, None))
        engine.audio_queue.put(engine.utterance_done)

        engine.status_callback(f"Playing {sound_path}")
        return True
    except Exception:
        release_pipeline(engine)
        raise


def close(engine: "Celune") -> None:
    """Shut off Celune and exit."""
    engine.log("Exiting...")
    engine._exit_requested = True

    with engine.queue_lock:
        clear_queue(engine.text_queue)
        clear_queue(engine.audio_queue)

    engine.text_queue.put(engine.sentinel)
    engine.audio_queue.put(engine.sentinel)

    if engine.generation_thread is not None:
        engine.generation_thread.join(timeout=2)

    if engine.playback_thread is not None:
        engine.playback_thread.join(timeout=2)

    close_stream(engine, abort=True)
    engine.glow.leave()
    engine.glow.finished.wait(timeout=5)


def split_text(engine: "Celune", text: str) -> list[str]:
    """Split text into chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    current_sentences = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue

        new_length = current_length + len(sentence) + 1

        if len(current_sentences) < engine.sentences_per_chunk and new_length <= (
            engine.sentences_per_chunk * 150
        ):
            current_sentences.append(sentence)
            current_length = new_length
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            current_sentences = [sentence]
            current_length = len(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    engine.log(f"Chunks: {len(chunks)}")
    return chunks


def generation_worker(engine: "Celune") -> None:
    """Generate audio tokens and send them to the audio pipeline."""
    while True:
        text = engine.text_queue.get()

        if text is engine.sentinel:
            engine.audio_queue.put(engine.sentinel)
            break

        if engine.exit_requested:
            engine.locked = False
            continue

        try:
            engine.model_ready.wait()

            if not engine.loaded:
                engine.log("Skipping generation because model is not ready.", "warning")
                engine.locked = False
                continue

            start_time = time.perf_counter()
            engine.log(f"[GEN] {text}")
            speech_len = 0.0

            chunks = split_text(engine, text)
            buffer = []

            with engine.model_lock:
                if engine.model is None:
                    raise NotAvailableError("self.model is None")

                for chunk_index, chunk_text in enumerate(chunks):
                    if engine.exit_requested:
                        break

                    if engine.utterance_force_stop.is_set():
                        break

                    is_first_chunk = chunk_index == 0

                    for (
                        audio_chunk,
                        sr,  # 24 kHz if Qwen3, 48 kHz if VoxCPM2
                        timing,
                    ) in engine.backend.generate_stream(
                        engine.model,
                        text=chunk_text,
                        language=engine.language,
                        chunk_size=engine.chunk_size,
                        instruct=engine.voice_prompt,
                        voice=engine.current_voice,
                        temperature=0.15,
                        top_k=20,
                        top_p=0.7,
                        repetition_penalty=1.1,
                    ):
                        if engine.exit_requested:
                            break

                        if engine.utterance_force_stop.is_set():
                            break

                        if hasattr(audio_chunk, "cpu"):
                            audio_chunk = audio_chunk.cpu().numpy()

                        audio_chunk = _to_48khz(audio_chunk, sr)

                        if engine.speed != 1.0 and engine.can_use_rubberband:
                            try:
                                audio_chunk = rb.time_stretch(
                                    audio_chunk, 48000, engine.speed
                                )
                            except RuntimeError:
                                engine.log(
                                    "Rubber Band is unavailable, speed controls disabled.",
                                    "warning",
                                )
                                engine.can_use_rubberband = False
                        if engine.reverb.strength > 0.0:
                            audio_chunk = engine.reverb.process(audio_chunk, 48000)

                        if is_first_chunk:
                            audio_chunk = _soften_onset(audio_chunk, 48000)
                            is_first_chunk = False

                        if engine.exit_requested:
                            break

                        buffer.append(audio_chunk)
                        engine.audio_queue.put((audio_chunk, 48000, timing))

                        chunk_dur = len(audio_chunk) / 48000
                        speech_len += chunk_dur

            generation_time = time.perf_counter() - start_time

            if engine.exit_requested:
                release_pipeline(engine)
                continue

            if engine.utterance_force_stop.is_set():
                engine.reverb.reset()
                continue

            engine.log(
                f"[GEN] {format_number(speech_len, 2)} seconds, took {format_number(generation_time, 2)} seconds, "
                f"RTF: {format_number(speech_len / generation_time, 2)}"
            )
            engine.status_callback("Speaking")
            engine.cur_state = "speaking"
            engine.log("[GEN] done")
            engine.queue_avail_callback()

            if not engine.exit_requested:
                if engine.reverb.strength > 0.0:
                    tail = engine.reverb.flush()
                    if len(tail) > 0:
                        engine.audio_queue.put((tail, 48000, None))
                        buffer.append(tail)

                engine.reverb.reset()
                engine.audio_queue.put(engine.utterance_done)

            if buffer:
                wav = np.concatenate(buffer)
                timestamp = time.time_ns() // 1000000
                if not os.path.exists("outputs"):
                    engine.log("Outputs path not found, creating...", "warning")
                    try:
                        os.mkdir("outputs")
                    except OSError as e:
                        engine.log(
                            "Cannot create outputs directory, not saving WAV file: "
                            f"{engine.format_error(e, engine.dev)}",
                            "warning",
                        )

                if os.path.exists("outputs"):
                    sf.write(
                        f"outputs/celune_speech_{timestamp}.wav",
                        wav,
                        48000,
                        subtype="PCM_24",
                    )

        except Exception as e:
            if engine.exit_requested:
                release_pipeline(engine)
                continue

            engine.log(f"[GEN ERROR] {engine.format_error(e, engine.dev)}", "error")
            engine.cur_state = "error"
            engine.locked = False
            engine.playback_done.set()
            engine.error_callback("Celune could not generate the input")


def playback_worker(engine: "Celune") -> None:
    """Receive audio chunks and play them."""
    started = False

    while True:
        if engine.exit_requested:
            with engine.queue_lock:
                clear_queue(engine.audio_queue)

            close_stream(engine, abort=True)
            release_pipeline(engine)
            engine.idle_callback()
            return

        if not started:
            while engine.audio_queue.qsize() < engine.prebuffer_chunks:
                if engine.exit_requested:
                    break
                time.sleep(0.01)

            if engine.exit_requested:
                continue

        item = engine.audio_queue.get()

        if item is engine.sentinel:
            break

        if item is engine.force_stop_marker:
            engine.utterance_force_stop.clear()
            close_stream(engine, abort=True)
            engine.playback_done.set()
            release_pipeline(engine)
            engine.idle_callback()
            started = False
            continue

        if engine.exit_requested:
            continue

        if item is engine.utterance_done:
            engine.playback_done.set()

            more_pending = (not engine.audio_queue.empty()) or (
                not engine.text_queue.empty()
            )

            if more_pending:
                silence = np.zeros((48000, 2), dtype=np.float32)
                if engine.stream is not None and not engine.exit_requested:
                    engine.stream.write(silence)
            else:
                release_pipeline(engine)
                engine.idle_callback()

                if random.random() < 0.01:
                    flavor_texts = [
                        "I will speak.",
                        "I'll answer.",
                        "I'm always listening.",
                        "I'm all ears.",
                        "You shall hear.",
                    ]

                    choice = random.choice(flavor_texts)

                    if choice == getattr(engine, "_last_flavor", None):
                        choice = random.choice(flavor_texts)

                    engine._last_flavor = choice
                    engine.log(f"Just type. {choice}")
                else:
                    engine.log("Ready to speak.")

                avail, total = tuple(v / 1024**3 for v in torch.cuda.mem_get_info(0))
                if avail <= 2:
                    engine.log(
                        "Celune is running out of VRAM "
                        f"({format_number(avail, 2)}/{format_number(total, 2)} GB available).",
                        "warning",
                    )
                    engine.log(
                        "Please close any memory-resident applications to improve performance.",
                        "warning",
                    )
                else:
                    engine.log(
                        f"Available VRAM: {format_number(avail, 2)}/{format_number(total, 2)} GB"
                    )
            continue

        audio_chunk, sr, _ = item

        if engine.stream is None:
            try:
                engine.current_sr = sr
                engine.stream = sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    dtype="float32",
                    blocksize=0,
                )
                engine.stream.start()
                started = True
                engine.log(f"[PLAY] started stream at {sr} Hz")
            except sd.PortAudioError:
                if not engine.audio_unavailable:
                    engine.log("Celune could not initialize the audio stream.", "error")
                    engine.log("No suitable audio device is available.", "error")
                    engine.error_callback("No suitable audio devices")
                engine._audio_unavailable = True

        if engine.exit_requested:
            continue

        try:
            engine.glow.glow(audio_chunk)
            engine.stream.write(audio_chunk)
        except Exception as e:
            engine.log(f"[PLAY ERROR] {engine.format_error(e, engine.dev)}", "error")
            engine.error_callback("Playback error")
            close_stream(engine, abort=True)
            engine._stream = None
            engine._current_sr = None
            continue
