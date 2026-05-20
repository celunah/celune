# SPDX-License-Identifier: MIT
"""Speech pipeline helpers for Celune."""

import os
import re
import json
import time
import queue
import random
import pathlib
import datetime
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np
import numpy.typing as npt
import soundfile as sf
import sounddevice as sd
import pyrubberband as rb
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue

from .dsp import (
    _resample_audio,
    _soften,
    _split,
    _to_48khz,
    is_silent_utterance,
    readiness_signal,
)
from .exceptions import NotAvailableError
from .utils import (
    format_number,
    run_async,
    format_error,
    detect_language,
    is_april_fools,
    rng_replace,
)
from .analysis import analyze_voice_audio
from .constants import BASE_SR, N_A_NUMERIC, JSON, JSONSerializable
from . import __version__

if TYPE_CHECKING:
    from .celune import Celune

_FLAC_MAGIC = b"fLaC"
_FLAC_STREAMINFO_BLOCK = 0
_FLAC_VORBIS_COMMENT_BLOCK = 4
_MAX_FLAC_METADATA_BLOCK_SIZE = 0xFFFFFF


@dataclass(frozen=True)
class SpeechRequest:
    """Queued speech input and output persistence preference."""

    text: str
    display_text: str
    save: bool = True
    stream_queue: Optional[queue.Queue] = None
    normalize: bool = False


@dataclass(frozen=True)
class SpeechDone:
    """Playback completion marker for one generated utterance."""

    saved_path: Optional[str] = None
    analysis_audio: Optional[npt.NDArray[np.float32]] = None


@dataclass
class SpeechTiming:
    """Timing data for a generated speech utterance."""

    start_time: float
    first_chunk_time: Optional[float] = None
    first_playback_time: Optional[float] = None

    def mark_first_chunk(self) -> None:
        """Record when the backend yields its first audio chunk.

        Returns:
            None: The time the first chunk was received.
        """
        if self.first_chunk_time is None:
            self.first_chunk_time = time.monotonic()

    def mark_first_playback(self) -> None:
        """Record when the first audio chunk is sent to the output stream.

        Returns:
            None: The time the first audio chunk was sent to the playback pipeline.
        """
        if self.first_playback_time is None:
            self.first_playback_time = time.monotonic()

    def ttfc_ms(self) -> float:
        """Return time to first generated chunk in milliseconds.

        Returns:
            float: How much time it took to generate the first chunk.
        """
        if self.first_chunk_time is None:
            return N_A_NUMERIC

        return (self.first_chunk_time - self.start_time) * 1000

    def ttfp_seconds(self) -> float:
        """Return time to first playback in seconds.

        Returns:
            float: How much time it took to play any part of the current utterance.
        """
        if self.first_playback_time is None:
            return N_A_NUMERIC

        return self.first_playback_time - self.start_time


def _celune_metadata_payload(
    engine: "Celune",
    *,
    text: str,
    display_text: str,
    generation_params: JSON,
    sample_rate: int,
    subtype: str,
    included_kept_sfx: bool,
) -> JSON:
    """Build the Celune generation metadata payload.

    Args:
        engine: The instance of Celune to use data from.
        text: The input text given to Celune.
        display_text: The displayed text shown in Celune's UI.
        generation_params: The generation parameters used with this generation.
        sample_rate: The saved sample rate in Hz.
        subtype: The saved audio subtype.
        included_kept_sfx: Whether the included utterance has a preceding sound effect.

    Returns:
        JSON: JSON-serializable metadata.
    """
    return {
        "format": "celune_metadata",
        "format_version": 1,
        "celune_version": __version__,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "text": text,
        "display_text": display_text,
        "backend": getattr(engine, "tts_backend", None),
        "backend_mode": engine.config.get("qwen3_mode"),
        "qwen3_x_vector_only": getattr(engine.backend, "x_vector_only", None),
        "model_name": getattr(engine, "model_name", ""),
        "voice": getattr(engine, "current_voice", None),
        "voice_prompt": getattr(engine, "voice_prompt", None),
        "language": getattr(engine, "language", None),
        "chunk_size": getattr(engine, "chunk_size", None),
        "speed": getattr(engine, "speed", None),
        "reverb_strength": getattr(engine.reverb, "strength", None),
        "use_normalizer": getattr(engine, "use_normalization", None),
        "sample_rate": sample_rate,
        "subtype": subtype,
        "included_kept_sfx": included_kept_sfx,
        "generation": generation_params,
    }


def _valid_vorbis_comment_key(key: str) -> bool:
    """Return whether ``key`` is a valid Vorbis comment field name."""
    return (
        bool(key) and "=" not in key and all(0x20 <= ord(char) <= 0x7D for char in key)
    )


def _read_vorbis_string(payload: bytes, offset: int) -> tuple[bytes, int]:
    """Read one little-endian length-prefixed Vorbis comment string."""
    if offset + 4 > len(payload):
        raise ValueError("truncated Vorbis comment")

    length = int.from_bytes(payload[offset : offset + 4], "little")
    offset += 4
    end = offset + length
    if end > len(payload):
        raise ValueError("truncated Vorbis comment")

    return payload[offset:end], end


def _parse_vorbis_comment_block(payload: bytes) -> tuple[bytes, list[tuple[str, str]]]:
    """Parse a Vorbis comment block into a vendor string and field pairs."""
    vendor, offset = _read_vorbis_string(payload, 0)
    if offset + 4 > len(payload):
        raise ValueError("truncated Vorbis comment list")

    comment_count = int.from_bytes(payload[offset : offset + 4], "little")
    offset += 4
    comments: list[tuple[str, str]] = []
    for _ in range(comment_count):
        raw_comment, offset = _read_vorbis_string(payload, offset)
        decoded = raw_comment.decode("utf-8", errors="replace")
        key, separator, value = decoded.partition("=")
        if separator and _valid_vorbis_comment_key(key):
            comments.append((key, value))

    return vendor, comments


def _encode_vorbis_comment_block(
    vendor: bytes, comments: list[tuple[str, str]]
) -> bytes:
    """Encode Vorbis comments into a FLAC metadata block payload."""
    payload = bytearray()
    payload.extend(len(vendor).to_bytes(4, "little"))
    payload.extend(vendor)
    payload.extend(len(comments).to_bytes(4, "little"))
    for key, value in comments:
        raw_comment = f"{key}={value}".encode("utf-8")
        payload.extend(len(raw_comment).to_bytes(4, "little"))
        payload.extend(raw_comment)

    return bytes(payload)


def _flac_metadata_blocks(data: bytes) -> tuple[list[tuple[int, bytes]], int]:
    """Return FLAC metadata blocks and the byte offset where audio frames start."""
    if not data.startswith(_FLAC_MAGIC):
        raise ValueError("not a FLAC file")

    offset = len(_FLAC_MAGIC)
    blocks: list[tuple[int, bytes]] = []
    while True:
        if offset + 4 > len(data):
            raise ValueError("truncated FLAC metadata")

        header = data[offset]
        block_type = header & 0x7F
        block_length = int.from_bytes(data[offset + 1 : offset + 4], "big")
        offset += 4
        end = offset + block_length
        if end > len(data):
            raise ValueError("truncated FLAC metadata")

        blocks.append((block_type, data[offset:end]))
        offset = end
        if header & 0x80:
            return blocks, offset


def _encode_flac_metadata_blocks(blocks: list[tuple[int, bytes]]) -> bytes:
    """Encode FLAC metadata blocks with the final-block flag repaired."""
    encoded = bytearray(_FLAC_MAGIC)
    for index, (block_type, payload) in enumerate(blocks):
        if len(payload) > _MAX_FLAC_METADATA_BLOCK_SIZE:
            raise ValueError("FLAC metadata block is too large")

        final_flag = 0x80 if index == len(blocks) - 1 else 0
        encoded.append(final_flag | block_type)
        encoded.extend(len(payload).to_bytes(3, "big"))
        encoded.extend(payload)

    return bytes(encoded)


def _stringify_flac_metadata(value: JSONSerializable) -> str:
    """Convert an arbitrary metadata value into a Vorbis comment value."""
    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_flac_metadata(path: str, tags: JSON) -> None:
    """Write arbitrary valid FLAC Vorbis comment tags to ``path``."""
    valid_tags = {
        key: _stringify_flac_metadata(value)
        for key, value in tags.items()
        if _valid_vorbis_comment_key(key)
    }
    if not valid_tags:
        return

    path_obj = pathlib.Path(path)
    data = path_obj.read_bytes()
    blocks, audio_offset = _flac_metadata_blocks(data)
    audio_data = data[audio_offset:]

    comment_index: Optional[int] = None
    vendor = f"Celune {__version__}".encode("utf-8")
    comments: list[tuple[str, str]] = []
    for index, (block_type, payload) in enumerate(blocks):
        if block_type == _FLAC_VORBIS_COMMENT_BLOCK:
            comment_index = index
            vendor, comments = _parse_vorbis_comment_block(payload)
            break

    replaced_keys = {key.casefold() for key in valid_tags}
    comments = [
        (key, value) for key, value in comments if key.casefold() not in replaced_keys
    ]
    comments.extend(valid_tags.items())
    vorbis_payload = _encode_vorbis_comment_block(vendor, comments)

    if comment_index is None:
        insert_index = 1 if blocks and blocks[0][0] == _FLAC_STREAMINFO_BLOCK else 0
        blocks.insert(insert_index, (_FLAC_VORBIS_COMMENT_BLOCK, vorbis_payload))
    else:
        blocks[comment_index] = (_FLAC_VORBIS_COMMENT_BLOCK, vorbis_payload)

    path_obj.write_bytes(_encode_flac_metadata_blocks(blocks) + audio_data)


def _write_celune_flac(
    engine: "Celune",
    path: str,
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    subtype: str,
    metadata: JSON,
) -> None:
    """Write a FLAC file with Celune metadata in Vorbis comments."""
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    with sf.SoundFile(
        path,
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        format="FLAC",
        subtype=subtype,
    ) as audio_file:
        audio_file.write(audio)

    created_at = metadata.get(
        "created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    display_text = metadata.get("display_text")

    if not isinstance(display_text, str):
        display_text = f"Celune speech from {created_at}"

    prompt = display_text.split()
    words = " ".join(prompt[:5])
    if len(prompt) > 5:
        words += "..."

    tags: JSON = {
        "encoder": f"Celune {__version__}",
        "artist": engine.current_character or "Celune",
        "album": f"Celune via {engine.backend.name}",
        "title": words,
        "comment": encoded,
        "created_at": created_at,
        "date": datetime.datetime.now(datetime.timezone.utc).year,
    }
    _write_flac_metadata(path, tags)


def clear_queue(q: queue.Queue) -> None:
    """Drain all pending items from a queue.

    Args:
        q: The queue to empty.

    Returns:
        None: This method removes all currently pending items.
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def log_first_playback(engine: "Celune", timing: JSON) -> None:
    """Log time to first playback for a queued speech timing object.

    Args:
        engine: The instance of Celune to log back into.
        timing: The JSON-formatted timing data.

    Returns:
        None: This function logs timing data to Celune's configured logger.
    """
    start_time = getattr(timing, "start_time", None)
    if not isinstance(start_time, float):
        return

    mark_first_playback = getattr(timing, "mark_first_playback", None)
    if callable(mark_first_playback):
        mark_first_playback()
    elif getattr(timing, "first_playback_time", None) is None:
        return

    ttfp_seconds = getattr(timing, "ttfp_seconds", None)
    if callable(ttfp_seconds):
        elapsed = ttfp_seconds()
        if not isinstance(elapsed, float):
            return
    else:
        elapsed = time.monotonic() - start_time

    engine.log(f"TTFP: {format_number(elapsed, 2)} seconds")


def close_stream(engine: "Celune", abort: bool = False) -> None:
    """Close the current audio stream if one exists.

    Args:
        engine: The Celune engine that owns the audio stream.
        abort: Whether to abort immediately instead of stopping gracefully.

    Returns:
        None: This method closes the active output stream and clears stream state.
    """
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
    """Forcefully stop Celune from speaking.

    Args:
        engine: The Celune engine whose queues and playback should be interrupted.

    Returns:
        bool: ``True`` when an active utterance was stopped, otherwise ``False``.
    """
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
        engine.kept_sfx_audio = None
        engine.audio_queue.put(engine.force_stop_marker)

    return True


def acquire_pipeline(engine: "Celune", action: str) -> bool:
    """Atomically claim Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.
        action: A short label describing the action requesting the lock.

    Returns:
        bool: ``True`` when the pipeline was claimed, otherwise ``False``.
    """
    with engine.say_lock:
        engine.log_dev(f"[LOCK] acquire requested by {action}, locked={engine.locked}")
        if engine.locked:
            engine.log(f"Tried to {action} while Celune was busy.", "warning")
            engine.error_callback("Celune is currently busy")
            return False

        engine.locked = True
        if action != "play readiness signal":
            engine._ready_announced = False
        engine.playback_done.clear()
        engine.log_dev(f"[LOCK] acquired by {action}")
        return True


def release_pipeline(engine: "Celune") -> None:
    """Release Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.

    Returns:
        None: This method clears the busy state and marks playback as done.
    """
    with engine.say_lock:
        engine.locked = False
        engine.playback_done.set()
        engine.cur_state = "idle"
        engine.log_dev("[LOCK] released")


def say(
    engine: "Celune",
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
    return queue_speech(
        engine, text, save=save, stream_queue=None, display_text=display_text
    )


def queue_speech(
    engine: "Celune",
    text: str,
    save: bool = True,
    stream_queue: Optional[queue.Queue] = None,
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
        bool: ``True`` when the text was queued successfully, otherwise
            ``False``.
    """
    if engine.is_in_tutorial:
        engine.log("Speech input is disabled during the tutorial.", "warning")
        return False

    if not engine.model_ready.is_set():
        engine.status_callback("Waiting for model")
        engine.progress_callback(None, None)
        engine.log("Speak request is waiting for model reload to finish.", "info")

    engine.model_ready.wait()

    if not engine.loaded:
        engine.log("Model became unavailable before speaking.", "warning")
        engine.error_callback("Celune is not currently ready")
        engine.progress_callback(0, 1)
        return False

    language_meta = detect_language(text, list(engine.backend.supported_languages))
    if not language_meta["supported"]:
        # "zh-cn" has to be clipped to just "zh" to be a valid language code
        try:
            language = Lang(language_meta["language"][:2]).name
        except (InvalidLanguageValue, DeprecatedLanguageValue):
            language = language_meta["language"]

        engine.log(
            f"Received unsupported input in the following language: {language}",
            "warning",
        )
        engine.log("Celune may not say the input properly.", "warning")

    if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
        "1",
        "true",
        "on",
        "yes",
        "enabled",
    }:
        engine.log("We are about to do a funny!")
        text = rng_replace(text, targets=["celune"], replacements=["celine"])

    if not acquire_pipeline(engine, "speak"):
        engine.progress_callback(0, 1)
        return False

    try:
        if not engine.loaded:
            engine.log("Model became unavailable before queueing speech.", "warning")
            engine.error_callback("Celune is not currently ready")
            release_pipeline(engine)
            engine.progress_callback(0, 1)
            return False

        engine.cur_state = "generating"
        engine.text_queue.put(
            SpeechRequest(
                text,
                display_text=display_text if display_text is not None else text,
                save=save,
                stream_queue=stream_queue,
                normalize=engine.use_normalization,
            )
        )
        engine.status_callback("Generating")
        engine.progress_callback(None, None)
        return True
    except Exception:
        release_pipeline(engine)
        raise


def queue_sfx_audio(
    engine: "Celune",
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    label: str,
    keep: bool = False,
) -> bool:
    """Queue decoded SFX audio through Celune's playback pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.
        label: Human-readable label for logs and status.
        keep: Whether to prepend this SFX to the next saved utterance.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup
            fails.
    """
    if not acquire_pipeline(engine, "play"):
        return False

    try:
        audio = np.asarray(audio, dtype=np.float32)
        audio_len = len(audio) / sample_rate
        engine.log(
            f"Sample rate: {sample_rate} Hz, length: {format_number(audio_len, 2)} seconds"
        )

        audio = _resample_audio(audio, sample_rate)
        if keep:
            engine.kept_sfx_audio = audio.copy()

        engine.cur_state = "speaking"
        # push the smallest possible chunks for responsive stopping
        for chunk in _split(audio, BASE_SR, 1):
            engine.audio_queue.put((chunk, BASE_SR, None))
        engine.audio_queue.put(engine.utterance_done)

        engine.status_callback(f"Playing {label}")
        return True
    except Exception:
        release_pipeline(engine)
        raise


def play(engine: "Celune", sound_path: str, keep: bool = False) -> bool:
    """Play a sound via Celune's pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        sound_path: The path to the audio file to play.
        keep: Whether to prepend this SFX to the next saved utterance.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup
            fails.
    """
    if not os.path.exists(sound_path):
        engine.log(f"Celune cannot find {sound_path}.", "warning")
        return False

    audio, sr = sf.read(sound_path, dtype="float32")
    return queue_sfx_audio(
        engine, np.asarray(audio, dtype=np.float32), sr, sound_path, keep
    )


def close(engine: "Celune") -> None:
    """Shut off Celune and exit.

    Args:
        engine: The Celune engine to shut down.

    Returns:
        None: This method stops worker threads, closes audio, and fades out RGB.
    """
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
    sentence_checker = re.compile(r"\S.*?(?:[.!?]+[\"')\]]*(?=\s+|$)|$)", re.S)

    # detected quoted text with a boundary
    quote_checker = re.compile(r'"[^"]*[.!?]"')

    if len(text) <= max_length and not quote_checker.search(text):
        # input is short, return as is
        return [text]

    def split_long_unit(value: str) -> list[str]:
        """Split a sentence-like unit that is too long for one chunk."""
        pieces = [piece.strip() for piece in value.splitlines() if piece.strip()]
        if not pieces:
            pieces = value.split()

        chunks = []
        current = ""

        for piece in pieces:
            if len(piece) > max_length:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(split_words(piece))
                continue

            if current and len(current) + 1 + len(piece) > max_length:
                chunks.append(current)
                current = piece
            elif current and len(current) >= chunk_length:
                chunks.append(current)
                current = piece
            elif current:
                current = f"{current} {piece}"
            else:
                current = piece

        if current:
            chunks.append(current)

        return chunks

    def split_words(value: str) -> list[str]:
        """Split text on word boundaries when no stronger boundary exists."""
        chunks = []
        current = ""

        for word in value.split():
            if current and len(current) + 1 + len(word) > max_length:
                chunks.append(current)
                current = word
            elif current:
                current = f"{current} {word}"
            else:
                current = word

        if current:
            chunks.append(current)

        return chunks

    def split_sentences(value: str) -> list[str]:
        """Split a text fragment into sentence-like units.

        Args:
            value: Text fragment to split.

        Returns:
            list[str]: Sentence-like units with surrounding whitespace removed.
        """

        # match is a keyword, but it's allowed as an identifier
        units = []
        for match in sentence_checker.finditer(value):
            unit = match.group(0).strip()
            if len(unit) > max_length:
                units.extend(split_long_unit(unit))
            elif unit:
                units.append(unit)
        return units

    def split_units(value: str) -> list[str]:
        """Split text into sentence units while isolating quoted sentences.

        Args:
            value: Text to split.

        Returns:
            list[str]: Sentence units and complete quoted sentence units.
        """
        units = []
        start = 0

        for match in quote_checker.finditer(value):
            units.extend(split_sentences(value[start : match.start()]))
            units.append(match.group(0).strip())
            start = match.end()

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

        if current and len(current) + 1 + len(u) > max_length:
            chunks.append(current)
            current = u
        elif current and len(current) >= chunk_length:
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


def play_readiness_signal(engine: "Celune") -> bool:
    """Queue a readiness signal to be played.

    Args:
        engine: The instance of Celune to do this with.

    Returns:
        bool: Whether the readiness signal was processed successfully.
    """
    if acquire_pipeline(engine, "play readiness signal"):
        engine.cur_state = "speaking"
        engine.audio_queue.put((readiness_signal(), BASE_SR, None))
        engine.audio_queue.put(engine.utterance_done)
        return True
    return False


def generation_worker(engine: "Celune") -> None:
    """Generate audio tokens and send them to the audio pipeline.

    Args:
        engine: The Celune engine whose generation queue should be processed.

    Returns:
        None: This worker loop runs until it receives the shutdown sentinel.

    Raises:
        NotAvailableError: The speech model is unavailable during generation.
    """
    while True:
        item = engine.text_queue.get()
        engine.regenerate = False

        if item is engine.sentinel:
            engine.audio_queue.put(engine.sentinel)
            break

        text = item.text
        display_text = item.display_text
        save_output = item.save
        stream_queue = item.stream_queue
        kept_sfx_audio = engine.kept_sfx_audio
        engine.kept_sfx_audio = None

        if engine.exit_requested:
            if stream_queue is not None:
                stream_queue.put(NotAvailableError("stream queue interrupted"))
                stream_queue.put(None)
            release_pipeline(engine)
            continue

        while True:
            try:
                engine.model_ready.wait()

                if not engine.loaded:
                    engine.log(
                        "Skipping generation because model is not ready.", "warning"
                    )
                    engine.locked = False
                    if stream_queue is not None:
                        stream_queue.put(NotAvailableError("model is not ready"))
                        stream_queue.put(None)
                    release_pipeline(engine)
                    break

                start_time = time.monotonic()
                engine.log(f"[GEN] {display_text}")
                speech_len = 0.0
                buffered_speech_len = 0.0
                speech_timing = SpeechTiming(start_time)
                pushed_audio = False

                # these generation parameters are fixed and do not change
                generation_params = {
                    "temperature": 0.15,
                    "top_k": 20,
                    "top_p": 0.7,
                    "repetition_penalty": 1.1,
                }

                chunks = split_text(engine, text)
                if not chunks:
                    engine.progress_callback(0, 1)
                    engine.error_callback("Nothing to say")
                    release_pipeline(engine)
                    if stream_queue is not None:
                        stream_queue.put(NotAvailableError("nothing to say"))
                        stream_queue.put(None)
                    break

                buffer: list[npt.NDArray[np.float32]] = []
                full_audio: list[npt.NDArray[np.float32]] = []
                generated_text_parts: list[str] = []

                for chunk_index, chunk_text in enumerate(chunks):
                    if engine.exit_requested:
                        break

                    if engine.utterance_force_stop.is_set():
                        break

                    if item.normalize:
                        engine.status_callback("Normalizing")
                        engine.progress_callback(None, None)
                        normalized = engine.normalize(chunk_text)
                        if normalized is not None:
                            chunk_text = normalized

                    generated_text_parts.append(chunk_text)
                    is_first_chunk = chunk_index == 0
                    progress_total = engine.backend.generation_progress_total(
                        chunk_text
                    )
                    generated_steps = 0
                    engine.progress_callback(0, progress_total or 1)

                    with engine.model_lock:
                        if engine.model is None:
                            raise NotAvailableError(
                                "cannot generate without a model reference"
                            )

                        for (
                            audio_chunk,
                            sr,  # 24 kHz if Qwen3, 48 kHz if VoxCPM2
                            timing,
                        ) in engine.backend.generate_stream(  # some args will be discarded as needed
                            engine.model,
                            text=chunk_text,
                            language=engine.language,
                            chunk_size=engine.chunk_size,
                            instruct=engine.voice_prompt,
                            voice=engine.current_voice,
                            temperature=generation_params["temperature"],
                            top_k=generation_params["top_k"],
                            top_p=generation_params["top_p"],
                            repetition_penalty=generation_params["repetition_penalty"],
                        ):
                            if engine.exit_requested:
                                break

                            if engine.utterance_force_stop.is_set():
                                break

                            if progress_total is not None:
                                generated_steps += (
                                    engine.backend.generation_progress_steps(timing)
                                )
                                engine.progress_callback(
                                    min(generated_steps, progress_total),
                                    progress_total,
                                )

                            speech_timing.mark_first_chunk()

                            if isinstance(audio_chunk, torch.Tensor):
                                audio_chunk = audio_chunk.cpu().numpy()

                            audio_chunk = _to_48khz(
                                np.asarray(audio_chunk, dtype=np.float32), sr
                            )

                            if engine.speed != 1.0 and engine.can_use_rubberband:
                                try:
                                    audio_chunk = rb.time_stretch(
                                        audio_chunk, BASE_SR, engine.speed
                                    )
                                except RuntimeError:
                                    engine.log(
                                        "Rubber Band is unavailable, speed controls disabled.",
                                        "warning",
                                    )
                                    engine.can_use_rubberband = False
                                else:
                                    audio_chunk = np.asarray(
                                        audio_chunk, dtype=np.float32
                                    )
                            if engine.reverb.strength > 0.0:
                                audio_chunk = engine.reverb.process(
                                    audio_chunk, BASE_SR
                                )
                                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                            if is_first_chunk:
                                audio_chunk = _soften(audio_chunk, BASE_SR, end=False)
                                is_first_chunk = False

                            if engine.exit_requested:
                                break

                            buffer.append(audio_chunk)
                            if save_output:
                                full_audio.append(audio_chunk)
                            chunk_dur = len(audio_chunk) / BASE_SR
                            speech_len += chunk_dur
                            buffered_speech_len += chunk_dur

                            # buffering helps Celune speak smoothly when performance is bad
                            if buffered_speech_len >= 10.0:
                                queued_audio = np.concatenate(buffer)
                                engine.audio_queue.put(
                                    (
                                        queued_audio,
                                        BASE_SR,
                                        speech_timing if not pushed_audio else None,
                                    )
                                )
                                if stream_queue is not None:
                                    stream_queue.put(queued_audio.copy())
                                buffer = []
                                buffered_speech_len = 0.0

                                if not pushed_audio:
                                    pushed_audio = True
                                    engine.status_callback("Speaking")
                                    engine.cur_state = "speaking"
                                    engine.queue_avail_callback()

                        if progress_total is None:
                            engine.progress_callback(1, 1)

                if generated_text_parts:
                    text = " ".join(generated_text_parts)

                generation_time = time.monotonic() - start_time

                if engine.exit_requested:
                    if stream_queue is not None:
                        stream_queue.put(None)
                    release_pipeline(engine)
                    break

                if engine.utterance_force_stop.is_set():
                    if stream_queue is not None:
                        stream_queue.put(None)
                    engine.reverb.reset()
                    break

                engine.log(
                    f"[GEN] {format_number(speech_len, 2)} seconds, "
                    f"took {format_number(generation_time, 2)} seconds"
                )
                engine.log(f"Speed: x{format_number(speech_len / generation_time, 2)}")
                engine.log(f"TTFC: {format_number(speech_timing.ttfc_ms(), 1)} ms")

                if buffer:
                    queued_audio = np.concatenate(buffer)
                    engine.audio_queue.put(
                        (
                            queued_audio,
                            BASE_SR,
                            speech_timing if not pushed_audio else None,
                        )
                    )
                    if stream_queue is not None:
                        stream_queue.put(queued_audio.copy())
                    if not pushed_audio:
                        # noinspection PyUnusedLocal
                        pushed_audio = True
                        engine.status_callback("Speaking")
                        engine.cur_state = "speaking"
                        engine.queue_avail_callback()

                engine.log("[GEN] done")

                saved_path = None
                analysis_audio = None
                if not engine.exit_requested:
                    if engine.reverb.strength > 0.0:
                        tail = engine.reverb.flush()
                        if len(tail) > 0:
                            engine.audio_queue.put((tail, BASE_SR, None))
                            if stream_queue is not None:
                                stream_queue.put(tail.copy())
                            buffer.append(tail)
                            if save_output:
                                full_audio.append(tail)

                    engine.reverb.reset()
                    is_silent, silence_tier = is_silent_utterance(
                        np.concatenate(full_audio)
                    )

                    if is_silent and silence_tier == 2:
                        engine.regenerate = True
                        # push recently processed item back so Celune can process it again
                        engine.text_queue.put(item)
                        engine.log(
                            "Previous utterance was silent, regenerating...", "warning"
                        )
                        continue
                    if is_silent and silence_tier == 1:
                        engine.log(
                            "This utterance may be unexpectedly silent.", "warning"
                        )

                    if save_output and full_audio:
                        wav = np.concatenate(full_audio)
                        analysis_audio = wav.copy()
                        if kept_sfx_audio is not None:
                            wav = np.concatenate((kept_sfx_audio, wav))
                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                        # get up to first three words of input and sanitize for use in a file name
                        first_words = "_".join(text.split()[:3]).lower()
                        first_words = re.sub(r"[^a-zA-Z0-9_]", "", first_words)

                        if not os.path.exists("outputs"):
                            engine.log("Outputs path not found, creating...", "warning")
                            try:
                                os.mkdir("outputs")
                            except OSError as e:
                                engine.log(
                                    "Cannot create outputs directory, not saving WAV file: "
                                    f"{format_error(e, engine.dev)}",
                                    "warning",
                                )

                        if os.path.exists("outputs"):
                            saved_path = (
                                f"outputs/celune_speech_{timestamp}_{first_words}.flac"
                            )
                            sample_rate = BASE_SR
                            subtype = "PCM_24"
                            metadata = _celune_metadata_payload(
                                engine,
                                text=text,
                                display_text=display_text,
                                generation_params=generation_params,
                                sample_rate=sample_rate,
                                subtype=subtype,
                                included_kept_sfx=kept_sfx_audio is not None,
                            )
                            try:
                                _write_celune_flac(
                                    engine,
                                    saved_path,
                                    wav,
                                    sample_rate,
                                    subtype=subtype,
                                    metadata=metadata,
                                )
                            except Exception as e:
                                engine.log(
                                    "Could not save FLAC output: "
                                    f"{format_error(e, engine.dev)}",
                                    "warning",
                                )
                                saved_path = None

                    engine.recently_saved = saved_path
                    engine.audio_queue.put(
                        SpeechDone(
                            saved_path=saved_path,
                            analysis_audio=analysis_audio,
                        )
                    )
                    if stream_queue is not None:
                        stream_queue.put(None)
                break
            except Exception as e:
                if engine.exit_requested:
                    release_pipeline(engine)
                    break

                engine.log(f"[GEN ERROR] {format_error(e, engine.dev)}", "error")
                if stream_queue is not None:
                    stream_queue.put(e)
                    stream_queue.put(None)
                engine.cur_state = "error"
                engine.locked = False
                engine.playback_done.set()
                engine.progress_callback(0, 1)
                engine.error_callback("Celune could not generate the input")
                break


def playback_worker(engine: "Celune") -> None:
    """Receive audio chunks and play them.

    Args:
        engine: The Celune engine whose audio queue should be played back.

    Returns:
        None: This worker loop runs until playback is shut down.

    Raises:
        NotAvailableError: The audio stream is unavailable during playback.
    """
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

        if isinstance(item, SpeechDone) or item is engine.utterance_done:
            saved_path = item.saved_path if isinstance(item, SpeechDone) else None
            analysis_audio = (
                item.analysis_audio if isinstance(item, SpeechDone) else None
            )
            engine.playback_done.set()

            more_pending = (not engine.audio_queue.empty()) or (
                not engine.text_queue.empty()
            )

            if more_pending:
                silence = np.zeros((BASE_SR, 2), dtype=np.float32)
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
                    # queueing new speech during analysis may net you a reduced performance
                    if (
                        engine.dev
                        and saved_path is not None
                        and analysis_audio is not None
                    ):
                        engine.log_dev("Analyzing...")
                        saved = pathlib.Path(saved_path)
                        run_async(
                            analyze_voice_audio,
                            analysis_audio,
                            BASE_SR,
                            saved.name,
                            saved.parent,
                            saved.stem,
                            engine.current_voice,
                        )

                    if not getattr(engine, "_ready_announced", False):
                        engine.log("Ready to speak.")
                        engine._ready_announced = True

                if torch.cuda.is_available():
                    avail, total = tuple(
                        v / 1024**3 for v in torch.cuda.mem_get_info(0)
                    )
                    if avail <= total * 0.1:
                        engine.log(
                            "Celune is running out of VRAM. Check the bottom right of Celune's window to learn more.",
                            "warning",
                        )
                        engine.log(
                            "Please close any memory-resident applications to improve performance.",
                            "warning",
                        )
            continue

        audio_chunk, sr, timing = item

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
                engine.log_dev(f"[PLAY] started stream at {sr} Hz")
            except sd.PortAudioError:
                if not engine.audio_unavailable:
                    engine.log("Celune could not initialize the audio stream.", "error")
                    engine.log("No suitable audio device is available.", "error")
                    engine.error_callback("No suitable audio devices")
                engine._audio_unavailable = True

        if engine.exit_requested:
            continue

        try:
            stream = engine.stream
            if stream is None:
                raise NotAvailableError("audio stream is not available")
            log_first_playback(engine, timing)
            engine.glow.schedule(audio_chunk)
            stream.write(audio_chunk)
        except Exception as e:
            engine.log(f"[PLAY ERROR] {format_error(e, engine.dev)}", "error")
            engine.error_callback("Playback error")
            close_stream(engine, abort=True)
            engine._stream = None
            engine._current_sr = None
            continue
