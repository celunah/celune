# SPDX-License-Identifier: Apache-2.0
"""Metadata helpers for Celune-generated audio files."""

from __future__ import annotations

import json
import pathlib
import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, cast

import soundfile as sf

from . import __version__
from .constants import APP_NAME
from .typing.aliases import AudioChunk
from .typing.common import JSON, JSONSerializable


if TYPE_CHECKING:
    from .celune import Celune

_FLAC_MAGIC = b"fLaC"
_FLAC_STREAMINFO_BLOCK = 0
_FLAC_VORBIS_COMMENT_BLOCK = 4
_MAX_FLAC_METADATA_BLOCK_SIZE = 0xFFFFFF


def _json_value(value: JSONSerializable) -> JSONSerializable:
    """Return a value only when it is already JSON-compatible."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list) and all(_is_json_value(item) for item in value):
        return cast(list[JSONSerializable], value)
    if isinstance(value, dict) and all(
        isinstance(key, str) and _is_json_value(item) for key, item in value.items()
    ):
        return cast(dict[str, JSONSerializable], value)
    return None


def _is_json_value(value: JSONSerializable) -> bool:
    """Return whether a value can be stored in Celune JSON metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _celune_metadata_payload(
    engine: Celune,
    *,
    text: str,
    display_text: str,
    generation_params: Mapping[str, JSONSerializable],
    sample_rate: int,
    subtype: str,
    included_kept_sfx: bool,
) -> JSON:
    """Build the Celune generation metadata payload."""
    return {
        "format": "CEMETA",
        "format_version": 1,
        "celune_version": __version__,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "text": text,
        "display_text": display_text,
        "backend": _json_value(getattr(engine, "tts_backend", None)),
        "qwen3_x_vector_only": _json_value(
            getattr(engine.backend, "x_vector_only", None)
        ),
        "model_name": _json_value(getattr(engine, "model_name", "")),
        "voice": _json_value(getattr(engine, "current_voice", None)),
        "voice_prompt": _json_value(getattr(engine, "voice_prompt", None)),
        "language": _json_value(getattr(engine, "language", None)),
        "chunk_size": _json_value(getattr(engine, "chunk_size", None)),
        "speed": _json_value(getattr(engine, "speed", None)),
        "reverb_strength": _json_value(getattr(engine.reverb, "strength", None)),
        "use_normalizer": _json_value(getattr(engine, "use_normalization", None)),
        "sample_rate": sample_rate,
        "subtype": subtype,
        "included_kept_sfx": included_kept_sfx,
        "generation": dict(generation_params),
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
        raw_comment = f"{key}={value}".encode()
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
    vendor = f"{APP_NAME} {__version__}".encode()
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
    engine: Celune,
    path: str,
    audio: AudioChunk,
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
        "created_at", datetime.datetime.now(datetime.UTC).isoformat()
    )
    display_text = metadata.get("display_text")

    if not isinstance(display_text, str):
        display_text = f"{APP_NAME} speech from {created_at}"

    prompt = display_text.split()
    words = " ".join(prompt[:5])
    if len(prompt) > 5:
        words += "..."

    tags: JSON = {
        "encoder": f"{APP_NAME} {__version__}",
        "artist": engine.current_character or APP_NAME,
        "album": f"{APP_NAME} via {engine.backend.name}",
        "title": words,
        "comment": encoded,
        "created_at": created_at,
        "date": datetime.datetime.now(datetime.UTC).year,
    }
    _write_flac_metadata(path, tags)
