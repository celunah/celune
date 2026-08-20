# SPDX-License-Identifier: Apache-2.0
"""Private CEDTS framing and typed payload transport for backend workers."""

import json
import struct
from uuid import uuid4
from pathlib import Path
from math import isfinite
from dataclasses import dataclass
from typing import IO, Optional, cast
from collections.abc import Mapping, Sequence

import numpy as np

from ..i18n import string
from ..typing.common import JSONSerializable
from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from ..typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerControlMessage,
    WorkerPayloadDescriptor,
)

__all__ = [
    "CEDTS_VERSION",
    "CONTROL_PACKET_KINDS",
    "CORE_CAPABILITIES",
    "DEFAULT_CEDTS_LIMITS",
    "SUPPORTED_OPERATIONS",
    "WORKER_CAPABILITIES",
    "CEDTSLimits",
    "WorkerPayload",
    "WorkerProtocolError",
    "build_packet",
    "decode_message",
    "encode_message",
    "limits_from_capabilities",
    "receive_message",
    "receive_payloads",
    "send_message",
    "send_payloads",
    "validate_payload_descriptors",
]

_FRAME_HEADER = struct.Struct("!I")
_BINARY_HEADER = struct.Struct("!HQ")
_MAX_CONTROL_FRAME_SIZE = 1024 * 1024
_MAX_BINARY_FRAME_SIZE = 8 * 1024 * 1024
_MAX_AGGREGATE_PAYLOAD_SIZE = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_COLLECTION_ENTRIES = 1024
_MAX_STRING_LENGTH = 1024 * 1024
_MAX_PAYLOAD_ID_LENGTH = 256
_MAX_SHAPE_DIMENSIONS = 8
_MAX_SHAPE_DIMENSION = 16 * 1024 * 1024
_MAX_MESSAGE_ID_LENGTH = 256
_MAX_OPERATION_LENGTH = 128
_MAX_TYPED_VALUE_FIELDS = 16
_MAX_BACKEND_ARGUMENT_FIELDS = 128
_MAX_ARGUMENT_NAME_LENGTH = 128
_LIMIT_CAPABILITY_FIELDS = (
    "max_control_frame_size",
    "max_binary_frame_size",
    "max_aggregate_payload_size",
    "max_payload_descriptors",
    "max_json_depth",
    "max_string_length",
    "max_collection_entries",
)
_VALID_MEDIA_TYPES = frozenset(
    {
        "audio/pcm_f32le",
        "audio/pcm_s16le",
        "application/octet-stream",
        "application/x-tensor",
        "image/jpeg",
        "image/png",
    }
)
_AUDIO_MEDIA_TYPES = frozenset(
    {
        "audio/pcm_f32le",
        "audio/pcm_s16le",
    }
)
_TENSOR_MEDIA_TYPES = frozenset({"application/x-tensor"})
_VALID_DTYPES = frozenset({"bool", "float32", "float64", "int8", "int16", "uint8"})
CEDTS_VERSION = 1
CONTROL_PACKET_KINDS = frozenset(
    {
        "hello",
        "hello_ack",
        "ready",
        "request",
        "response",
        "event",
        "callback",
        "progress",
        "cancel",
        "cancel_ack",
        "error",
        "ping",
        "pong",
        "shutdown",
        "shutdown_ack",
    }
)
_SUPPORTED_OPERATIONS = (
    "describe",
    "model_is_available_locally",
    "preload_models",
    "load_model",
    "unload_model",
    "convert",
    "call",
    "generate_stream",
)
SUPPORTED_OPERATIONS = frozenset(_SUPPORTED_OPERATIONS)
_PROTOCOL_OPERATIONS = frozenset(
    {
        "cancel",
        "fatal",
        "handshake",
        "protocol",
        "ready",
        "shutdown",
    }
)
_VALID_OPERATIONS = SUPPORTED_OPERATIONS | _PROTOCOL_OPERATIONS
_EVENT_OPERATIONS = frozenset(
    {
        "fatal",
        "state_changed",
        "loading",
        "ready",
        "processing",
        "streaming",
        "paused",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "shutdown_requested",
    }
)
_STATE_VALUES = frozenset(
    {
        "loading",
        "ready",
        "processing",
        "streaming",
        "paused",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "shutdown_requested",
    }
)
_PACKET_FIELDS = frozenset(
    {
        "cedts_version",
        "data",
        "kind",
        "message_id",
        "operation",
        "payloads",
        "reply_to",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "byte_length",
        "channels",
        "dtype",
        "id",
        "media_type",
        "sample_rate",
        "shape",
    }
)


@dataclass(frozen=True)
class CEDTSLimits:
    """Effective per-connection bounds for CEDTS control and binary frames."""

    max_control_frame_size: int = _MAX_CONTROL_FRAME_SIZE
    max_binary_frame_size: int = _MAX_BINARY_FRAME_SIZE
    max_aggregate_payload_size: int = _MAX_AGGREGATE_PAYLOAD_SIZE
    max_payload_descriptors: int = _MAX_COLLECTION_ENTRIES
    max_json_depth: int = _MAX_JSON_DEPTH
    max_string_length: int = _MAX_STRING_LENGTH
    max_collection_entries: int = _MAX_COLLECTION_ENTRIES


DEFAULT_CEDTS_LIMITS = CEDTSLimits()
CORE_CAPABILITIES: dict[str, JSONSerializable] = {
    "supported_operations": cast(JSONSerializable, list(_SUPPORTED_OPERATIONS)),
    "supported_media_types": cast(JSONSerializable, sorted(_VALID_MEDIA_TYPES)),
    "max_control_frame_size": _MAX_CONTROL_FRAME_SIZE,
    "max_binary_frame_size": _MAX_BINARY_FRAME_SIZE,
    "max_aggregate_payload_size": _MAX_AGGREGATE_PAYLOAD_SIZE,
    "max_payload_descriptors": _MAX_COLLECTION_ENTRIES,
    "max_json_depth": _MAX_JSON_DEPTH,
    "max_string_length": _MAX_STRING_LENGTH,
    "max_collection_entries": _MAX_COLLECTION_ENTRIES,
    "streaming": True,
    "cancellation": True,
    "callback": False,
}
WORKER_CAPABILITIES: dict[str, JSONSerializable] = dict(CORE_CAPABILITIES)


def limits_from_capabilities(
    capabilities: Mapping[str, object],
    *,
    fallback: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> CEDTSLimits:
    """Build effective limits from peer capabilities and safe local defaults."""
    values = {
        field_name: getattr(fallback, field_name)
        for field_name in _LIMIT_CAPABILITY_FIELDS
    }
    for field_name in _LIMIT_CAPABILITY_FIELDS:
        value = capabilities.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values[field_name] = min(values[field_name], value)
    return CEDTSLimits(**values)


class WorkerProtocolError(RuntimeError):
    """Raised when a backend worker sends invalid CEDTS data."""


def _worker_protocol_error(key: str, **kwargs: str) -> WorkerProtocolError:
    """Create a localized protocol validation error."""
    return WorkerProtocolError(string(f"backends.worker_protocol.{key}", **kwargs))


def build_packet(
    kind: str,
    operation: str,
    data: Optional[Mapping[str, WorkerValue]] = None,
    *,
    reply_to: Optional[str] = None,
    message_id: Optional[str] = None,
) -> WorkerMessage:
    """Build one CEDTS packet with a unique identity and correlation target."""
    if kind not in CONTROL_PACKET_KINDS:
        raise _worker_protocol_error("backend_worker_packet_kind_is_invalid")
    _validate_operation(operation)
    if message_id is not None and not isinstance(message_id, str):
        raise _worker_protocol_error("backend_worker_packet_message_id_is_invalid")
    if reply_to is not None and not isinstance(reply_to, str):
        raise _worker_protocol_error("backend_worker_packet_reply_target_is_invalid")
    return {
        "cedts_version": CEDTS_VERSION,
        "kind": kind,
        "message_id": message_id or uuid4().hex,
        "reply_to": reply_to,
        "operation": operation,
        "data": dict(data or {}),
    }


@dataclass(frozen=True)
class WorkerPayload:
    """One validated binary payload and its CEDTS descriptor."""

    descriptor: WorkerPayloadDescriptor
    data: bytes


class _PayloadBuilder:
    """Collect binary values while encoding one logical control message."""

    def __init__(self, limits: CEDTSLimits) -> None:
        self.limits = limits
        self.payloads: list[WorkerPayload] = []
        self.descriptor_count = 0
        self.aggregate_bytes = 0

    def preflight(
        self,
        byte_length: int,
        *,
        media_type: str,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> None:
        """Validate a payload before materializing or retaining its bytes."""
        descriptor = self._descriptor(
            byte_length,
            media_type=media_type,
            dtype=dtype,
            shape=shape,
            sample_rate=sample_rate,
            channels=channels,
        )
        if self.descriptor_count >= self.limits.max_payload_descriptors:
            raise _worker_protocol_error("too_many_binary_payload_descriptors")
        if self.aggregate_bytes + byte_length > self.limits.max_aggregate_payload_size:
            raise _worker_protocol_error("aggregate_binary_payload_is_too_large")
        frame_size = (
            _BINARY_HEADER.size + len(descriptor["id"].encode("utf-8")) + byte_length
        )
        if frame_size > self.limits.max_binary_frame_size:
            raise _worker_protocol_error("binary_payload_frame_is_too_large")
        validate_payload_descriptors([descriptor], limits=self.limits)

    def _descriptor(
        self,
        byte_length: int,
        *,
        media_type: str,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> WorkerPayloadDescriptor:
        """Build a descriptor using the next payload sequence number."""
        descriptor: WorkerPayloadDescriptor = {
            "id": f"payload-{self.descriptor_count + 1}",
            "media_type": media_type,
            "byte_length": byte_length,
        }
        if dtype is not None:
            descriptor["dtype"] = dtype
        if shape is not None:
            descriptor["shape"] = list(shape)
        if sample_rate is not None:
            descriptor["sample_rate"] = sample_rate
        if channels is not None:
            descriptor["channels"] = channels
        return descriptor

    def add(
        self,
        data: bytes,
        *,
        media_type: str,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> dict[str, str]:
        """Add one bounded payload and return its JSON reference."""
        byte_length = len(data)
        self.preflight(
            byte_length,
            media_type=media_type,
            dtype=dtype,
            shape=shape,
            sample_rate=sample_rate,
            channels=channels,
        )
        descriptor = self._descriptor(
            byte_length,
            media_type=media_type,
            dtype=dtype,
            shape=shape,
            sample_rate=sample_rate,
            channels=channels,
        )
        payload_id = descriptor["id"]
        self.payloads.append(WorkerPayload(descriptor, data))
        self.descriptor_count += 1
        self.aggregate_bytes += byte_length
        return {"__cedts_payload_id__": payload_id}


def encode_message(
    message: Mapping[str, WorkerValue],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> tuple[WorkerControlMessage, tuple[WorkerPayload, ...]]:
    """Encode backend values into JSON metadata and separate binary payloads."""
    builder = _PayloadBuilder(limits)
    encoded = {key: _encode_value(value, builder) for key, value in message.items()}
    if not isinstance(encoded, dict):
        raise _worker_protocol_error("backend_worker_control_message_is_not_an_object")
    control = cast(WorkerControlMessage, dict(encoded))
    control["payloads"] = cast(
        JSONSerializable, [payload.descriptor for payload in builder.payloads]
    )
    validate_payload_descriptors(
        cast(list[WorkerPayloadDescriptor], control["payloads"]), limits=limits
    )
    return control, tuple(builder.payloads)


def decode_message(
    message: WorkerControlMessage,
    payloads: Mapping[str, WorkerPayload],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> dict[str, WorkerValue]:
    """Decode validated payload references into backend-owned Python values."""
    try:
        _validate_decoded_payload_budget(message, payloads, limits=limits)
        return {
            key: _decode_value(value, payloads)
            for key, value in message.items()
            if key != "payloads"
        }
    except (
        AttributeError,
        IndexError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise _worker_protocol_error(
            "backend_worker_payload_metadata_is_invalid"
        ) from error


class _DecodedPayloadBudget:
    """Track typed payload copies before decoding a CEDTS message."""

    def __init__(self, payloads: Mapping[str, WorkerPayload], limits: CEDTSLimits):
        self.payloads = payloads
        self.limits = limits
        self.typed_payload_ids: set[str] = set()
        self.allocated_bytes = 0

    def reserve(self, value: object, *, audio: bool) -> None:
        """Reserve one unique typed payload's decoded allocation."""
        payload_id = _payload_id(value)
        if payload_id in self.typed_payload_ids:
            raise _worker_protocol_error("typed_payload_reference_is_duplicated")
        payload = self.payloads.get(payload_id)
        if payload is None:
            raise _worker_protocol_error("binary_value_has_no_payload_id")
        byte_length = payload.descriptor.get("byte_length")
        if not isinstance(byte_length, int) or byte_length < 0:
            raise _worker_protocol_error("binary_payload_length_is_invalid")
        if audio:
            dtype = payload.descriptor.get("dtype")
            if dtype == "int16":
                byte_length *= 2
            elif dtype != "float32":
                raise _worker_protocol_error("audio_payload_metadata_is_invalid")
        if self.allocated_bytes + byte_length > self.limits.max_aggregate_payload_size:
            raise _worker_protocol_error("aggregate_decoded_payload_is_too_large")
        self.typed_payload_ids.add(payload_id)
        self.allocated_bytes += byte_length


def _validate_decoded_payload_budget(
    value: object,
    payloads: Mapping[str, WorkerPayload],
    *,
    limits: CEDTSLimits,
) -> None:
    """Bound decoded copies and reject duplicate typed payload references."""
    budget = _DecodedPayloadBudget(payloads, limits)
    _scan_decoded_payload_value(value, budget)


def _scan_decoded_payload_value(value: object, budget: _DecodedPayloadBudget) -> None:
    """Scan nested JSON values for typed payload allocations."""
    if isinstance(value, dict):
        if "__cedts_payload_id__" in value:
            return
        value_type = value.get("__cedts_type__")
        if value_type == "numpy_array":
            budget.reserve(value.get("payload"), audio=False)
            return
        if value_type == "audio_output":
            budget.reserve(value.get("audio"), audio=True)
            return
        if value_type == "voice_conversion_request":
            budget.reserve(value.get("source_audio"), audio=True)
            return
        if value_type == "backend_generation":
            budget.reserve(value.get("audio"), audio=True)
            _scan_decoded_payload_value(value.get("metadata"), budget)
            return
        if value_type == "tuple":
            items = value.get("items")
            if isinstance(items, list):
                for item in items:
                    _scan_decoded_payload_value(item, budget)
            return
        for item in value.values():
            _scan_decoded_payload_value(item, budget)
        return
    if isinstance(value, list):
        for item in value:
            _scan_decoded_payload_value(item, budget)


def send_message(
    stream: IO[bytes],
    message: WorkerMessage,
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Write one length-prefixed CEDTS JSON control object."""
    try:
        if not isinstance(message, Mapping):
            raise TypeError("control message is not an object")
        json_message = dict(message)
        _validate_json_value(json_message, limits=limits)
        _validate_packet(json_message, limits=limits)
        payload = json.dumps(
            json_message,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise _worker_protocol_error(
            "backend_worker_message_is_not_valid_cedts_json"
        ) from error
    if len(payload) > limits.max_control_frame_size:
        raise _worker_protocol_error("backend_worker_control_frame_is_too_large")
    stream.write(_FRAME_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def receive_message(
    stream: IO[bytes],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> WorkerControlMessage:
    """Read one length-prefixed CEDTS JSON control object."""
    try:
        header = _read_exact(stream, _FRAME_HEADER.size)
        (payload_size,) = _FRAME_HEADER.unpack(header)
        if payload_size > limits.max_control_frame_size:
            raise _worker_protocol_error("backend_worker_control_frame_is_too_large")
        payload = _read_exact(stream, payload_size)
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _object_from_pairs(
                pairs, limits.max_collection_entries
            ),
            parse_constant=_reject_json_constant,
        )
        if not isinstance(decoded, dict):
            raise _worker_protocol_error(
                "backend_worker_control_frame_is_not_an_object"
            )
        _validate_json_value(decoded, limits=limits)
        _validate_packet(decoded, limits=limits)
        descriptors = decoded.get("payloads", [])
        if not isinstance(descriptors, list):
            raise _worker_protocol_error(
                "backend_worker_payload_descriptors_are_not_an_array"
            )
        validate_payload_descriptors(descriptors, limits=limits)
        return decoded
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise _worker_protocol_error(
            "backend_worker_control_frame_is_invalid"
        ) from error


def send_payloads(
    stream: IO[bytes],
    payloads: Sequence[WorkerPayload],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Write one logical message's payload frames and terminal boundary."""
    if len(payloads) > limits.max_payload_descriptors:
        raise _worker_protocol_error("too_many_binary_payloads")
    total = 0
    for payload in payloads:
        validate_payload_descriptors([payload.descriptor], limits=limits)
        expected = payload.descriptor["byte_length"]
        if len(payload.data) != expected:
            raise _worker_protocol_error(
                "binary_payload_length_does_not_match_its_descriptor"
            )
        total += len(payload.data)
        if total > limits.max_aggregate_payload_size:
            raise _worker_protocol_error("aggregate_binary_payload_is_too_large")
    for payload in payloads:
        payload_id = payload.descriptor["id"].encode("utf-8")
        frame_size = _BINARY_HEADER.size + len(payload_id) + len(payload.data)
        if frame_size > limits.max_binary_frame_size:
            raise _worker_protocol_error("binary_payload_frame_is_too_large")
        stream.write(_FRAME_HEADER.pack(frame_size))
        stream.write(_BINARY_HEADER.pack(len(payload_id), len(payload.data)))
        stream.write(payload_id)
        stream.write(payload.data)
        stream.flush()
    # A zero-sized frame is the mandatory boundary for this control packet.
    # It lets the receiver distinguish an intentional no-payload message from
    # an undeclared frame on the separate binary channel.
    stream.write(_FRAME_HEADER.pack(0))
    stream.flush()


def receive_payloads(
    stream: IO[bytes],
    descriptors: Sequence[WorkerPayloadDescriptor],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> dict[str, WorkerPayload]:
    """Read one logical message's payload frames and terminal boundary."""
    validate_payload_descriptors(descriptors, limits=limits)
    expected = {descriptor["id"]: descriptor for descriptor in descriptors}
    received: dict[str, WorkerPayload] = {}
    total = 0
    for _ in descriptors:
        frame_header = _read_exact(stream, _FRAME_HEADER.size)
        (frame_size,) = _FRAME_HEADER.unpack(frame_header)
        if (
            frame_size > limits.max_binary_frame_size
            or frame_size < _BINARY_HEADER.size
        ):
            raise _worker_protocol_error("binary_payload_frame_size_is_invalid")
        binary_header = _read_exact(stream, _BINARY_HEADER.size)
        id_size, payload_size = _BINARY_HEADER.unpack(binary_header)
        if id_size > _MAX_PAYLOAD_ID_LENGTH:
            raise _worker_protocol_error("binary_payload_id_is_too_long")
        if frame_size != _BINARY_HEADER.size + id_size + payload_size:
            raise _worker_protocol_error(
                "binary_frame_length_does_not_match_its_header"
            )
        try:
            payload_id = _read_exact(stream, id_size).decode("utf-8")
        except UnicodeError as error:
            raise _worker_protocol_error(
                "binary_payload_id_is_not_valid_utf_8"
            ) from error
        descriptor = expected.get(payload_id)
        if descriptor is None or payload_id in received:
            raise _worker_protocol_error(
                "binary_payload_id_is_unexpected_or_duplicated"
            )
        if payload_size != descriptor["byte_length"]:
            raise _worker_protocol_error(
                "binary_payload_length_does_not_match_its_descriptor"
            )
        total += payload_size
        if total > limits.max_aggregate_payload_size:
            raise _worker_protocol_error("aggregate_binary_payload_is_too_large")
        data = _read_exact(stream, payload_size)
        _validate_payload_data(descriptor, data)
        received[payload_id] = WorkerPayload(descriptor, data)
    boundary_header = _read_exact(stream, _FRAME_HEADER.size)
    (boundary_size,) = _FRAME_HEADER.unpack(boundary_header)
    if boundary_size:
        if boundary_size <= limits.max_binary_frame_size:
            _read_exact(stream, boundary_size)
        raise _worker_protocol_error("binary_payload_is_not_declared_for_this_message")
    return received


def validate_payload_descriptors(
    descriptors: Sequence[WorkerPayloadDescriptor],
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Validate CEDTS payload metadata and declared resource limits."""
    if len(descriptors) > limits.max_payload_descriptors:
        raise _worker_protocol_error("too_many_binary_payload_descriptors")
    ids: set[str] = set()
    total = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise _worker_protocol_error("binary_payload_descriptor_is_not_an_object")
        if not set(descriptor).issubset(_DESCRIPTOR_FIELDS):
            raise _worker_protocol_error(
                "binary_payload_descriptor_contains_unknown_fields"
            )
        payload_id = descriptor.get("id")
        media_type = descriptor.get("media_type")
        byte_length = descriptor.get("byte_length")
        if (
            not isinstance(payload_id, str)
            or not payload_id
            or len(payload_id) > _MAX_PAYLOAD_ID_LENGTH
            or payload_id in ids
        ):
            raise _worker_protocol_error("binary_payload_id_is_invalid")
        if not isinstance(media_type, str) or media_type not in _VALID_MEDIA_TYPES:
            raise _worker_protocol_error("binary_payload_media_type_is_invalid")
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
        ):
            raise _worker_protocol_error("binary_payload_length_is_invalid")
        for field_name in ("dtype", "sample_rate", "channels"):
            field_value = descriptor.get(field_name)
            if field_name == "dtype":
                valid_type = isinstance(field_value, str)
            else:
                valid_type = isinstance(field_value, int) and not isinstance(
                    field_value, bool
                )
            if field_value is not None and not valid_type:
                raise _worker_protocol_error(
                    "binary_payload_field_type_invalid", field_name=field_name
                )
        shape = descriptor.get("shape")
        if shape is not None and not isinstance(shape, list):
            raise _worker_protocol_error("binary_payload_shape_is_invalid")
        frame_size = _BINARY_HEADER.size + len(payload_id.encode("utf-8")) + byte_length
        if frame_size > limits.max_binary_frame_size:
            raise _worker_protocol_error("binary_payload_frame_is_too_large")
        _validate_descriptor_shape(descriptor)
        ids.add(payload_id)
        total += byte_length
        if total > limits.max_aggregate_payload_size:
            raise _worker_protocol_error("aggregate_binary_payload_is_too_large")


def _validate_descriptor_shape(descriptor: WorkerPayloadDescriptor) -> None:
    """Validate dtype, shape, and audio metadata for one payload descriptor."""
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    media_type = descriptor["media_type"]
    if dtype is not None and dtype not in _VALID_DTYPES:
        raise _worker_protocol_error("binary_payload_dtype_is_invalid")
    if shape is not None:
        if (
            not isinstance(shape, list)
            or len(shape) > _MAX_SHAPE_DIMENSIONS
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                or dimension > _MAX_SHAPE_DIMENSION
                for dimension in shape
            )
        ):
            raise _worker_protocol_error("binary_payload_shape_is_invalid")
        if dtype is not None:
            expected_size = np.dtype(dtype).itemsize
            for dimension in shape:
                expected_size *= dimension
            if expected_size != descriptor["byte_length"]:
                raise _worker_protocol_error(
                    "binary_payload_shape_does_not_match_its_length"
                )
    if media_type.startswith("audio/"):
        sample_rate = descriptor.get("sample_rate")
        channels = descriptor.get("channels")
        expected_dtype = {
            "audio/pcm_f32le": "float32",
            "audio/pcm_s16le": "int16",
        }.get(media_type)
        if (
            dtype != expected_dtype
            or not isinstance(shape, list)
            or len(shape) not in {1, 2}
            or not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
            or not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels <= 0
            or channels > 32
            or channels != (shape[-1] if len(shape) == 2 else 1)
        ):
            raise _worker_protocol_error("audio_payload_metadata_is_invalid")


def _validate_payload_data(descriptor: WorkerPayloadDescriptor, data: bytes) -> None:
    """Validate one received payload after its bounded frame has been read."""
    if len(data) != descriptor["byte_length"]:
        raise _worker_protocol_error("received_binary_payload_has_an_unexpected_length")
    _validate_descriptor_shape(descriptor)


def _encode_value(value: WorkerValue, builder: _PayloadBuilder) -> object:
    """Encode one backend value as JSON-compatible metadata."""
    if isinstance(value, np.ndarray):
        dtype = str(value.dtype)
        if dtype not in _VALID_DTYPES:
            raise _worker_protocol_error("unsupported_numpy_dtype_for_binary_payload")
        builder.preflight(
            value.nbytes,
            media_type="application/x-tensor",
            dtype=dtype,
            shape=value.shape,
        )
        array = np.ascontiguousarray(value)
        reference = builder.add(
            array.tobytes(order="C"),
            media_type="application/x-tensor",
            dtype=dtype,
            shape=array.shape,
        )
        return {"__cedts_type__": "numpy_array", "payload": reference}
    if isinstance(value, bytes):
        return builder.add(value, media_type="application/octet-stream")
    if isinstance(value, VoiceConversionRequest):
        return {
            "__cedts_type__": "voice_conversion_request",
            "source_audio": _encode_audio(
                value.source_audio, value.sample_rate, builder
            ),
            "sample_rate": value.sample_rate,
            "target_voice": value.target_voice,
            "target_character": value.target_character,
            "target_references": [str(path) for path in value.target_references],
            "label": value.label,
            "pitch_shift": value.pitch_shift,
            "f0_condition": value.f0_condition,
        }
    if isinstance(value, AudioOutput):
        return {
            "__cedts_type__": "audio_output",
            "audio": _encode_audio(value.audio, value.sample_rate, builder),
            "sample_rate": value.sample_rate,
            "label": value.label,
        }
    if isinstance(value, tuple):
        if len(value) == 3 and isinstance(value[0], np.ndarray):
            audio = value[0]
            return {
                "__cedts_type__": "backend_generation",
                "audio": _encode_audio(audio, cast(int, value[1]), builder),
                "sample_rate": cast(int, value[1]),
                "shape": list(audio.shape),
                "channels": audio.shape[-1] if audio.ndim == 2 else 1,
                "metadata": _encode_value(value[2], builder),
            }
        return {
            "__cedts_type__": "tuple",
            "items": [_encode_value(item, builder) for item in value],
        }
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_ENTRIES:
            raise _worker_protocol_error("json_object_contains_too_many_fields")
        return {
            key: _encode_value(cast(WorkerValue, item), builder)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION_ENTRIES:
            raise _worker_protocol_error("json_array_contains_too_many_entries")
        return [_encode_value(item, builder) for item in value]
    if isinstance(value, Path):
        return str(value)
    return cast(WorkerValue, value)


def _encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    builder: _PayloadBuilder,
) -> dict[str, str]:
    """Encode one audio array with CEDTS audio metadata."""
    dtype = str(audio.dtype)
    media_type = {"float32": "audio/pcm_f32le", "int16": "audio/pcm_s16le"}.get(dtype)
    if media_type is None:
        raise _worker_protocol_error("audio_payload_must_use_float32_or_int16_dtype")
    builder.preflight(
        audio.nbytes,
        media_type=media_type,
        dtype=dtype,
        shape=audio.shape,
        sample_rate=sample_rate,
        channels=audio.shape[-1] if audio.ndim == 2 else 1,
    )
    array = np.ascontiguousarray(audio)
    return builder.add(
        array.tobytes(order="C"),
        media_type=media_type,
        dtype=dtype,
        shape=array.shape,
        sample_rate=sample_rate,
        channels=array.shape[-1] if array.ndim == 2 else 1,
    )


def _decode_value(value: object, payloads: Mapping[str, WorkerPayload]) -> WorkerValue:
    """Decode one JSON value and its payload references."""
    if isinstance(value, dict):
        payload_id = value.get("__cedts_payload_id__")
        if payload_id is not None:
            if not isinstance(payload_id, str) or len(value) != 1:
                raise _worker_protocol_error(
                    "binary_value_payload_reference_is_invalid"
                )
            return payloads[payload_id].data
        value_type = value.get("__cedts_type__")
        if value_type is not None and not isinstance(value_type, str):
            raise _worker_protocol_error("typed_cedts_value_has_an_invalid_type")
        if value_type == "numpy_array":
            _validate_typed_fields(value, {"__cedts_type__", "payload"})
            reference = value.get("payload")
            return _decode_array(reference, payloads)
        if value_type == "voice_conversion_request":
            _validate_typed_fields(
                value,
                {
                    "__cedts_type__",
                    "source_audio",
                    "sample_rate",
                    "target_voice",
                    "target_character",
                    "target_references",
                    "label",
                    "pitch_shift",
                    "f0_condition",
                },
            )
            target_references = value.get("target_references", [])
            if not isinstance(target_references, list) or not all(
                isinstance(path, str) for path in target_references
            ):
                raise _worker_protocol_error(
                    "voice_conversion_target_references_are_invalid"
                )
            sample_rate = _typed_int(value.get("sample_rate"), "sample_rate")
            _typed_optional(value.get("target_voice"), str, "target_voice")
            _typed_optional(value.get("target_character"), str, "target_character")
            _typed_optional(value.get("pitch_shift"), int, "pitch_shift")
            _typed_optional(value.get("f0_condition"), bool, "f0_condition")
            return VoiceConversionRequest(
                source_audio=_decode_audio(
                    value["source_audio"],
                    payloads,
                    expected_sample_rate=sample_rate,
                ),
                sample_rate=sample_rate,
                target_voice=value.get("target_voice"),
                target_character=value.get("target_character"),
                target_references=tuple(Path(path) for path in target_references),
                label=_typed_string(value.get("label", "audio input"), "label"),
                pitch_shift=value.get("pitch_shift"),
                f0_condition=value.get("f0_condition"),
            )
        if value_type == "audio_output":
            _validate_typed_fields(
                value,
                {"__cedts_type__", "audio", "sample_rate", "label"},
            )
            sample_rate = _typed_int(value.get("sample_rate"), "sample_rate")
            return AudioOutput(
                audio=_decode_audio(
                    value["audio"],
                    payloads,
                    expected_sample_rate=sample_rate,
                ),
                sample_rate=sample_rate,
                label=_typed_string(value.get("label", "audio output"), "label"),
            )
        if value_type == "backend_generation":
            _validate_typed_fields(
                value,
                {
                    "__cedts_type__",
                    "audio",
                    "sample_rate",
                    "shape",
                    "channels",
                    "metadata",
                },
            )
            sample_rate = _typed_int(value.get("sample_rate"), "sample_rate")
            shape = value.get("shape")
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or isinstance(dimension, bool)
                for dimension in shape
            ):
                raise _worker_protocol_error(
                    "typed_value_field_is_invalid", field_name="shape"
                )
            channels = _typed_int(value.get("channels"), "channels")
            metadata = _decode_value(value.get("metadata"), payloads)
            audio = _decode_audio(
                value["audio"],
                payloads,
                expected_sample_rate=sample_rate,
                expected_shape=tuple(shape),
                expected_channels=channels,
            )
            return (
                audio,
                sample_rate,
                metadata,
            )
        if value_type == "tuple":
            _validate_typed_fields(value, {"__cedts_type__", "items"})
            items = value.get("items")
            if not isinstance(items, list):
                raise _worker_protocol_error("typed_tuple_items_are_invalid")
            return tuple(_decode_value(item, payloads) for item in items)
        if value_type is not None:
            raise _worker_protocol_error("typed_cedts_value_is_unknown")
        return {key: _decode_value(item, payloads) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, payloads) for item in value]
    return cast(WorkerValue, value)


def _decode_array(value: object, payloads: Mapping[str, WorkerPayload]) -> np.ndarray:
    """Decode a typed NumPy payload without reconstructing Python objects."""
    payload_id = _payload_id(value)
    payload = payloads[payload_id]
    if payload.descriptor.get("media_type") not in _TENSOR_MEDIA_TYPES:
        raise _worker_protocol_error("binary_payload_media_type_is_invalid")
    dtype_name = payload.descriptor.get("dtype")
    shape = payload.descriptor.get("shape")
    if not isinstance(dtype_name, str) or not isinstance(shape, list):
        raise _worker_protocol_error("numpy_payload_metadata_is_incomplete")
    return (
        np.frombuffer(payload.data, dtype=np.dtype(dtype_name))
        .reshape(tuple(shape))
        .copy()
    )


def _decode_audio(
    value: object,
    payloads: Mapping[str, WorkerPayload],
    *,
    expected_sample_rate: Optional[int] = None,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_channels: Optional[int] = None,
) -> np.ndarray:
    """Decode an audio payload and normalize signed PCM to float32."""
    payload_id = _payload_id(value)
    payload = payloads[payload_id]
    try:
        if payload.descriptor.get("media_type") not in _AUDIO_MEDIA_TYPES:
            raise _worker_protocol_error("binary_payload_media_type_is_invalid")
        _validate_payload_data(payload.descriptor, payload.data)
        dtype_name = payload.descriptor.get("dtype")
        shape = payload.descriptor.get("shape")
        if not isinstance(dtype_name, str) or not isinstance(shape, list):
            raise _worker_protocol_error("audio_payload_metadata_is_incomplete")
        array = np.frombuffer(payload.data, dtype=np.dtype(dtype_name)).reshape(
            tuple(shape)
        )
        channels = array.shape[-1] if array.ndim == 2 else 1
        if (
            expected_sample_rate is not None
            and payload.descriptor.get("sample_rate") != expected_sample_rate
        ):
            raise _worker_protocol_error("audio_payload_metadata_is_invalid")
        if expected_shape is not None and tuple(array.shape) != expected_shape:
            raise _worker_protocol_error("audio_payload_metadata_is_invalid")
        if expected_channels is not None and channels != expected_channels:
            raise _worker_protocol_error("audio_payload_metadata_is_invalid")
        if (
            payload.descriptor.get("shape") != list(array.shape)
            or payload.descriptor.get("channels") != channels
        ):
            raise _worker_protocol_error("audio_payload_metadata_is_invalid")
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise _worker_protocol_error("audio_payload_metadata_is_invalid") from error
    if dtype_name == "int16":
        return (array.astype(np.float32) / np.float32(32768.0)).copy()
    normalized = array.astype(np.float32, copy=True)
    if (
        not np.all(np.isfinite(normalized))
        or np.any(normalized < np.float32(-1.0))
        or np.any(normalized > np.float32(1.0))
    ):
        raise _worker_protocol_error("audio_payload_samples_are_not_normalized")
    return normalized


def _payload_id(value: object) -> str:
    """Extract a payload ID from a JSON payload reference."""
    if not isinstance(value, dict) or len(value) != 1:
        raise _worker_protocol_error("binary_value_is_not_a_payload_reference")
    payload_id = value.get("__cedts_payload_id__")
    if not isinstance(payload_id, str):
        raise _worker_protocol_error("binary_value_has_no_payload_id")
    return payload_id


def _object_from_pairs(
    pairs: list[tuple[str, object]], max_entries: int = _MAX_COLLECTION_ENTRIES
) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate or excessive fields."""
    if len(pairs) > max_entries:
        raise ValueError("JSON object contains too many fields")
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as ``NaN`` and ``Infinity``."""
    raise ValueError(f"invalid JSON constant: {value}")


def _validate_json_value(
    value: object,
    depth: int = 0,
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Validate bounded JSON-compatible values without deserialization."""
    if depth > limits.max_json_depth:
        raise ValueError("JSON value is nested too deeply")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON number is not finite")
        return
    if isinstance(value, str):
        if len(value) > limits.max_string_length:
            raise ValueError("JSON string is too long")
        return
    if isinstance(value, list):
        if len(value) > limits.max_collection_entries:
            raise ValueError("JSON array contains too many entries")
        for item in value:
            _validate_json_value(item, depth + 1, limits=limits)
        return
    if isinstance(value, dict):
        if len(value) > limits.max_collection_entries:
            raise ValueError("JSON object contains too many fields")
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object field name is not a string")
            _validate_json_value(key, depth + 1, limits=limits)
            _validate_json_value(item, depth + 1, limits=limits)
        return
    raise TypeError(f"unsupported CEDTS JSON value: {type(value).__name__}")


def _validate_packet(
    packet: Mapping[str, object], *, limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS
) -> None:
    """Validate a CEDTS packet envelope and its kind-specific data schema."""
    if not set(packet).issubset(_PACKET_FIELDS):
        raise _worker_protocol_error("backend_worker_packet_contains_unknown_fields")
    version = packet.get("cedts_version")
    kind = packet.get("kind")
    message_id = packet.get("message_id")
    reply_to = packet.get("reply_to")
    operation = packet.get("operation")
    data = packet.get("data")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CEDTS_VERSION
        or not isinstance(kind, str)
        or kind not in CONTROL_PACKET_KINDS
        or not isinstance(message_id, str)
        or not message_id
        or len(message_id) > _MAX_MESSAGE_ID_LENGTH
        or not isinstance(reply_to, (str, type(None)))
        or (isinstance(reply_to, str) and len(reply_to) > _MAX_MESSAGE_ID_LENGTH)
        or not isinstance(operation, str)
        or not operation
        or len(operation) > _MAX_OPERATION_LENGTH
        or not isinstance(data, dict)
    ):
        raise _worker_protocol_error("backend_worker_packet_envelope_is_invalid")
    descriptors = packet.get("payloads", [])
    if not isinstance(descriptors, list):
        raise _worker_protocol_error(
            "backend_worker_payload_descriptors_are_not_an_array"
        )
    validate_payload_descriptors(descriptors, limits=limits)
    _validate_packet_data(kind, operation, data)


def _validate_packet_data(kind: str, operation: str, data: dict[str, object]) -> None:
    """Validate the typed data object associated with one packet kind."""
    if kind == "hello":
        _validate_operation_match(operation, "handshake")
        _validate_hello_data(data)
    elif kind == "hello_ack":
        _validate_operation_match(operation, "handshake")
        _validate_exact_fields(data, {"cedts_version", "capabilities"})
        _validate_version(data["cedts_version"])
        _validate_capabilities(data["capabilities"])
    elif kind == "ready":
        _validate_operation_match(operation, "ready")
        _validate_exact_fields(data, {"capabilities"})
        _validate_capabilities(data["capabilities"])
    elif kind == "request":
        if operation not in SUPPORTED_OPERATIONS:
            raise _worker_protocol_error("backend_worker_request_operation_is_invalid")
        _validate_exact_fields(data, {"arguments"})
        _validate_operation_arguments(operation, data["arguments"])
    elif kind == "response":
        if operation not in SUPPORTED_OPERATIONS:
            raise _worker_protocol_error("backend_worker_response_operation_is_invalid")
        _validate_response_data(data)
    elif kind == "event":
        if operation not in _EVENT_OPERATIONS:
            raise _worker_protocol_error("backend_worker_event_operation_is_invalid")
        _validate_event_data(operation, data)
    elif kind == "callback":
        if operation not in _STATE_VALUES:
            raise _worker_protocol_error("backend_worker_callback_operation_is_invalid")
        _validate_state_data(data)
    elif kind == "progress":
        if operation not in SUPPORTED_OPERATIONS:
            raise _worker_protocol_error("backend_worker_progress_operation_is_invalid")
        _validate_progress_data(data)
    elif kind == "cancel":
        _validate_operation_match(operation, "cancel")
        _validate_exact_fields(data, {"target_message_id"})
        _validate_identifier(data["target_message_id"], "cancel target message ID")
    elif kind == "cancel_ack":
        _validate_operation_match(operation, "cancel")
        _validate_exact_fields(data, {"ok", "cancelled", "target_message_id"})
        _validate_bool(data["ok"], "cancel acknowledgement status")
        _validate_bool(data["cancelled"], "cancelled status")
        target = data["target_message_id"]
        if target is not None:
            _validate_identifier(target, "cancel target message ID")
    elif kind == "error":
        if operation not in _VALID_OPERATIONS:
            raise _worker_protocol_error("backend_worker_error_operation_is_invalid")
        _validate_error_data(data)
    elif kind in {"ping", "pong"}:
        _validate_operation_match(operation, "protocol")
        _validate_exact_fields(data, set())
    elif kind == "shutdown":
        _validate_operation_match(operation, "shutdown")
        _validate_exact_fields(data, {"active_job_policy"})
        if data["active_job_policy"] not in {"cancel", "finish"}:
            raise _worker_protocol_error("shutdown_active_job_policy_is_invalid")
    elif kind == "shutdown_ack":
        _validate_operation_match(operation, "shutdown")
        _validate_response_data(data, required={"ok", "value"})


def _validate_operation_match(operation: str, expected: str) -> None:
    """Require a packet operation to match its protocol-level operation."""
    if operation != expected:
        raise _worker_protocol_error(
            "backend_worker_packet_operation_does_not_match_kind"
        )


def _validate_exact_fields(
    value: object,
    required: set[str],
    optional: Optional[set[str]] = None,
) -> dict[str, object]:
    """Return an object after rejecting missing and unknown fields."""
    if not isinstance(value, dict):
        raise _worker_protocol_error("backend_worker_packet_data_is_not_an_object")
    allowed = required | (optional or set())
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise _worker_protocol_error("backend_worker_packet_data_fields_are_invalid")
    return value


def _validate_version(value: object) -> None:
    """Validate one negotiated CEDTS version."""
    if not isinstance(value, int) or isinstance(value, bool) or value != CEDTS_VERSION:
        raise _worker_protocol_error("backend_worker_cedts_version_is_invalid")


def _validate_identifier(value: object, field_name: str) -> None:
    """Validate a bounded packet identifier."""
    if not isinstance(value, str) or not value or len(value) > _MAX_MESSAGE_ID_LENGTH:
        raise _worker_protocol_error("field_is_invalid", field_name=field_name)


def _validate_bool(value: object, field_name: str) -> None:
    """Validate a required boolean packet field."""
    if not isinstance(value, bool):
        raise _worker_protocol_error("field_is_invalid", field_name=field_name)


def _validate_hello_data(data: dict[str, object]) -> None:
    """Validate the core-to-worker handshake data."""
    _validate_exact_fields(
        data, {"versions", "capabilities"}, {"required_capabilities"}
    )
    versions = data["versions"]
    if (
        not isinstance(versions, list)
        or not versions
        or any(
            not isinstance(version, int) or isinstance(version, bool)
            for version in versions
        )
    ):
        raise _worker_protocol_error("cedts_hello_versions_are_invalid")
    _validate_capabilities(data["capabilities"])
    if "required_capabilities" in data:
        _validate_capability_requirements(data["required_capabilities"])


def _validate_capabilities(value: object) -> None:
    """Validate the bounded capability negotiation object."""
    if not isinstance(value, dict):
        raise _worker_protocol_error("cedts_capabilities_are_invalid")
    required = {
        "supported_operations",
        "supported_media_types",
        "max_control_frame_size",
        "max_binary_frame_size",
        "streaming",
        "cancellation",
        "callback",
    }
    optional = set(_LIMIT_CAPABILITY_FIELDS) - {
        "max_control_frame_size",
        "max_binary_frame_size",
    }
    _validate_exact_fields(value, required, optional)
    for field_name in ("supported_operations", "supported_media_types"):
        values = value[field_name]
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise _worker_protocol_error(
                "cedts_capability_field_is_invalid", field_name=field_name
            )
    for field_name in _LIMIT_CAPABILITY_FIELDS:
        if field_name not in value:
            continue
        limit = value[field_name]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise _worker_protocol_error(
                "cedts_capability_field_is_invalid", field_name=field_name
            )
    for field_name in ("streaming", "cancellation", "callback"):
        _validate_bool(value[field_name], f"CEDTS capability {field_name}")


def _validate_capability_requirements(value: object) -> None:
    """Validate required capability flags without accepting arbitrary values."""
    if not isinstance(value, dict):
        raise _worker_protocol_error("cedts_required_capabilities_are_invalid")
    allowed = {
        "streaming",
        "cancellation",
        "callback",
        "supported_operations",
        "supported_media_types",
    }
    if not set(value).issubset(allowed):
        raise _worker_protocol_error(
            "cedts_required_capabilities_contain_unknown_fields"
        )
    for name, required in value.items():
        if name in {"streaming", "cancellation", "callback"}:
            _validate_bool(required, f"required capability {name}")
        elif not isinstance(required, list) or any(
            not isinstance(item, str) for item in required
        ):
            raise _worker_protocol_error("required_capability_is_invalid", name=name)


def _validate_operation_arguments(operation: str, value: object) -> None:
    """Validate one request's operation-specific argument object."""
    if not isinstance(value, dict):
        raise _worker_protocol_error(
            "backend_worker_request_arguments_are_not_an_object"
        )
    if operation in {"describe", "preload_models"}:
        _validate_exact_fields(value, set())
    elif operation == "model_is_available_locally":
        _validate_exact_fields(value, {"model"}, {"lang"})
        if not isinstance(value["model"], str):
            raise _worker_protocol_error("backend_worker_model_argument_is_invalid")
        if (
            "lang" in value
            and value["lang"] is not None
            and not isinstance(value["lang"], str)
        ):
            raise _worker_protocol_error("backend_worker_language_argument_is_invalid")
    elif operation == "load_model":
        _validate_model_arguments(value, str)
    elif operation == "generate_stream":
        _validate_model_arguments(value, int)
    elif operation == "unload_model":
        _validate_exact_fields(value, set(), {"release_cuda_cache"})
        if "release_cuda_cache" in value:
            _validate_bool(value["release_cuda_cache"], "release CUDA cache argument")
    elif operation == "convert":
        _validate_exact_fields(value, {"request"})
        if not isinstance(value["request"], dict):
            raise _worker_protocol_error("backend_worker_conversion_request_is_invalid")
    elif operation == "call":
        _validate_call_arguments(value)


def _validate_model_arguments(value: dict[str, object], model_type: type) -> None:
    """Validate a model handle plus bounded backend-specific keyword fields."""
    if not isinstance(value, dict) or "model_id" not in value:
        raise _worker_protocol_error("backend_worker_model_arguments_are_invalid")
    model_id = value["model_id"]
    if not isinstance(model_id, model_type) or (
        model_type is int and isinstance(model_id, bool)
    ):
        raise _worker_protocol_error("backend_worker_model_identifier_is_invalid")
    _validate_backend_kwargs(value, {"model_id"})


def _validate_backend_kwargs(value: dict[str, object], reserved: set[str]) -> None:
    """Validate explicitly bounded backend-specific JSON keyword arguments."""
    if len(value) - len(reserved) > _MAX_BACKEND_ARGUMENT_FIELDS:
        raise _worker_protocol_error(
            "backend_worker_backend_arguments_contain_too_many_fields"
        )
    for name, argument in value.items():
        if name in reserved:
            continue
        if (
            not isinstance(name, str)
            or not name
            or len(name) > _MAX_ARGUMENT_NAME_LENGTH
        ):
            raise _worker_protocol_error("backend_worker_argument_name_is_invalid")
        _validate_json_value(argument)


def _validate_call_arguments(value: dict[str, object]) -> None:
    """Validate one allowlisted backend callback invocation."""
    method = value.get("method")
    if not isinstance(method, str):
        raise _worker_protocol_error("backend_worker_callback_method_is_invalid")
    fields = {
        "resolve_generation_language": {"method", "lang"},
        "should_reload_for_language": {"method", "lang"},
        "convert_live": {"method", "request"},
        "stop_live": {"method"},
    }.get(method)
    if fields is None or set(value) != fields:
        raise _worker_protocol_error("backend_worker_callback_arguments_are_invalid")
    if method in {"resolve_generation_language", "should_reload_for_language"}:
        if value["lang"] is not None and not isinstance(value["lang"], str):
            raise _worker_protocol_error("backend_worker_language_argument_is_invalid")
    elif method == "convert_live" and not isinstance(value["request"], dict):
        raise _worker_protocol_error(
            "backend_worker_live_conversion_request_is_invalid"
        )


def _validate_response_data(
    data: dict[str, object],
    *,
    required: Optional[set[str]] = None,
) -> None:
    """Validate response, stream-terminal, and shutdown acknowledgement data."""
    allowed = {"ok", "value", "error", "error_type", "stream", "done", "cancelled"}
    required_fields = {"ok"} if required is None else required
    _validate_exact_fields(data, required_fields, allowed - required_fields)
    _validate_bool(data["ok"], "backend worker response status")
    for field_name in ("stream", "done", "cancelled"):
        if field_name in data:
            _validate_bool(data[field_name], f"backend worker response {field_name}")
    for field_name in ("error", "error_type"):
        if field_name in data and not isinstance(data[field_name], str):
            raise _worker_protocol_error(
                "response_field_is_invalid", field_name=field_name
            )


def _validate_error_data(data: dict[str, object]) -> None:
    """Validate a structured error packet."""
    _validate_exact_fields(data, {"ok", "error", "error_type"})
    _validate_bool(data["ok"], "backend worker error status")
    if (
        data["ok"] is not False
        or not isinstance(data["error"], str)
        or not isinstance(data["error_type"], str)
    ):
        raise _worker_protocol_error("backend_worker_error_data_is_invalid")


def _validate_state_data(data: dict[str, object]) -> None:
    """Validate a state callback payload."""
    _validate_exact_fields(data, {"state"}, {"message"})
    if data["state"] not in _STATE_VALUES:
        raise _worker_protocol_error("backend_worker_callback_state_is_invalid")
    if "message" in data and not isinstance(data["message"], str):
        raise _worker_protocol_error("backend_worker_callback_message_is_invalid")


def _validate_event_data(operation: str, data: dict[str, object]) -> None:
    """Validate fatal and state event payloads."""
    if operation == "fatal":
        _validate_exact_fields(data, {"fatal"}, {"message", "error", "error_type"})
        _validate_bool(data["fatal"], "backend worker fatal status")
        for field_name in ("message", "error", "error_type"):
            if field_name in data and not isinstance(data[field_name], str):
                raise _worker_protocol_error(
                    "fatal_field_is_invalid", field_name=field_name
                )
        return
    _validate_state_data(data)


def _validate_progress_data(data: dict[str, object]) -> None:
    """Validate a bounded progress event payload."""
    _validate_exact_fields(data, {"step"}, {"total", "message", "state"})
    step = data["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise _worker_protocol_error("backend_worker_progress_step_is_invalid")
    if "total" in data and (
        not isinstance(data["total"], int)
        or isinstance(data["total"], bool)
        or data["total"] < 0
    ):
        raise _worker_protocol_error("backend_worker_progress_total_is_invalid")
    if "message" in data and not isinstance(data["message"], str):
        raise _worker_protocol_error("backend_worker_progress_message_is_invalid")
    if "state" in data and data["state"] not in _STATE_VALUES:
        raise _worker_protocol_error("backend_worker_progress_state_is_invalid")


def _validate_operation(operation: object) -> None:
    """Reject operations outside the CEDTS protocol and backend allowlist."""
    if not isinstance(operation, str) or operation not in _VALID_OPERATIONS:
        raise _worker_protocol_error("backend_worker_packet_operation_is_invalid")


def _validate_typed_fields(value: Mapping[str, object], fields: set[str]) -> None:
    """Reject unknown or excessive fields in a typed binary wrapper."""
    if len(value) > _MAX_TYPED_VALUE_FIELDS or not set(value).issubset(fields):
        raise _worker_protocol_error("typed_cedts_value_fields_are_invalid")


def _typed_string(value: object, field_name: str) -> str:
    """Return a required typed string or reject malformed metadata."""
    if not isinstance(value, str):
        raise _worker_protocol_error(
            "typed_value_field_is_invalid", field_name=field_name
        )
    return value


def _typed_int(value: object, field_name: str) -> int:
    """Return a required typed integer or reject malformed metadata."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise _worker_protocol_error(
            "typed_value_field_is_invalid", field_name=field_name
        )
    return value


def _typed_optional(value: object, expected_type: type, field_name: str) -> None:
    """Validate an optional typed metadata field."""
    if value is not None and (
        not isinstance(value, expected_type)
        or expected_type is int
        and isinstance(value, bool)
    ):
        raise _worker_protocol_error(
            "typed_value_field_is_invalid", field_name=field_name
        )


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    """Read exactly ``size`` bytes or raise when the worker closes its stream."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise _worker_protocol_error("backend_worker_closed_its_protocol_stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
