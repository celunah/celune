# SPDX-License-Identifier: MIT
"""Private framed protocol used by isolated backend worker processes."""

import pickle
import struct
from typing import IO

from ..typing.worker import WorkerMessage

__all__ = ["WorkerProtocolError", "receive_message", "send_message"]

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_SIZE = 64 * 1024 * 1024


class WorkerProtocolError(RuntimeError):
    """Raised when a backend worker sends an invalid or incomplete frame."""


def send_message(stream: IO[bytes], message: WorkerMessage) -> None:
    """Write one length-prefixed pickle message to a worker stream."""
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_FRAME_SIZE:
        raise WorkerProtocolError("backend worker message is too large")
    stream.write(_FRAME_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def receive_message(stream: IO[bytes]) -> WorkerMessage:
    """Read one length-prefixed pickle message from a worker stream."""
    header = _read_exact(stream, _FRAME_HEADER.size)
    (payload_size,) = _FRAME_HEADER.unpack(header)
    if payload_size > _MAX_FRAME_SIZE:
        raise WorkerProtocolError("backend worker message is too large")
    return pickle.loads(_read_exact(stream, payload_size))


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    """Read exactly ``size`` bytes or raise when the worker closes its stream."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise WorkerProtocolError("backend worker closed its protocol stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
