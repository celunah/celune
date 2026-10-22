# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for isolated backend tests."""

from __future__ import annotations
import io
import threading
import subprocess
from unittest import mock
from typing import Optional, cast
from collections import OrderedDict, deque

from celune.cedts import remote
from celune.typing.worker import WorkerValue
from celune.backends.environment import BackendManifest
from celune.cedts.protocol import build_packet, send_message

from .backend_processes import ShutdownProcess


def make_shutdown_proxy(
    *,
    active_request_id: Optional[str] = None,
    waits_before_exit: int = 0,
    shutdown_ok: bool = True,
) -> tuple[remote.RemoteBackendProxy, ShutdownProcess]:
    """Build a proxy with an in-memory CEDTS shutdown acknowledgement."""
    stream = io.BytesIO()
    send_message(
        stream,
        build_packet(
            "shutdown_ack",
            "shutdown",
            {
                "ok": shutdown_ok,
                "value": cast(
                    dict[str, WorkerValue],
                    {
                        "active_job_policy": "cancel",
                        "active_job_cancelled": active_request_id is not None,
                    },
                ),
            },
            reply_to="shutdown-id",
        ),
    )
    stream.seek(0)
    process = ShutdownProcess(stream, waits_before_exit)
    proxy = object.__new__(remote.RemoteBackendProxy)
    proxy._manifest = BackendManifest("fake", "tts", (), "module", "Backend")
    proxy._process = cast(subprocess.Popen[bytes], process)
    proxy._close_lock = threading.Lock()
    proxy._closing = False
    proxy._closed = False
    proxy._protocol_lock = threading.Lock()
    proxy._send_lock = threading.Lock()
    proxy._log_callback = mock.Mock()
    proxy._worker_stderr = deque()
    proxy._worker_stderr_lock = threading.Lock()
    proxy._stderr_thread = None
    proxy._binary_input = None
    proxy._binary_output = None
    proxy._received_message_ids = OrderedDict()
    proxy._active_request_lock = threading.Lock()
    proxy._active_request_id = active_request_id
    proxy._cancel_target = None
    proxy._cancel_packet_id = None
    proxy._cancel_ack_event = None
    proxy._cancel_ack_result = None
    proxy._cancel_sent = False
    proxy._send_packet = mock.Mock(return_value="shutdown-id")
    return proxy, process
