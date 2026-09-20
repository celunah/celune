# SPDX-License-Identifier: Apache-2.0
"""Tests for isolated backend environment metadata and installation."""

import io
import os
import time
import threading
import subprocess
from types import SimpleNamespace
from typing import IO, Optional, cast
from pathlib import Path
from unittest import mock
from contextlib import suppress
from collections import OrderedDict, deque
from collections.abc import Callable, Generator


from celune.cedts import remote, worker
from celune.exceptions import (
    CEDTSError,
    BackendError,
    CEDTSEOFError,
    CEDTSTimeoutError,
    CEDTSProtocolError,
)
from celune.typing.worker import (
    WorkerValue,
    WorkerMessage,
)
from celune.cedts.protocol import (
    CEDTSLimits,
    WorkerPayload,
    build_packet,
    send_message,
    send_payloads,
    receive_message,
    receive_payloads,
    limits_from_capabilities,
    validate_payload_descriptors,
)
from celune.typing.backends import (
    BackendModel,
    _BackendRuntime,
)
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendManifest,
    BackendEnvironment,
)

from .backend_processes import ShutdownProcess
from .backend_environment_and_cedts import (
    TestBackendEnvironment as _TestBackendEnvironment,
)


class TestBackendEnvironment(_TestBackendEnvironment):
    """Exercise remote backend proxy lifecycle behavior."""

    def test_remote_proxy_aborts_timed_out_model_operation(self) -> None:
        """Verify timed-out model operations terminate their isolated worker."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BACKEND_MANIFESTS["mini"]
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(
                stdin=io.BytesIO(),
                stdout=io.BytesIO(),
                poll=lambda: None,
            ),
        )
        proxy._protocol_lock = threading.Lock()
        proxy._log_callback = mock.Mock()
        proxy._send_packet = mock.Mock(return_value="request-id")
        proxy._read_response = mock.Mock(
            side_effect=TimeoutError("response did not arrive")
        )
        proxy.abort = mock.Mock()

        with self.assertRaisesRegex(
            CEDTSTimeoutError,
            "preload_models timed out after 0.01 seconds",
        ):
            proxy._request("preload_models", response_timeout=0.01)

        proxy.abort.assert_called_once_with()
        proxy._read_response.assert_called_once_with(
            proxy._process,
            "request-id",
            timeout=0.01,
            packet_name="preload_models",
        )

    def test_remote_proxy_unload_is_idempotent_after_worker_failure(self) -> None:
        """Verify cleanup does not report a stale worker from a dead proxy."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._closed = True
        proxy._closing = True
        proxy._process = None
        proxy.model = remote.RemoteModelHandle(1)
        proxy._request = mock.Mock()

        proxy.unload_model()

        self.assertIsNone(proxy.model)
        proxy._request.assert_not_called()

    def test_remote_proxy_allows_slow_model_initialization(self) -> None:
        """Verify model construction receives the long backend-operation deadline."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._request = mock.Mock(return_value=7)

        handle = proxy.load_model("slow-model")

        self.assertEqual(handle.identifier, 7)
        proxy._request.assert_called_once_with(
            "load_model",
            response_timeout=900.0,
            model_id="slow-model",
        )

    def test_remote_cancel_ack_is_consumed_before_stream_terminal_response(
        self,
    ) -> None:
        """Verify cancel acknowledgement correlation does not interrupt stream alignment."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "cancel_ack",
                "cancel",
                {
                    "ok": True,
                    "cancelled": True,
                    "target_message_id": "active-request",
                },
                reply_to="cancel-packet",
                message_id="cancel-ack",
            ),
        )
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": False, "cancelled": True, "done": True},
                reply_to="active-request",
                message_id="stream-done",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "active-request"
        proxy._cancel_target = "active-request"
        proxy._cancel_packet_id = "cancel-packet"
        proxy._cancel_ack_event = threading.Event()
        proxy._cancel_ack_result = None
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()

        response = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream)),
            "active-request",
        )

        self.assertTrue(response["cancelled"])
        self.assertTrue(proxy._cancel_ack_event.is_set())
        self.assertTrue(proxy._cancel_ack_result)

    def test_remote_late_cancel_ack_does_not_poison_request_reuse(self) -> None:
        """Verify a terminal request's late acknowledgement is safely ignored."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": False, "cancelled": True, "done": True},
                reply_to="old-request",
                message_id="old-terminal",
            ),
        )
        send_message(
            stream,
            build_packet(
                "cancel_ack",
                "cancel",
                {
                    "ok": True,
                    "cancelled": True,
                    "target_message_id": "old-request",
                },
                reply_to="old-cancel",
                message_id="late-cancel-ack",
            ),
        )
        send_message(
            stream,
            build_packet(
                "response",
                "describe",
                {"ok": True, "value": "reused"},
                reply_to="new-request",
                message_id="new-response",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "old-request"
        proxy._cancel_target = "old-request"
        proxy._cancel_packet_id = "old-cancel"
        proxy._cancel_ack_event = threading.Event()
        proxy._cancel_ack_result = None
        proxy._cancel_sent = True
        proxy._request_cancellation_states = {
            "old-request": remote._RequestCancellationState(
                request_id="old-request",
                terminal=True,
                cancel_packet_id="old-cancel",
                cancel_ack_event=proxy._cancel_ack_event,
                cancel_sent=True,
            )
        }
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()

        terminal = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream)),
            "old-request",
        )
        with proxy._active_request_lock:
            proxy._active_request_id = "new-request"
            proxy._request_cancellation_states["new-request"] = (
                remote._RequestCancellationState(request_id="new-request")
            )
        reused = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream)),
            "new-request",
        )

        self.assertTrue(terminal["done"])
        self.assertEqual(reused["value"], "reused")

    def test_remote_repeated_cancel_after_terminal_is_a_noop(self) -> None:
        """Verify a completed request cannot emit a stale repeated cancellation."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "completed-request"
        proxy._cancel_target = "completed-request"
        proxy._cancel_packet_id = "cancel-packet"
        proxy._cancel_ack_event = threading.Event()
        proxy._cancel_ack_result = True
        proxy._cancel_sent = True
        proxy._request_cancellation_states = {
            "completed-request": remote._RequestCancellationState(
                request_id="completed-request",
                terminal=True,
                cancel_packet_id="cancel-packet",
                cancel_ack_event=proxy._cancel_ack_event,
                cancel_ack_result=True,
                cancel_sent=True,
            )
        }
        proxy._send_packet = mock.Mock()

        self.assertFalse(proxy.cancel_active_request("completed-request"))
        proxy._send_packet.assert_not_called()

    def test_remote_terminal_request_releases_cancellation_state(self) -> None:
        """Verify terminal requests do not retain cancellation state indefinitely."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._active_request_lock = threading.Lock()
        proxy._request_cancellation_states = {
            "completed-request": remote._RequestCancellationState(
                request_id="completed-request",
            )
        }
        proxy._terminal_cancellation_states = OrderedDict()

        with proxy._active_request_lock:
            proxy._mark_request_terminal_locked("completed-request")

        self.assertFalse(proxy._request_cancellation_states)
        self.assertFalse(proxy._terminal_cancellation_states)

    def test_remote_terminal_request_preserves_inflight_cancellation_ack_race(
        self,
    ) -> None:
        """Verify a late cancellation acknowledgement resolves a terminal request."""
        cancel_event = threading.Event()
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "active-request"
        proxy._cancel_target = "active-request"
        proxy._cancel_packet_id = "cancel-packet"
        proxy._cancel_ack_event = cancel_event
        proxy._cancel_ack_result = None
        proxy._cancel_sent = True
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), poll=lambda: None),
        )
        proxy._request_cancellation_states = {
            "active-request": remote._RequestCancellationState(
                request_id="active-request",
                cancel_packet_id="cancel-packet",
                cancel_ack_event=cancel_event,
                cancel_sent=True,
            )
        }
        proxy._terminal_cancellation_states = OrderedDict()
        proxy._send_packet = mock.Mock()

        with proxy._active_request_lock:
            proxy._mark_request_terminal_locked("active-request")

        self.assertFalse(proxy._request_cancellation_states)
        self.assertIn("active-request", proxy._terminal_cancellation_states)
        self.assertFalse(
            proxy.cancel_active_request("active-request", wait_for_ack=False)
        )
        proxy._handle_cancel_ack(
            cast(
                WorkerMessage,
                {
                    "data": {
                        "target_message_id": "active-request",
                        "cancelled": True,
                    },
                    "reply_to": "cancel-packet",
                },
            )
        )

        self.assertTrue(cancel_event.is_set())
        self.assertFalse(proxy._terminal_cancellation_states)
        self.assertTrue(proxy._cancel_ack_result)
        proxy._send_packet.assert_not_called()

    def test_remote_proxy_gracefully_shuts_down_when_idle(self) -> None:
        """Verify idle shutdown sends a correlated packet and waits for its acknowledgement."""
        proxy, process = self._make_shutdown_proxy()
        send_packet = cast(mock.Mock, proxy._send_packet)

        proxy.close()

        send_packet.assert_called_once_with(
            proxy._process if proxy._process is not None else process.stdin,
            "shutdown",
            "shutdown",
            {"active_job_policy": "cancel"},
        )
        self.assertFalse(process.terminated)
        self.assertEqual(process.returncode, 0)

    def test_remote_proxy_abort_wakes_event_waiters_before_termination(self) -> None:
        """Verify abort releases a consumer blocked on the next worker event."""
        proxy, _process = self._make_shutdown_proxy()
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()
        proxy._response_condition = threading.Condition()
        proxy._reader_stop = threading.Event()
        proxy._reader_error = None
        error: list[CEDTSError] = []

        def wait_for_event() -> None:
            """Wait for the proxy event queue in a background consumer."""
            try:
                proxy.get_worker_event(timeout=30)
            except CEDTSError as raised:
                error.append(raised)

        waiter = threading.Thread(target=wait_for_event)
        waiter.start()
        proxy.abort()
        waiter.join(timeout=1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(error), 1)
        self.assertIsInstance(error[0], CEDTSEOFError)

    def test_remote_proxy_cancels_active_work_before_graceful_shutdown(self) -> None:
        """Verify active work is cancelled before the correlated shutdown exchange."""
        proxy, process = self._make_shutdown_proxy(active_request_id="active-request")
        send_packet = mock.Mock(side_effect=["cancel-id", "shutdown-id"])
        proxy._send_packet = send_packet

        proxy.close()

        self.assertEqual(send_packet.call_count, 2)
        send_packet.assert_has_calls(
            [
                mock.call(
                    process.stdin,
                    "cancel",
                    "cancel",
                    {"target_message_id": "active-request"},
                    message_id=mock.ANY,
                ),
                mock.call(
                    process.stdin,
                    "shutdown",
                    "shutdown",
                    {"active_job_policy": "cancel"},
                ),
            ]
        )
        self.assertFalse(process.terminated)

    def test_remote_proxy_closes_paused_stream_without_waiting_for_iteration(
        self,
    ) -> None:
        """Verify close shuts down a stream paused at a consumer yield."""
        binary_read, binary_write = os.pipe()
        worker_output = os.fdopen(binary_read, "rb", buffering=0)
        worker_input = os.fdopen(binary_write, "wb", buffering=0)
        shutdown_sent = threading.Event()
        request_sent = threading.Event()
        request_id: list[str] = []
        shutdown_ack = build_packet(
            "shutdown_ack",
            "shutdown",
            {
                "ok": True,
                "value": cast(
                    dict[str, WorkerValue],
                    {
                        "active_job_policy": "cancel",
                        "active_job_cancelled": True,
                    },
                ),
            },
            reply_to="shutdown-id",
        )

        def write_worker_frames() -> None:
            """Send the paused stream frame, then acknowledge shutdown."""
            if not request_sent.wait(2):
                return
            send_message(
                worker_input,
                build_packet(
                    "response",
                    "generate_stream",
                    {"ok": True, "stream": True, "value": "chunk"},
                    reply_to=request_id[0],
                    message_id="stream-frame",
                ),
            )
            if not shutdown_sent.wait(2):
                return
            send_message(worker_input, shutdown_ack)
            worker_input.close()

        writer_thread = threading.Thread(target=write_worker_frames)
        writer_thread.start()
        process = ShutdownProcess(worker_output)
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
        proxy._active_request_id = None
        proxy._cancel_target = None
        proxy._cancel_packet_id = None
        proxy._cancel_ack_event = None
        proxy._cancel_ack_result = None
        proxy._cancel_sent = False
        proxy._reader_stop = threading.Event()
        proxy._reader_thread = None
        proxy._reader_error = None
        proxy._response_condition = threading.Condition()
        proxy._pending_reply_ids = set()
        proxy._response_queues = {}
        proxy._response_queue_item_sizes = {}
        proxy._response_queue_bytes = {}
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()

        def send_packet(
            _stream: IO[bytes],
            kind: str,
            _operation: str,
            _data: Optional[dict[str, WorkerValue]] = None,
            *,
            message_id: Optional[str] = None,
            reply_to: Optional[str] = None,
        ) -> str:
            """Stand in for packet transmission while preserving reply registration."""
            del reply_to
            packet_id = message_id or f"{kind}-id"
            if kind in {"request", "shutdown"}:
                with proxy._response_condition:
                    proxy._pending_reply_ids.add(packet_id)
            if kind == "request":
                request_id.append(packet_id)
                request_sent.set()
            if kind == "shutdown":
                shutdown_sent.set()
                return "shutdown-id"
            return packet_id

        proxy._send_packet = mock.Mock(side_effect=send_packet)
        proxy._reader_thread = threading.Thread(
            target=proxy._read_worker_packets,
            args=(process,),
        )
        proxy._reader_thread.start()
        generator = cast(
            Generator[WorkerValue, None, None],
            proxy._stream_request("generate_stream"),
        )

        try:
            self.assertEqual(next(generator), "chunk")
            self.assertFalse(proxy._protocol_lock.acquire(timeout=0.01))
            close_thread = threading.Thread(target=proxy.close)
            close_thread.start()
            close_thread.join(timeout=2)

            self.assertFalse(close_thread.is_alive())
            self.assertTrue(shutdown_sent.is_set())
            self.assertFalse(process.terminated)
            self.assertEqual(process.returncode, 0)
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
            sent_kinds = [call.args[1] for call in proxy._send_packet.call_args_list]
            self.assertEqual(sent_kinds, ["request", "cancel", "shutdown"])
        finally:
            generator.close()
            writer_thread.join(timeout=2)
            with suppress(OSError, ValueError):
                worker_input.close()

    def test_remote_proxy_escalates_when_shutdown_ack_is_missing(self) -> None:
        """Verify a missing shutdown acknowledgement reaches process termination escalation."""
        proxy, process = self._make_shutdown_proxy(waits_before_exit=1)
        proxy._await_shutdown_ack = mock.Mock(
            side_effect=TimeoutError("shutdown acknowledgement timed out")
        )

        with self.assertRaises(TimeoutError):
            proxy.close()

        proxy._await_shutdown_ack.assert_called_once()
        self.assertTrue(process.terminated)
        self.assertEqual(process.returncode, 0)

    def test_remote_proxy_surfaces_failed_shutdown_acknowledgement(self) -> None:
        """Verify a failed shutdown acknowledgement is not hidden by process exit."""
        proxy, process = self._make_shutdown_proxy(shutdown_ok=False)

        with self.assertRaises(BackendError) as context:
            proxy.close()

        self.assertIn(
            "shutdown acknowledgement reported failure", str(context.exception)
        )
        self.assertEqual(process.returncode, 0)
        self.assertFalse(process.terminated)

    def test_remote_proxy_close_is_idempotent(self) -> None:
        """Verify repeated close calls do not send another shutdown packet."""
        proxy, _ = self._make_shutdown_proxy()
        send_packet = cast(mock.Mock, proxy._send_packet)
        proxy._received_message_ids = OrderedDict({"received-id": None})

        proxy.close()
        proxy.close()

        send_packet.assert_called_once()
        self.assertFalse(proxy._received_message_ids)

    def test_remote_proxy_close_releases_all_runtime_state(self) -> None:
        """Verify shutdown releases queued responses, requests, cancellation, and events."""
        proxy, _ = self._make_shutdown_proxy(active_request_id="active-request")
        cancel_event = threading.Event()
        proxy._request_cancellation_states = {
            "active-request": remote._RequestCancellationState(
                request_id="active-request",
                cancel_packet_id="cancel-packet",
                cancel_ack_event=cancel_event,
            )
        }
        proxy._stream_active = threading.Event()
        proxy._stream_active.set()
        proxy._response_condition = threading.Condition()
        proxy._pending_reply_ids = {"active-request", "pending-request"}
        proxy._response_queues = {
            "active-request": deque(
                [{"ok": True, "stream": True, "value": b"retained audio"}]
            ),
            "pending-request": deque([{"ok": True, "value": b"retained result"}]),
        }
        proxy._response_queue_item_sizes = {
            "active-request": deque([128]),
            "pending-request": deque([256]),
        }
        proxy._response_queue_bytes = {
            "active-request": 128,
            "pending-request": 256,
        }
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque(
            [{"kind": "event", "operation": "progress"}],
        )

        proxy.close()

        with proxy._response_condition:
            self.assertFalse(proxy._pending_reply_ids)
            self.assertFalse(proxy._response_queues)
            self.assertFalse(proxy._response_queue_item_sizes)
            self.assertFalse(proxy._response_queue_bytes)
        with proxy._event_condition:
            self.assertFalse(proxy._event_queue)
        with proxy._active_request_lock:
            self.assertIsNone(proxy._active_request_id)
            self.assertFalse(proxy._request_cancellation_states)
        self.assertFalse(proxy._stream_active.is_set())
        self.assertTrue(cancel_event.is_set())

    def test_remote_proxy_close_wakes_and_cleans_inflight_generator(self) -> None:
        """Verify an in-flight generator finalizes after proxy shutdown clears its state."""
        proxy, process = self._make_shutdown_proxy(active_request_id="active-request")
        send_packet = mock.Mock(
            side_effect=("active-request", "cancel-packet", "shutdown-id")
        )
        proxy._send_packet = send_packet
        read_started = threading.Event()
        release_read = threading.Event()

        def hold_stream_read(
            _process: subprocess.Popen[bytes], _reply_to: str
        ) -> remote.WorkerResponse:
            """Hold one stream read until shutdown has released the generator."""
            read_started.set()
            release_read.wait(2)
            raise CEDTSProtocolError("stream read interrupted by shutdown")

        generator = proxy._stream_request("generate_stream")
        generator_errors: list[Exception] = []

        def consume_generator() -> None:
            """Consume the test generator and retain expected shutdown errors."""
            try:
                next(generator)
            except CEDTSError as error:
                generator_errors.append(error)

        consumer = threading.Thread(target=consume_generator)
        with mock.patch.object(
            proxy, "_read_stream_frame", side_effect=hold_stream_read
        ):
            consumer.start()
            self.assertTrue(read_started.wait(1))
            proxy.close()
            release_read.set()
            consumer.join(timeout=2)

        self.assertFalse(consumer.is_alive())
        self.assertEqual(len(generator_errors), 1)
        self.assertFalse(proxy._stream_active.is_set())
        self.assertFalse(proxy._pending_reply_ids)
        self.assertFalse(proxy._response_queues)
        self.assertFalse(proxy._request_cancellation_states)
        self.assertEqual(process.returncode, 0)

    def test_remote_proxy_abort_closes_streams_after_worker_exit(self) -> None:
        """Verify abort closes exited-worker streams and reader threads repeatedly."""
        process = ShutdownProcess(io.BytesIO())
        process.returncode = 0
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(subprocess.Popen[bytes], process)
        proxy._closed = False
        proxy._closing = False
        proxy._close_lock = threading.Lock()
        proxy._binary_input = io.BytesIO()
        proxy._binary_output = io.BytesIO()
        proxy._reader_stop = threading.Event()
        stderr_thread = threading.Thread(target=proxy._reader_stop.wait)
        reader_thread = threading.Thread(target=proxy._reader_stop.wait)
        proxy._stderr_thread = stderr_thread
        proxy._reader_thread = reader_thread
        stderr_thread.start()
        reader_thread.start()

        proxy.abort()
        proxy.abort()

        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertIsNone(proxy._binary_input)
        self.assertIsNone(proxy._binary_output)
        self.assertIsNone(proxy._stderr_thread)
        self.assertIsNone(proxy._reader_thread)
        self.assertFalse(stderr_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())

    def test_remote_proxy_abort_stops_reader_before_termination(self) -> None:
        """Verify intentional process termination cannot be reported as protocol EOF."""
        process = ShutdownProcess(io.BytesIO())
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(subprocess.Popen[bytes], process)
        proxy._closed = False
        proxy._closing = False
        proxy._close_lock = threading.Lock()
        proxy._binary_input = None
        proxy._binary_output = None
        proxy._reader_stop = threading.Event()
        proxy._stderr_thread = None
        proxy._reader_thread = None
        proxy._received_message_ids = OrderedDict()
        termination_observed = False

        def terminate(_process: subprocess.Popen[bytes]) -> None:
            """Record that the reader was stopped before termination."""
            nonlocal termination_observed
            termination_observed = proxy._reader_stop.is_set()
            _process.wait()

        with mock.patch.object(proxy, "_terminate_process", side_effect=terminate):
            proxy.abort()

        self.assertTrue(termination_observed)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_remote_proxy_startup_failure_closes_partial_worker(self) -> None:
        """Verify a partial stream setup terminates the worker and closes ownership."""
        process = ShutdownProcess(io.BytesIO())
        binary_input = io.BytesIO()
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BACKEND_MANIFESTS["mini"]
        proxy._process = None
        proxy._binary_input = None
        proxy._binary_output = None
        proxy._stderr_thread = None
        proxy._reader_thread = None
        proxy._reader_stop = threading.Event()

        backend_environment = BackendEnvironment(
            BACKEND_MANIFESTS["mini"], Path("C:/backend")
        )
        with (
            mock.patch.object(remote.subprocess, "Popen", return_value=process),
            mock.patch.object(
                remote.os,
                "fdopen",
                side_effect=[binary_input, OSError("binary output failed")],
            ),
            self.assertRaises(OSError),
        ):
            proxy._start_worker(
                backend_environment,
                lambda msg, severity="info", *, loglevel="info": None,
                {},
            )

        self.assertTrue(process.terminated)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertTrue(binary_input.closed)
        self.assertIsNone(proxy._process)
        self.assertIsNone(proxy._binary_input)
        self.assertIsNone(proxy._binary_output)

    def test_remote_stream_disconnect_sends_cancel_and_drains_terminal_frames(
        self,
    ) -> None:
        """Verify closing a consumer generator cancels its request without deadlocking."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": True, "stream": True, "value": "chunk"},
                reply_to="active-request",
                message_id="stream-frame",
            ),
        )
        send_message(
            stream,
            build_packet(
                "cancel_ack",
                "cancel",
                {
                    "ok": True,
                    "cancelled": True,
                    "target_message_id": "active-request",
                },
                reply_to="cancel-packet",
                message_id="cancel-ack",
            ),
        )
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": False, "cancelled": True, "done": True},
                reply_to="active-request",
                message_id="stream-done",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BackendManifest("fake", "tts", (), "module", "Backend")
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(
                stdin=io.BytesIO(),
                stdout=stream,
                poll=lambda: None,
            ),
        )
        proxy._protocol_lock = threading.Lock()
        proxy._log_callback = mock.Mock()
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        proxy._binary_output = None
        proxy._send_packet = mock.Mock(side_effect=("active-request", "cancel-packet"))

        generator = cast(
            Generator[WorkerValue, None, None],
            proxy._stream_request("generate_stream"),
        )
        self.assertEqual(next(generator), "chunk")
        generator.close()

        self.assertEqual(proxy._send_packet.call_count, 2)
        proxy._send_packet.assert_called_with(
            proxy._process.stdin,
            "cancel",
            "cancel",
            {"target_message_id": "active-request"},
            message_id=mock.ANY,
        )

    def test_remote_stream_disconnect_escalates_after_drain_timeout(self) -> None:
        """Verify an ignored cancellation cannot leave the worker alive indefinitely."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": True, "stream": True, "value": "chunk"},
                reply_to="active-request",
                message_id="stream-frame",
            ),
        )
        stream.seek(0)
        process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(
                stdin=io.BytesIO(),
                stdout=stream,
                poll=lambda: None,
            ),
        )
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BackendManifest("fake", "tts", (), "module", "Backend")
        proxy._process = process
        proxy._protocol_lock = threading.Lock()
        proxy._log_callback = mock.Mock()
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        proxy._binary_output = None
        proxy._send_packet = mock.Mock(side_effect=("active-request", "cancel-packet"))
        proxy._terminate_process = mock.Mock()

        generator = cast(
            Generator[WorkerValue, None, None],
            proxy._stream_request("generate_stream"),
        )
        self.assertEqual(next(generator), "chunk")
        with mock.patch.object(proxy, "_drain_stream", return_value=False):
            generator.close()

        proxy._terminate_process.assert_called_once_with(process)

    def test_remote_stream_drain_passes_a_bounded_timeout(self) -> None:
        """Verify stream draining stops when the cancellation deadline expires."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdout=io.BytesIO()),
        )
        read_response = mock.Mock(side_effect=TimeoutError("drain timed out"))
        proxy._read_response = read_response

        with mock.patch.object(remote, "_STREAM_DRAIN_TIMEOUT_SECONDS", 0.01):
            drained = proxy._drain_stream(process, "request-id")

        self.assertFalse(drained)
        read_response.assert_called_once()
        self.assertGreater(read_response.call_args.kwargs["timeout"], 0)

    def test_worker_unload_closes_runtime_models(self) -> None:
        """Verify worker unload releases models stored outside the backend object."""

        class FakeModel:
            """Runtime stand-in that records whether the worker closed it."""

            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                """Record worker-owned runtime cleanup."""
                self.closed = True

        class FakeBackend:
            """Backend stand-in whose unload hook does not own the model table."""

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                """Leave model-table cleanup to the worker.

                Args:
                    release_cuda_cache: Whether the worker should release cached accelerator blocks.
                """

            @staticmethod
            def load_model(**_kwargs: object) -> FakeModel:
                """Return one runtime owned by the worker model table."""
                return FakeModel()

        models: dict[int, BackendModel] = {}
        loaded, next_model_id = worker._run_request(
            cast(_BackendRuntime, FakeBackend()),
            {"operation": "load_model", "arguments": {"model_id": "fake"}},
            models,
            1,
            io.BytesIO(),
        )
        model = cast(FakeModel, models[1])

        unloaded, _ = worker._run_request(
            cast(_BackendRuntime, FakeBackend()),
            {"operation": "unload_model", "arguments": {}},
            models,
            next_model_id,
            io.BytesIO(),
        )

        assert loaded["ok"]
        assert unloaded["ok"]
        assert model.closed
        assert not models

    def test_worker_unload_ignores_broken_model_attribute_lookup(self) -> None:
        """Verify torn-down Torch-like models cannot break worker cleanup."""

        class PartiallyTornDownModel:
            """Model stand-in whose Torch-style lookup is no longer usable."""

            def __getattribute__(self, name: str) -> object:
                if name in {"close", "unload"}:
                    raise TypeError("argument of type 'NoneType' is not iterable")
                return super().__getattribute__(name)

        models: dict[int, BackendModel] = {1: PartiallyTornDownModel()}
        worker._release_worker_models(models)

        assert not models

    def test_remote_proxy_handles_fatal_frames_out_of_band(self) -> None:
        """Verify fatal worker notifications do not consume the response frame."""
        stream = io.BytesIO()
        send_message(stream, build_packet("event", "fatal", {"fatal": True}))
        send_message(
            stream,
            build_packet("response", "describe", {"ok": True, "value": "ready"}),
        )
        stream.seek(0)
        callback = mock.Mock()
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._fatal_callback = callback
        proxy._received_message_ids = OrderedDict()

        response = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream))
        )

        callback.assert_called_once_with()
        assert response["value"] == "ready"

    def _make_packet_reader_proxy(
        self,
        *,
        event_callback: Optional[Callable[[WorkerMessage], None]] = None,
        fatal_callback: Optional[Callable[[], None]] = None,
    ) -> tuple[remote.RemoteBackendProxy, IO[bytes], IO[bytes]]:
        """Build a proxy whose packet reader consumes a POSIX pipe."""
        reader_fd, writer_fd = os.pipe()
        reader = os.fdopen(reader_fd, "rb", buffering=0)
        writer = os.fdopen(writer_fd, "wb", buffering=0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(pid=1234, stdout=reader),
        )
        proxy._binary_output = None
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        proxy._log_callback = mock.Mock()
        proxy._fatal_callback = fatal_callback
        proxy._event_callback = event_callback
        proxy._reader_stop = threading.Event()
        proxy._reader_thread = None
        proxy._reader_error = None
        proxy._response_condition = threading.Condition()
        proxy._pending_reply_ids = set()
        proxy._response_queues = {}
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque(maxlen=256)
        proxy._start_packet_reader()
        return proxy, reader, writer

    def test_remote_proxy_reports_eof_to_an_active_request_immediately(self) -> None:
        """Verify an EOF wakes the active request instead of waiting for its deadline."""
        proxy, reader, writer = self._make_packet_reader_proxy()
        process = cast(SimpleNamespace, proxy._process)
        process.stdin = io.BytesIO()
        process.poll = lambda: None
        proxy._protocol_lock = threading.Lock()
        request_sent = threading.Event()

        def send_packet(*_args: object, **_kwargs: object) -> str:
            """Signal that the active request reached the stand-in worker."""
            request_sent.set()
            return "request-id"

        proxy._send_packet = mock.Mock(side_effect=send_packet)
        proxy.abort = mock.Mock()
        errors: list[Exception] = []

        def request() -> None:
            """Wait for the active request to observe the closed worker stream."""
            try:
                proxy._request(
                    "load_model",
                    response_timeout=900.0,
                    model_id="model",
                )
            except Exception as error:
                errors.append(error)

        request_thread = threading.Thread(target=request)
        request_thread.start()
        self.assertTrue(request_sent.wait(timeout=1))
        writer.close()
        request_thread.join(timeout=1)

        try:
            self.assertFalse(request_thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], CEDTSEOFError)
            log_messages = [
                str(call.args[0])
                for call in cast(mock.Mock, proxy._log_callback).call_args_list
                if call.args
            ]
            self.assertTrue(
                any(
                    "CEDTS transport error operation=load_model" in message
                    and "unexpected EOF" in message
                    for message in log_messages
                )
            )
        finally:
            proxy._reader_stop.set()
            with suppress(OSError, ValueError):
                reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=1)

    def test_remote_proxy_dispatches_idle_worker_events(self) -> None:
        """Verify fatal notifications and correlated worker events reach consumers."""
        event_callback = mock.Mock()
        fatal_callback = mock.Mock()
        proxy, reader, writer = self._make_packet_reader_proxy(
            event_callback=event_callback,
            fatal_callback=fatal_callback,
        )
        request_id = "active-event-request"
        with proxy._response_condition:
            proxy._pending_reply_ids.add(request_id)
        try:
            send_message(
                writer,
                build_packet("event", "fatal", {"fatal": True}),
            )
            send_message(
                writer,
                build_packet(
                    "progress",
                    "load_model",
                    {"step": 2},
                    reply_to=request_id,
                ),
            )
            send_message(
                writer,
                build_packet(
                    "callback",
                    "ready",
                    {"state": "ready"},
                    reply_to=request_id,
                ),
            )

            events = [proxy.get_worker_event(timeout=2) for _ in range(3)]
            self.assertEqual(
                [event["kind"] for event in events if event is not None],
                ["event", "progress", "callback"],
            )
            self.assertEqual(event_callback.call_count, 3)
            fatal_callback.assert_called_once_with()
        finally:
            proxy._reader_stop.set()
            writer.close()
            reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)

    def test_remote_proxy_treats_shutdown_eof_as_expected(self) -> None:
        """Verify a shutdown acknowledgement stops the reader before worker EOF."""
        proxy, reader, writer = self._make_packet_reader_proxy()
        reply_to = "shutdown-request"
        with proxy._response_condition:
            proxy._pending_reply_ids.add(reply_to)
        try:
            send_message(
                writer,
                build_packet(
                    "shutdown_ack",
                    "shutdown",
                    {"ok": True, "value": {}},
                    reply_to=reply_to,
                ),
            )

            response = proxy._read_response(
                cast(subprocess.Popen[bytes], proxy._process),
                reply_to,
                timeout=2,
            )

            self.assertTrue(response["ok"])
            self.assertTrue(proxy._reader_stop.is_set())
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)
                self.assertFalse(reader_thread.is_alive())
            self.assertIsNone(proxy._reader_error)
            log_messages = [
                str(call.args[0])
                for call in cast(mock.Mock, proxy._log_callback).call_args_list
                if call.args
            ]
            self.assertFalse(
                any(
                    "worker packet reader failed" in message for message in log_messages
                )
            )
        finally:
            proxy._reader_stop.set()
            writer.close()
            reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)

    def test_remote_proxy_rejects_uncorrelated_worker_progress_and_callbacks(
        self,
    ) -> None:
        """Verify progress and callback packets require an active request correlation."""
        for kind, operation, data in (
            ("progress", "load_model", {"step": 1}),
            ("callback", "ready", {"state": "ready"}),
        ):
            for reply_to in (None, "unknown-request"):
                with self.subTest(kind=kind, reply_to=reply_to):
                    event_callback = mock.Mock()
                    proxy, reader, writer = self._make_packet_reader_proxy(
                        event_callback=event_callback,
                    )
                    try:
                        send_message(
                            writer,
                            build_packet(
                                kind,
                                operation,
                                data,
                                reply_to=reply_to,
                            ),
                        )
                        reader_error: Optional[Exception] = None
                        for _ in range(200):
                            with proxy._response_condition:
                                reader_error = proxy._reader_error
                            if reader_error is not None:
                                break
                            threading.Event().wait(0.01)

                        self.assertIsInstance(reader_error, CEDTSError)
                        event_callback.assert_not_called()
                        with proxy._event_condition:
                            self.assertFalse(proxy._event_queue)
                    finally:
                        proxy._reader_stop.set()
                        writer.close()
                        reader.close()
                        reader_thread = proxy._reader_thread
                        if reader_thread is not None:
                            reader_thread.join(timeout=2)

    def test_remote_proxy_event_waiter_ignores_spurious_notifications(self) -> None:
        """Verify a spurious event-condition notification does not end the wait."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()
        proxy._response_condition = threading.Condition()
        proxy._reader_error = None
        event = build_packet("progress", "load_model", {"step": 1})

        def notify_spuriously() -> None:
            threading.Event().wait(0.02)
            with proxy._event_condition:
                proxy._event_condition.notify_all()
            threading.Event().wait(0.02)
            with proxy._event_condition:
                proxy._event_queue.append(event)
                proxy._event_condition.notify_all()

        notifier = threading.Thread(target=notify_spuriously)
        notifier.start()
        received = proxy.get_worker_event(timeout=2)
        notifier.join(timeout=2)

        self.assertEqual(received, event)
        self.assertFalse(notifier.is_alive())

    def test_remote_proxy_event_waiter_returns_none_on_timeout(self) -> None:
        """Verify an idle event waiter distinguishes timeout from reader failure."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()
        proxy._response_condition = threading.Condition()
        proxy._reader_error = None

        started = time.monotonic()
        received = proxy.get_worker_event(timeout=0.05)

        self.assertIsNone(received)
        self.assertGreaterEqual(time.monotonic() - started, 0.04)

    def test_remote_proxy_event_waiter_surfaces_reader_failure(self) -> None:
        """Verify a worker reader failure wakes and fails an idle event waiter."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()
        proxy._response_condition = threading.Condition()
        reader_error = CEDTSProtocolError("worker reader failed")
        proxy._reader_error = None
        waiter_started = threading.Event()
        errors: list[Exception] = []

        def wait_for_event() -> None:
            waiter_started.set()
            try:
                proxy.get_worker_event(timeout=2)
            except CEDTSError as error:
                errors.append(error)

        waiter = threading.Thread(target=wait_for_event)
        waiter.start()
        self.assertTrue(waiter_started.wait(timeout=1))
        threading.Event().wait(0.02)
        with proxy._response_condition:
            proxy._reader_error = reader_error
        with proxy._event_condition:
            proxy._event_condition.notify_all()
        waiter.join(timeout=2)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0], reader_error)

    def test_remote_proxy_dispatches_progress_while_waiting_for_response(self) -> None:
        """Verify the reader routes progress independently of a response waiter."""
        event_callback = mock.Mock()
        progress_callback = mock.Mock()
        proxy, reader, writer = self._make_packet_reader_proxy(
            event_callback=event_callback,
        )
        proxy.bind_progress(progress_callback)
        request_id = "active-request"
        with proxy._response_condition:
            proxy._pending_reply_ids.add(request_id)
        try:
            send_message(
                writer,
                build_packet(
                    "progress",
                    "generate_stream",
                    {"step": 1},
                    reply_to=request_id,
                ),
            )
            send_message(
                writer,
                build_packet(
                    "response",
                    "generate_stream",
                    {"ok": True, "done": True},
                    reply_to=request_id,
                ),
            )

            response = proxy._read_response(
                cast(subprocess.Popen[bytes], proxy._process),
                request_id,
                timeout=2,
            )
            progress = proxy.get_worker_event(timeout=2)
            self.assertTrue(response["done"])
            self.assertIsNotNone(progress)
            self.assertEqual(progress["kind"], "progress")
            self.assertEqual(progress["reply_to"], request_id)
            event_callback.assert_called_once()
            progress_callback.assert_called_once_with(1.0, None)
        finally:
            proxy._reader_stop.set()
            writer.close()
            reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)

    def test_remote_proxy_bounds_paused_stream_response_queue_items(self) -> None:
        """Verify a paused stream flood fails instead of growing without bound."""
        proxy, reader, writer = self._make_packet_reader_proxy()
        request_id = "paused-stream"
        with proxy._response_condition:
            proxy._pending_reply_ids.add(request_id)
        try:
            with mock.patch.object(remote, "_MAX_RESPONSE_QUEUE_ITEMS", 2):
                for index in range(3):
                    send_message(
                        writer,
                        build_packet(
                            "response",
                            "generate_stream",
                            {
                                "ok": True,
                                "stream": True,
                                "value": f"chunk-{index}",
                            },
                            reply_to=request_id,
                            message_id=f"frame-{index}",
                        ),
                    )
                for _ in range(200):
                    with proxy._response_condition:
                        reader_error = proxy._reader_error
                    if reader_error is not None:
                        break
                    threading.Event().wait(0.01)

            self.assertIsInstance(reader_error, CEDTSError)
            with proxy._response_condition:
                self.assertEqual(len(proxy._response_queues[request_id]), 2)
                queued_bytes = proxy._response_queue_bytes[request_id]
            self.assertGreater(queued_bytes, 0)

            process = cast(subprocess.Popen[bytes], proxy._process)
            self.assertEqual(
                proxy._read_response(process, request_id)["value"], "chunk-0"
            )
            self.assertEqual(
                proxy._read_response(process, request_id)["value"], "chunk-1"
            )
            with proxy._response_condition:
                self.assertNotIn(request_id, proxy._response_queues)
                self.assertNotIn(request_id, proxy._response_queue_item_sizes)
                self.assertNotIn(request_id, proxy._response_queue_bytes)
        finally:
            proxy._reader_stop.set()
            writer.close()
            reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)

    def test_remote_proxy_bounds_paused_stream_response_queue_bytes(self) -> None:
        """Verify one oversized queued response is rejected deterministically."""
        proxy, reader, writer = self._make_packet_reader_proxy()
        request_id = "oversized-paused-stream"
        with proxy._response_condition:
            proxy._pending_reply_ids.add(request_id)
        try:
            with mock.patch.object(remote, "_MAX_RESPONSE_QUEUE_BYTES", 32):
                send_message(
                    writer,
                    build_packet(
                        "response",
                        "generate_stream",
                        {
                            "ok": True,
                            "stream": True,
                            "value": "a response larger than the queue limit",
                        },
                        reply_to=request_id,
                        message_id="oversized-frame",
                    ),
                )
                for _ in range(200):
                    with proxy._response_condition:
                        reader_error = proxy._reader_error
                    if reader_error is not None:
                        break
                    threading.Event().wait(0.01)

            self.assertIsInstance(reader_error, CEDTSError)
            with proxy._response_condition:
                self.assertNotIn(request_id, proxy._response_queues)
                self.assertNotIn(request_id, proxy._response_queue_item_sizes)
                self.assertNotIn(request_id, proxy._response_queue_bytes)
        finally:
            proxy._reader_stop.set()
            writer.close()
            reader.close()
            reader_thread = proxy._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=2)

    def test_worker_handshake_negotiates_the_cedts_capabilities(self) -> None:
        """Verify the worker accepts a compatible hello and rejects requirements it lacks."""
        self.assertEqual(remote.CEDTS_VERSION, (1, 1))
        hello = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [[1, 1]],
                    "capabilities": remote.CORE_CAPABILITIES,
                    "required_capabilities": {"streaming": True},
                },
            ),
        )
        self.assertEqual(hello["cedts_version"], [1, 1])
        negotiated = worker._negotiate_hello(hello)
        self.assertEqual(negotiated["streaming"], True)
        self.assertEqual(negotiated["cancellation"], True)

        unsupported_version = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [[2, 1]],
                    "capabilities": remote.CORE_CAPABILITIES,
                    "required_capabilities": {"streaming": True},
                },
            ),
        )
        with self.assertRaises(CEDTSError):
            worker._negotiate_hello(unsupported_version)

        incompatible = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [[1, 1]],
                    "capabilities": remote.CORE_CAPABILITIES,
                    "required_capabilities": {"callback": True},
                },
            ),
        )
        with self.assertRaises(CEDTSError):
            worker._negotiate_hello(incompatible)

    def test_cedts_handshake_retains_smaller_peer_frame_limits(self) -> None:
        """Verify handshake negotiation retains the minimum transport bounds."""
        offered_capabilities = dict(remote.CORE_CAPABILITIES)
        offered_capabilities.update(
            {
                "max_control_frame_size": 768,
                "max_binary_frame_size": 64,
                "max_aggregate_payload_size": 8,
                "max_payload_descriptors": 1,
                "max_json_depth": 8,
                "max_string_length": 1024,
                "max_collection_entries": 16,
            }
        )
        hello = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [[1, 1]],
                    "capabilities": offered_capabilities,
                },
            ),
        )

        negotiated = worker._negotiate_hello(hello)
        limits = limits_from_capabilities(negotiated)
        self.assertEqual(
            limits,
            CEDTSLimits(
                max_control_frame_size=768,
                max_binary_frame_size=64,
                max_aggregate_payload_size=8,
                max_payload_descriptors=1,
                max_json_depth=8,
                max_string_length=1024,
                max_collection_entries=16,
            ),
        )

        oversized_control = build_packet(
            "request",
            "load_model",
            cast(
                dict[str, WorkerValue],
                {"arguments": {"model_id": "model-id-" + "x" * 900}},
            ),
        )
        with self.assertRaises(CEDTSError):
            send_message(io.BytesIO(), oversized_control, limits=limits)
        control_stream = io.BytesIO()
        send_message(control_stream, oversized_control)
        control_stream.seek(0)
        with self.assertRaises(CEDTSError):
            receive_message(control_stream, limits=limits)

        oversized_payload = WorkerPayload(
            {
                "id": "payload-1",
                "media_type": "application/octet-stream",
                "byte_length": 9,
            },
            b"x" * 9,
        )
        with self.assertRaises(CEDTSError):
            send_payloads(io.BytesIO(), [oversized_payload], limits=limits)
        binary_stream = io.BytesIO()
        send_payloads(binary_stream, [oversized_payload])
        binary_stream.seek(0)
        with self.assertRaises(CEDTSError):
            receive_payloads(
                binary_stream,
                [oversized_payload.descriptor],
                limits=limits,
            )

        with self.assertRaises(CEDTSError):
            validate_payload_descriptors(
                [
                    {
                        "id": "payload-1",
                        "media_type": "application/octet-stream",
                        "byte_length": 0,
                    },
                    {
                        "id": "payload-2",
                        "media_type": "application/octet-stream",
                        "byte_length": 0,
                    },
                ],
                limits=limits,
            )

        error_stream = io.BytesIO()
        with mock.patch.object(worker.traceback, "print_exc"):
            worker._send_error(
                error_stream,
                io.BytesIO(),
                "handshake",
                CEDTSProtocolError("unsupported CEDTS version"),
                reply_to="hello-id",
            )
        error_stream.seek(0)
        error_packet = receive_message(error_stream)
        self.assertEqual(error_packet["kind"], "error")
        self.assertEqual(error_packet["reply_to"], "hello-id")

    def test_remote_proxy_requires_hello_ack_then_ready(self) -> None:
        """Verify core handshake packets are correlated before backend requests begin."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "hello_ack",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "cedts_version": [1, 1],
                        "capabilities": remote.CORE_CAPABILITIES,
                    },
                ),
                reply_to="hello-id",
            ),
        )
        send_message(
            stream,
            build_packet(
                "ready",
                "ready",
                {"capabilities": remote.CORE_CAPABILITIES},
                reply_to="hello-id",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), stdout=stream),
        )
        proxy._binary_input = None
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        with mock.patch.object(proxy, "_send_packet", return_value="hello-id"):
            proxy._handshake()
        self.assertEqual(proxy._negotiated_capabilities["streaming"], True)

    def test_remote_proxy_forwards_fatal_events_during_handshake(self) -> None:
        """Verify fatal worker events remain out-of-band during capability negotiation."""
        stream = io.BytesIO()
        for packet in (
            build_packet(
                "hello_ack",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "cedts_version": [1, 1],
                        "capabilities": remote.CORE_CAPABILITIES,
                    },
                ),
                reply_to="hello-id",
            ),
            build_packet("event", "fatal", {"fatal": True}),
            build_packet(
                "ready",
                "ready",
                {"capabilities": remote.CORE_CAPABILITIES},
                reply_to="hello-id",
            ),
        ):
            send_message(stream, packet)
        stream.seek(0)
        fatal_callback = mock.Mock()
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), stdout=stream),
        )
        proxy._binary_input = None
        proxy._received_message_ids = OrderedDict()
        proxy._event_condition = threading.Condition()
        proxy._event_queue = deque()
        proxy._fatal_callback = fatal_callback
        proxy._log_callback = mock.Mock()
        with mock.patch.object(proxy, "_send_packet", return_value="hello-id"):
            proxy._handshake()

        fatal_callback.assert_called_once_with()
        self.assertEqual(proxy._negotiated_capabilities["streaming"], True)

    def test_remote_proxy_surfaces_handshake_error_before_hello_ack(self) -> None:
        """Verify a worker error before hello acknowledgement remains a backend error."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "error",
                "handshake",
                {
                    "ok": False,
                    "error": "backend dependency missing",
                    "error_type": "ImportError",
                },
                reply_to="hello-id",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), stdout=stream),
        )
        proxy._binary_input = None
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        with (
            mock.patch.object(proxy, "_send_packet", return_value="hello-id"),
            self.assertRaises(BackendError) as context,
        ):
            proxy._handshake()

        error = cast(BackendError, context.exception)
        self.assertEqual(error.error_code, "backend_worker_error")
        self.assertEqual(error.error_type, "ImportError")
        self.assertIn("backend dependency missing", str(context.exception))

    def test_remote_proxy_surfaces_handshake_error_before_ready(self) -> None:
        """Verify a worker error after hello acknowledgement remains a backend error."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "hello_ack",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "cedts_version": [1, 1],
                        "capabilities": remote.CORE_CAPABILITIES,
                    },
                ),
                reply_to="hello-id",
            ),
        )
        send_message(
            stream,
            build_packet(
                "error",
                "handshake",
                {
                    "ok": False,
                    "error": "backend dependency missing",
                    "error_type": "ImportError",
                },
                reply_to="hello-id",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), stdout=stream),
        )
        proxy._binary_input = None
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        with (
            mock.patch.object(proxy, "_send_packet", return_value="hello-id"),
            self.assertRaises(BackendError) as context,
        ):
            proxy._handshake()

        error = cast(BackendError, context.exception)
        self.assertEqual(error.error_code, "backend_worker_error")
        self.assertEqual(error.error_type, "ImportError")
        self.assertIn("backend dependency missing", str(context.exception))

    def test_remote_proxy_bounds_packet_id_replay_window(self) -> None:
        """Verify recent packet IDs reject duplicates while old IDs are pruned."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()

        for index in range(remote._MESSAGE_ID_REPLAY_WINDOW + 1):
            proxy._register_packet({"message_id": f"packet-{index}"})

        self.assertEqual(
            len(proxy._received_message_ids), remote._MESSAGE_ID_REPLAY_WINDOW
        )
        with self.assertRaises(CEDTSError):
            proxy._register_packet(
                {"message_id": f"packet-{remote._MESSAGE_ID_REPLAY_WINDOW}"}
            )
        proxy._register_packet({"message_id": "packet-0"})

    def test_worker_bounds_packet_id_replay_window(self) -> None:
        """Verify the worker rejects recent duplicates and prunes old IDs."""
        received_message_ids: OrderedDict[str, None] = OrderedDict()

        for index in range(worker._MESSAGE_ID_REPLAY_WINDOW + 1):
            self.assertTrue(
                worker._remember_message_id(received_message_ids, f"packet-{index}")
            )

        self.assertEqual(len(received_message_ids), worker._MESSAGE_ID_REPLAY_WINDOW)
        self.assertFalse(
            worker._remember_message_id(
                received_message_ids,
                f"packet-{worker._MESSAGE_ID_REPLAY_WINDOW}",
            )
        )
        self.assertTrue(worker._remember_message_id(received_message_ids, "packet-0"))

    def test_remote_proxy_retains_smaller_handshake_frame_limits(self) -> None:
        """Verify the proxy applies peer-advertised limits after handshake."""
        small_capabilities = dict(remote.CORE_CAPABILITIES)
        small_capabilities.update(
            {
                "max_control_frame_size": 768,
                "max_binary_frame_size": 64,
                "max_aggregate_payload_size": 8,
                "max_payload_descriptors": 1,
                "max_json_depth": 8,
                "max_string_length": 1024,
                "max_collection_entries": 16,
            }
        )
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "hello_ack",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "cedts_version": [1, 1],
                        "capabilities": small_capabilities,
                    },
                ),
                reply_to="hello-id",
            ),
        )
        send_message(
            stream,
            build_packet(
                "ready",
                "ready",
                {"capabilities": small_capabilities},
                reply_to="hello-id",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), stdout=stream),
        )
        proxy._binary_input = None
        proxy._binary_output = None
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        with mock.patch.object(proxy, "_send_packet", return_value="hello-id"):
            proxy._handshake()

        self.assertEqual(proxy._cedts_limits.max_control_frame_size, 768)
        self.assertEqual(proxy._cedts_limits.max_binary_frame_size, 64)
        self.assertEqual(proxy._cedts_limits.max_aggregate_payload_size, 8)
        self.assertEqual(proxy._cedts_limits.max_payload_descriptors, 1)
