# SPDX-License-Identifier: Apache-2.0
"""Tests for isolated backend environment metadata and installation."""

import io
import os
import sys
import time
import select
import argparse
import threading
import subprocess
from types import SimpleNamespace
from typing import Optional, cast
from pathlib import Path
from unittest import mock
from contextlib import suppress
from collections import OrderedDict, deque

import numpy as np

from celune.cedts import remote, worker
from celune.exceptions import (
    CEDTSError,
    BackendError,
)
from celune.backends.tts import resolve_backend
from celune.typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerResponse,
    WorkerControlMessage,
    WorkerPayloadDescriptor,
)
from celune.cedts.protocol import (
    build_packet,
    send_message,
    send_payloads,
    encode_message,
    receive_message,
    receive_payloads,
)
from celune.typing.backends import (
    _BackendRuntime,
)
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendManifest,
    BackendEnvironment,
)
from celune.dataclasses.pipeline import VoiceConversionRequest

from .backend_processes import send_no_payload_packet
from .platform import LINUX_ONLY, WINDOWS_ONLY
from .backend_proxy_lifecycle import TestBackendEnvironment as _TestBackendEnvironment


class TestBackendEnvironment(_TestBackendEnvironment):
    """Exercise the CEDTS worker wire protocol in subprocesses."""

    @LINUX_ONLY
    def test_worker_subprocess_completes_cedts_lifecycle_without_backend_dependencies(
        self,
    ) -> None:
        """Verify a real worker process performs handshake, request, and shutdown."""
        child_code = """
from celune.cedts import worker


class FakeBackend:
    name = "subprocess-fake"
    chunk_rate = 0.0
    supported_languages = ()
    voice_models = None
    default_voice = None
    model_name = None
    voices = []
    clone_model_id = None
    uses_voice_bundles = False
    max_new_tokens = 512
    is_fake = True

    def load_model(self, model_id):
        print("backend load diagnostic", flush=True)
        return model_id

    def unload_model(self, release_cuda_cache=True):
        del release_cuda_cache


def fake_load_backend(manifest, log, fatal, kwargs):
    del manifest, log, fatal, kwargs
    return FakeBackend()


worker._load_backend = fake_load_backend
raise SystemExit(worker.main())
"""
        worker_binary_input, core_binary_output = os.pipe()
        core_binary_input, worker_binary_output = os.pipe()
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            process = subprocess.Popen(  # pylint: disable=R1732
                [
                    sys.executable,
                    "-c",
                    child_code,
                    "--backend",
                    "mini",
                    "--backend-kwargs",
                    "{}",
                    "--binary-input-fd",
                    str(worker_binary_input),
                    "--binary-output-fd",
                    str(worker_binary_output),
                ],
                cwd=Path.cwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                pass_fds=(worker_binary_input, worker_binary_output),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            worker_binary_input = -1
            worker_binary_output = -1
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None

            def receive_worker_message() -> WorkerMessage:
                """Read one worker packet within a bounded test deadline."""
                assert process is not None
                assert process.stdout is not None
                ready, _, _ = select.select([process.stdout], [], [], 10.0)
                if not ready:
                    self.fail("worker did not send the next CEDTS packet")
                return receive_message(process.stdout)

            hello = build_packet(
                "hello",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "versions": [1],
                        "capabilities": remote.CORE_CAPABILITIES,
                        "required_capabilities": {"streaming": True},
                    },
                ),
                message_id="hello-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, hello)
            hello_ack = receive_worker_message()
            ready = receive_worker_message()
            self.assertEqual(hello_ack["kind"], "hello_ack")
            self.assertEqual(hello_ack["reply_to"], "hello-request")
            self.assertEqual(ready["kind"], "ready")
            self.assertEqual(ready["reply_to"], "hello-request")

            request = build_packet(
                "request",
                "describe",
                {"arguments": {}},
                message_id="describe-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, request)
            response = receive_worker_message()
            self.assertEqual(response["kind"], "response")
            self.assertEqual(response["reply_to"], "describe-request")
            self.assertTrue(cast(dict, response["data"])["ok"])

            load_request = build_packet(
                "request",
                "load_model",
                cast(
                    dict[str, WorkerValue],
                    {"arguments": {"model_id": "fake/model"}},
                ),
                message_id="load-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, load_request)
            load_response = receive_worker_message()
            self.assertEqual(load_response["kind"], "response")
            self.assertEqual(load_response["reply_to"], "load-request")
            self.assertEqual(
                cast(dict, load_response["data"])["value"],
                1,
            )

            shutdown = build_packet(
                "shutdown",
                "shutdown",
                {"active_job_policy": "cancel"},
                message_id="shutdown-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, shutdown)
            shutdown_ack = receive_worker_message()
            self.assertEqual(shutdown_ack["kind"], "shutdown_ack")
            self.assertEqual(shutdown_ack["reply_to"], "shutdown-request")
            self.assertEqual(
                cast(dict, shutdown_ack["data"])["value"]["active_job_cancelled"],
                False,
            )
            stderr_output = b""
            try:
                _, stderr_output = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                self.fail("worker did not exit after shutdown acknowledgement")
            self.assertEqual(process.returncode, 0)
            self.assertIn(
                "backend load diagnostic",
                stderr_output.decode("utf-8", errors="replace"),
            )
        finally:
            for descriptor in (
                worker_binary_input,
                worker_binary_output,
                core_binary_output,
                core_binary_input,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

    def test_worker_flushes_redirected_backend_output_before_exit(self) -> None:
        """Verify redirected backend output is visible before worker shutdown."""
        child_code = """
from celune.cedts import worker


protocol_stream = worker._detach_protocol_stream()
print("backend diagnostic")
protocol_stream.write(b"ready\\n")
protocol_stream.flush()
input()
protocol_stream.close()
"""
        process = subprocess.Popen(  # pylint: disable=R1732
            [sys.executable, "-c", child_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_lines: list[bytes] = []

        def read_stderr() -> None:
            """Read the child diagnostic without waiting for child exit."""
            assert process.stderr is not None
            stderr_lines.append(process.stderr.readline())

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        try:
            self.assertEqual(process.stdout.readline(), b"ready\n")
            stderr_thread.join(timeout=5)
            self.assertFalse(stderr_thread.is_alive())
            self.assertEqual(
                [line.rstrip(b"\r\n") for line in stderr_lines],
                [b"backend diagnostic"],
            )
            process.stdin.write(b"\n")
            process.stdin.flush()
            process.wait(timeout=10)
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                with suppress(Exception):
                    process.stdin.write(b"\n")
                    process.stdin.flush()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2)
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def _run_worker_stream_shutdown_policy(
        self, policy: str, *, ignore_cancellation: bool = False
    ) -> tuple[dict[str, WorkerValue], list[WorkerControlMessage]]:
        """Run an active stream through one CEDTS shutdown policy."""
        child_code = """
import time
from celune.cedts import worker


class FakeBackend:
    name = "subprocess-shutdown-fake"
    chunk_rate = 0.0
    supported_languages = ()
    voice_models = None
    default_voice = None
    model_name = None
    voices = []
    clone_model_id = None
    uses_voice_bundles = False
    max_new_tokens = 512
    is_fake = True

    def load_model(self, model_id):
        return model_id

    def generate_stream(self, model):
        del model
        yield {"audio": [0.0], "sample_rate": 48000}
        if IGNORE_CANCELLATION:
            while True:
                time.sleep(0.01)
        time.sleep(3.0)
        yield {"audio": [0.0], "sample_rate": 48000}

    def unload_model(self, release_cuda_cache=True):
        del release_cuda_cache


def fake_load_backend(manifest, log, fatal, kwargs):
    del manifest, log, fatal, kwargs
    return FakeBackend()


worker._load_backend = fake_load_backend
if IGNORE_CANCELLATION:
    worker._SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0.1
raise SystemExit(worker.main())
""".replace("IGNORE_CANCELLATION", repr(ignore_cancellation))
        worker_binary_input, core_binary_output = os.pipe()
        core_binary_input, worker_binary_output = os.pipe()
        core_binary_input_stream = os.fdopen(core_binary_input, "rb", buffering=0)
        core_binary_input = -1
        process: Optional[subprocess.Popen[bytes]] = None
        packets: list[WorkerControlMessage] = []
        try:
            process = subprocess.Popen(  # pylint: disable=R1732
                [
                    sys.executable,
                    "-c",
                    child_code,
                    "--backend",
                    "mini",
                    "--backend-kwargs",
                    "{}",
                    "--binary-input-fd",
                    str(worker_binary_input),
                    "--binary-output-fd",
                    str(worker_binary_output),
                ],
                cwd=Path.cwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(worker_binary_input, worker_binary_output),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            worker_binary_input = -1
            worker_binary_output = -1
            assert process.stdin is not None
            assert process.stdout is not None

            def receive_worker_packet() -> WorkerControlMessage:
                """Read one worker packet and its matching binary boundary."""
                packet = receive_message(process.stdout)
                descriptors = packet.get("payloads", [])
                receive_payloads(
                    core_binary_input_stream,
                    cast(list[WorkerPayloadDescriptor], descriptors),
                )
                return packet

            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "hello",
                    "handshake",
                    cast(
                        dict[str, WorkerValue],
                        {
                            "versions": [1],
                            "capabilities": remote.CORE_CAPABILITIES,
                            "required_capabilities": {"streaming": True},
                        },
                    ),
                    message_id="hello-request",
                ),
            )
            receive_worker_packet()
            receive_worker_packet()
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "request",
                    "load_model",
                    cast(
                        dict[str, WorkerValue],
                        {"arguments": {"model_id": "test-model"}},
                    ),
                    message_id="load-request",
                ),
            )
            loaded = receive_worker_packet()
            model_id = cast(dict, loaded["data"])["value"]
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "request",
                    "generate_stream",
                    cast(
                        dict[str, WorkerValue],
                        {"arguments": {"model_id": model_id}},
                    ),
                    message_id="stream-request",
                ),
            )
            first_frame = receive_worker_packet()
            first_data = cast(dict[str, WorkerValue], first_frame["data"])
            self.assertEqual(
                first_frame["kind"],
                "response",
                f"unexpected worker packet: {first_frame!r}",
            )
            self.assertTrue(
                first_data.get("stream"),
                f"unexpected worker packet: {first_frame!r}",
            )
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "shutdown",
                    "shutdown",
                    {"active_job_policy": policy},
                    message_id="shutdown-request",
                ),
            )
            while True:
                packet = receive_worker_packet()
                if packet["kind"] == "shutdown_ack":
                    acknowledgement = cast(dict[str, WorkerValue], packet["data"])
                    break
                packets.append(packet)
            process.wait(timeout=5)
            return cast(dict[str, WorkerValue], acknowledgement["value"]), packets
        finally:
            for descriptor in (
                worker_binary_input,
                worker_binary_output,
                core_binary_output,
                core_binary_input,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
            with suppress(OSError, ValueError):
                core_binary_input_stream.close()
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    @LINUX_ONLY
    def test_worker_shutdown_finish_waits_for_active_stream(self) -> None:
        """Verify finish waits for completion and reports no cancellation."""
        value, packets = self._run_worker_stream_shutdown_policy("finish")

        self.assertEqual(value["active_job_policy"], "finish")
        self.assertEqual(value["active_job_result"], "finished")
        self.assertFalse(value["active_job_cancelled"])
        self.assertTrue(value["active_job_completed"])
        self.assertTrue(
            any(
                cast(dict, packet["data"]).get("stream")
                for packet in packets
                if packet["kind"] == "response"
            )
        )

    @LINUX_ONLY
    def test_worker_shutdown_cancel_interrupts_active_stream(self) -> None:
        """Verify cancel interrupts an active stream and reports cancellation."""
        value, packets = self._run_worker_stream_shutdown_policy("cancel")

        self.assertEqual(value["active_job_policy"], "cancel")
        self.assertEqual(value["active_job_result"], "cancelled")
        self.assertTrue(value["active_job_cancelled"])
        self.assertTrue(value["active_job_completed"])
        self.assertTrue(
            any(
                cast(dict, packet["data"]).get("cancelled")
                for packet in packets
                if packet["kind"] == "response"
            )
        )

    @LINUX_ONLY
    def test_worker_shutdown_cancel_bounds_ignored_stream_join(self) -> None:
        """Verify shutdown acknowledges an active generator that ignores cancellation."""
        value, packets = self._run_worker_stream_shutdown_policy(
            "cancel",
            ignore_cancellation=True,
        )

        self.assertEqual(value["active_job_policy"], "cancel")
        self.assertEqual(value["active_job_result"], "timed_out")
        self.assertTrue(value["active_job_cancelled"])
        self.assertFalse(value["active_job_completed"])
        self.assertFalse(
            any(
                cast(dict, packet["data"]).get("done")
                for packet in packets
                if packet["kind"] == "response"
            )
        )

    def _run_blocking_worker_control_test(
        self, packet_kind: str, operation: str = "preload_models"
    ) -> float:
        """Verify control traffic remains responsive during a blocking request."""
        child_code = """
import os
import threading
from celune.cedts import worker


release_fd = int(os.environ["CELUNE_TEST_RELEASE_FD"])
started_fd = int(os.environ["CELUNE_TEST_STARTED_FD"])
operation_finished = threading.Event()
response_sent = threading.Event()


def signal_operation_started():
    os.write(started_fd, b"1")


def wait_for_release():
    # Keep the fake backend operation blocked until the parent releases it.
    os.read(release_fd, 1)


class FakeBackend:
    name = "subprocess-blocking-fake"
    chunk_rate = 0.0
    supported_languages = ()
    voice_models = None
    default_voice = None
    model_name = None
    voices = []
    clone_model_id = None
    uses_voice_bundles = False
    max_new_tokens = 512
    is_fake = True

    def preload_models(self):
        signal_operation_started()
        try:
            wait_for_release()
        finally:
            operation_finished.set()

    def load_model(self, model_id):
        signal_operation_started()
        try:
            wait_for_release()
            return model_id
        finally:
            operation_finished.set()

    def convert(self, request):
        del request
        signal_operation_started()
        try:
            wait_for_release()
            return {"converted": True}
        finally:
            operation_finished.set()

    def resolve_generation_language(self, lang):
        signal_operation_started()
        try:
            wait_for_release()
            return lang
        finally:
            operation_finished.set()

    def unload_model(self, release_cuda_cache=True):
        del release_cuda_cache


def fake_load_backend(manifest, log, fatal, kwargs):
    del manifest, log, fatal, kwargs
    return FakeBackend()


worker._load_backend = fake_load_backend

real_send_message = worker._send_message
def controlled_send_message(
    protocol_stream, binary_output, packet, send_lock, *, limits
):
    # Hold shutdown completion until the released operation has cleaned up.
    real_send_message(
        protocol_stream,
        binary_output,
        packet,
        send_lock,
        limits=limits,
    )
    if packet.get("kind") == "response" and packet.get("reply_to") == "blocking-request":
        response_sent.set()
    if packet.get("kind") == "shutdown_ack":
        os.read(release_fd, 1)
        operation_finished.wait(timeout=2.0)
        response_sent.wait(timeout=2.0)


worker._send_message = controlled_send_message
worker._SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0.1
try:
    raise SystemExit(worker.main())
finally:
    os.close(release_fd)
"""
        worker_binary_input, core_binary_output = os.pipe()
        core_binary_input, worker_binary_output = os.pipe()
        release_read, release_write = os.pipe()
        started_read, started_write = os.pipe()
        process: Optional[subprocess.Popen[bytes]] = None

        def release_blocked_operation() -> None:
            """Release the child operation once its control result is observed."""
            nonlocal release_write
            if release_write < 0:
                return
            with suppress(OSError):
                os.write(release_write, b"r")
            with suppress(OSError):
                os.close(release_write)
            release_write = -1

        try:
            worker_environment = os.environ.copy()
            worker_environment["CELUNE_TEST_RELEASE_FD"] = str(release_read)
            worker_environment["CELUNE_TEST_STARTED_FD"] = str(started_write)
            process = subprocess.Popen(  # pylint: disable=R1732
                [
                    sys.executable,
                    "-c",
                    child_code,
                    "--backend",
                    "mini",
                    "--backend-kwargs",
                    "{}",
                    "--binary-input-fd",
                    str(worker_binary_input),
                    "--binary-output-fd",
                    str(worker_binary_output),
                ],
                cwd=Path.cwd(),
                env=worker_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(
                    worker_binary_input,
                    worker_binary_output,
                    release_read,
                    started_write,
                ),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            os.close(release_read)
            os.close(started_write)
            worker_binary_input = -1
            worker_binary_output = -1
            release_read = -1
            started_write = -1
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None

            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "hello",
                    "handshake",
                    cast(
                        dict[str, WorkerValue],
                        {
                            "versions": [1],
                            "capabilities": remote.CORE_CAPABILITIES,
                            "required_capabilities": {"streaming": True},
                        },
                    ),
                    message_id="hello-request",
                ),
            )
            receive_message(process.stdout)
            receive_message(process.stdout)
            request_arguments: dict[str, WorkerValue]
            if operation == "preload_models":
                request_arguments = {}
            elif operation == "load_model":
                request_arguments = {"model_id": "test-model"}
            elif operation == "call":
                request_arguments = {
                    "method": "resolve_generation_language",
                    "lang": "en",
                }
            else:
                raise AssertionError(f"unsupported blocking operation: {operation}")
            request_packet = build_packet(
                "request",
                operation,
                {"arguments": request_arguments},
                message_id="blocking-request",
            )
            request_control, request_payloads = encode_message(request_packet)
            send_message(process.stdin, request_control)
            with os.fdopen(
                os.dup(core_binary_output), "wb", buffering=0
            ) as binary_stream:
                send_payloads(binary_stream, request_payloads)
            blocking_deadline = time.monotonic() + 10.0
            remaining = blocking_deadline - time.monotonic()
            ready, _, _ = select.select([started_read], [], [], remaining)
            if not ready:
                stderr = process.stderr.read() if process.poll() is not None else b""
                self.fail(
                    "blocking backend operation did not start "
                    f"(returncode={process.poll()}, stderr={stderr!r})"
                )
            if not os.read(started_read, 1):
                stderr = process.stderr.read() if process.poll() is not None else b""
                self.fail(
                    "blocking backend operation did not start "
                    f"(returncode={process.poll()}, stderr={stderr!r})"
                )
            os.close(started_read)
            started_read = -1

            started = time.monotonic()
            if packet_kind == "cancel":
                send_no_payload_packet(
                    process.stdin,
                    core_binary_output,
                    build_packet(
                        "cancel",
                        "cancel",
                        {"target_message_id": "blocking-request"},
                        message_id="cancel-request",
                    ),
                )
                cancellation_ack: Optional[WorkerControlMessage] = None
                response: Optional[WorkerControlMessage] = None
                cancellation_ack_elapsed: Optional[float] = None
                while cancellation_ack is None or response is None:
                    packet = receive_message(process.stdout)
                    if packet["kind"] == "cancel_ack":
                        cancellation_ack = packet
                        cancellation_ack_elapsed = time.monotonic() - started
                        release_blocked_operation()
                    elif packet["kind"] == "response":
                        response = packet
                assert cancellation_ack is not None
                assert response is not None
                self.assertEqual(cancellation_ack["kind"], "cancel_ack")
                self.assertFalse(cast(dict, cancellation_ack["data"])["cancelled"])
                self.assertEqual(response["kind"], "response")
                self.assertTrue(cast(dict, response["data"])["ok"])
                assert cancellation_ack_elapsed is not None
                return cancellation_ack_elapsed

            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "shutdown",
                    "shutdown",
                    {"active_job_policy": "cancel"},
                    message_id="shutdown-request",
                ),
            )
            shutdown_ack: Optional[WorkerControlMessage] = None
            deadline = started + 1.0
            while shutdown_ack is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail("shutdown acknowledgement was not prompt")
                ready, _, _ = select.select([process.stdout], [], [], remaining)
                if not ready:
                    self.fail("shutdown acknowledgement was not prompt")
                packet = receive_message(process.stdout)
                if packet["kind"] == "shutdown_ack":
                    shutdown_ack = packet
            elapsed = time.monotonic() - started
            self.assertEqual(shutdown_ack["kind"], "shutdown_ack")
            self.assertEqual(shutdown_ack["reply_to"], "shutdown-request")
            shutdown_data = cast(dict[str, WorkerValue], shutdown_ack["data"])
            self.assertFalse(shutdown_data["ok"])
            shutdown_value = cast(dict[str, WorkerValue], shutdown_data["value"])
            self.assertEqual(shutdown_value["active_job_result"], "timed_out")
            self.assertFalse(shutdown_value["active_job_cancelled"])
            release_blocked_operation()
            self.assertEqual(process.wait(timeout=2), 0)
            return elapsed
        finally:
            for descriptor in (
                worker_binary_input,
                worker_binary_output,
                core_binary_output,
                core_binary_input,
                release_read,
                started_read,
                started_write,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
            release_blocked_operation()
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    @LINUX_ONLY
    def test_worker_control_loop_handles_cancel_during_blocking_operation(self) -> None:
        """Verify non-stream cancellation is rejected before successful completion."""
        for operation in ("preload_models", "load_model", "call"):
            with self.subTest(operation=operation):
                self.assertLess(
                    self._run_blocking_worker_control_test("cancel", operation),
                    1.0,
                )

    def test_worker_blocking_conversion_does_not_report_successful_cancellation(
        self,
    ) -> None:
        """Verify a blocked conversion completes normally after cancellation is requested."""
        started = threading.Event()
        release = threading.Event()
        cancellation = threading.Event()
        responses: list[WorkerResponse] = []

        class FakeBackend:
            """Backend stand-in for a non-cooperative conversion operation."""

            def convert(self, request: VoiceConversionRequest) -> dict[str, bool]:
                """Block conversion until the test releases the backend."""
                del request
                started.set()
                release.wait(timeout=2)
                return {"converted": True}

        def run_conversion() -> None:
            response, _ = worker._run_request(
                cast(_BackendRuntime, FakeBackend()),
                {
                    "operation": "convert",
                    "arguments": {
                        "request": VoiceConversionRequest(
                            np.array([0.0], dtype=np.float32), 48000
                        )
                    },
                },
                {},
                1,
                io.BytesIO(),
                cancellation_event=cancellation,
            )
            responses.append(response)

        thread = threading.Thread(target=run_conversion)
        thread.start()
        self.assertTrue(started.wait(timeout=1))
        cancellation.set()
        release.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(responses), 1)
        self.assertTrue(responses[0]["ok"])
        self.assertNotIn("cancelled", responses[0])

    @LINUX_ONLY
    def test_worker_control_loop_handles_shutdown_during_blocking_operation(
        self,
    ) -> None:
        """Verify shutdown is acknowledged while preload remains blocked."""
        self.assertLess(self._run_blocking_worker_control_test("shutdown"), 1.0)

    @LINUX_ONLY
    def test_worker_rejects_cancel_after_terminal_state_is_marked(self) -> None:
        """Verify a cancel racing with terminal transmission cannot cancel a finished job."""
        child_code = """
import threading
from celune.cedts import worker


class FakeBackend:
    name = "subprocess-stream-fake"
    chunk_rate = 0.0
    supported_languages = ()
    voice_models = None
    default_voice = None
    model_name = None
    voices = []
    clone_model_id = None
    uses_voice_bundles = False
    max_new_tokens = 512
    is_fake = True

    def load_model(self, model_id):
        del model_id
        return object()

    def generate_stream(self, model):
        del model
        yield {"audio": [0.0], "sample_rate": 48000}

    def unload_model(self, release_cuda_cache=True):
        del release_cuda_cache


def fake_load_backend(manifest, log, fatal, kwargs):
    del manifest, log, fatal, kwargs
    return FakeBackend()


real_send_message = worker._send_message
terminal_release = threading.Event()


def controlled_send_message(
    protocol_stream, binary_output, packet, send_lock, *, limits
):
    data = packet.get("data", {})
    if packet.get("kind") == "response" and isinstance(data, dict) and data.get("done"):
        print("TERMINAL_READY", file=worker._WORKER_STDERR, flush=True)
        threading.Timer(1.0, terminal_release.set).start()
        terminal_release.wait(2.0)
    return real_send_message(
        protocol_stream, binary_output, packet, send_lock, limits=limits
    )


worker._load_backend = fake_load_backend
worker._send_message = controlled_send_message
raise SystemExit(worker.main())
"""
        worker_binary_input, core_binary_output = os.pipe()
        core_binary_input, worker_binary_output = os.pipe()
        process: Optional[subprocess.Popen[bytes]] = None
        try:
            process = subprocess.Popen(  # pylint: disable=R1732
                [
                    sys.executable,
                    "-c",
                    child_code,
                    "--backend",
                    "mini",
                    "--backend-kwargs",
                    "{}",
                    "--binary-input-fd",
                    str(worker_binary_input),
                    "--binary-output-fd",
                    str(worker_binary_output),
                ],
                cwd=Path.cwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(worker_binary_input, worker_binary_output),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            worker_binary_input = -1
            worker_binary_output = -1
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None

            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "hello",
                    "handshake",
                    cast(
                        dict[str, WorkerValue],
                        {
                            "versions": [1],
                            "capabilities": remote.CORE_CAPABILITIES,
                            "required_capabilities": {"streaming": True},
                        },
                    ),
                    message_id="hello-request",
                ),
            )
            receive_message(process.stdout)
            receive_message(process.stdout)
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "request",
                    "load_model",
                    cast(
                        dict[str, WorkerValue],
                        {
                            "arguments": cast(
                                dict[str, WorkerValue], {"model_id": "test-model"}
                            )
                        },
                    ),
                    message_id="load-request",
                ),
            )
            loaded = receive_message(process.stdout)
            model_id = cast(dict, loaded["data"])["value"]
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "request",
                    "generate_stream",
                    {"arguments": {"model_id": model_id}},
                    message_id="stream-request",
                ),
            )
            first_frame = receive_message(process.stdout)
            self.assertTrue(cast(dict, first_frame["data"])["stream"])
            terminal_line = b""
            for _ in range(20):
                terminal_line = process.stderr.readline()
                if b"TERMINAL_READY" in terminal_line:
                    break
            self.assertIn(b"TERMINAL_READY", terminal_line)
            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "cancel",
                    "cancel",
                    {"target_message_id": "stream-request"},
                    message_id="cancel-request",
                ),
            )
            cancel_ack = receive_message(process.stdout)
            terminal = receive_message(process.stdout)
            self.assertEqual(cancel_ack["kind"], "cancel_ack")
            self.assertFalse(cast(dict, cancel_ack["data"])["cancelled"])
            self.assertTrue(cast(dict, terminal["data"])["done"])
        finally:
            for descriptor in (
                worker_binary_input,
                worker_binary_output,
                core_binary_output,
                core_binary_input,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def test_remote_proxy_rejects_mismatched_and_duplicate_replies(self) -> None:
        """Verify response IDs must match the active request exactly once."""
        mismatched = io.BytesIO()
        send_message(
            mismatched,
            build_packet(
                "response",
                "describe",
                {"ok": True, "value": "wrong"},
                reply_to="other-request",
            ),
        )
        mismatched.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        with self.assertRaises(CEDTSError):
            proxy._read_response(
                cast(subprocess.Popen[bytes], SimpleNamespace(stdout=mismatched)),
                "active-request",
            )

        duplicate = io.BytesIO()
        packet = build_packet(
            "response",
            "describe",
            {"ok": True, "value": "once"},
            reply_to="active-request",
        )
        send_message(duplicate, packet)
        send_message(duplicate, packet)
        duplicate.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        process = cast(subprocess.Popen[bytes], SimpleNamespace(stdout=duplicate))
        self.assertEqual(
            proxy._read_response(process, "active-request")["value"], "once"
        )
        with self.assertRaises(CEDTSError):
            proxy._read_response(process, "active-request")

    def test_remote_proxy_does_not_treat_progress_as_a_response(self) -> None:
        """Verify progress packets are consumed until the correlated response arrives."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "progress",
                "generate_stream",
                {"step": 1},
                reply_to="active-request",
            ),
        )
        send_message(
            stream,
            build_packet(
                "response",
                "generate_stream",
                {"ok": True, "done": True},
                reply_to="active-request",
            ),
        )
        stream.seek(0)
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        response = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream)),
            "active-request",
        )
        self.assertTrue(response["done"])

    def test_remote_proxy_keeps_worker_builtin_exception_types_inert(self) -> None:
        """Verify worker ValueErrors never select a core exception class."""
        error = remote._worker_exception("builtins.ValueError", "bad input")

        self.assertIsInstance(error, remote.BackendError)
        self.assertNotIsInstance(error, ValueError)
        self.assertEqual(error.error_code, "backend_worker_error")
        self.assertEqual(error.error_type, "builtins.ValueError")
        self.assertEqual(
            str(error),
            "backend_worker_error (builtins.ValueError): bad input",
        )

    def test_remote_proxy_rejects_unmapped_worker_exception_names(self) -> None:
        """Verify unknown and nested worker exception names remain backend errors."""
        for error_type in (
            "builtins.Exception",
            "celune.exceptions.BackendError.Nested",
            "backend_pkg.ModelError",
        ):
            with self.subTest(error_type=error_type):
                error = remote._worker_exception(error_type, "generation failed")

                self.assertIsInstance(error, remote.BackendError)
                self.assertEqual(error.error_code, "backend_worker_error")
                self.assertEqual(error.error_type, error_type)
                self.assertEqual(
                    str(error),
                    f"backend_worker_error ({error_type}): generation failed",
                )

    def test_remote_proxy_does_not_invoke_constructor_sensitive_worker_names(
        self,
    ) -> None:
        """Verify constructor-like wire names cannot select executable attributes."""
        error = remote._worker_exception(
            "builtins.__class_getitem__",
            "generation failed",
        )

        self.assertIsInstance(error, remote.BackendError)
        self.assertEqual(error.error_type, "builtins.__class_getitem__")
        self.assertEqual(
            str(error),
            "backend_worker_error (builtins.__class_getitem__): generation failed",
        )

    def test_remote_proxy_does_not_drain_after_consuming_worker_error_frame(
        self,
    ) -> None:
        """Verify worker errors reach the caller without waiting for a done frame."""
        stream = io.BytesIO()
        send_message(
            stream,
            build_packet(
                "error",
                "generate_stream",
                {
                    "ok": False,
                    "error": "missing codec",
                    "error_type": "builtins.ImportError",
                },
                reply_to="request-id",
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

        with (
            mock.patch.object(proxy, "_send_packet", return_value="request-id"),
            mock.patch.object(proxy, "_drain_stream") as drain,
            self.assertRaisesRegex(BackendError, "missing codec"),
        ):
            list(proxy._stream_request("generate_stream"))

        drain.assert_not_called()

    def test_remote_proxy_classifies_worker_tracebacks_as_errors(self) -> None:
        """Verify raw worker traceback lines are not forwarded as informational logs."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        log = mock.Mock()
        stream = io.BytesIO(
            b"Traceback (most recent call last):\n"
            b'  File "backend.py", line 1, in load\n'
            b"ImportError: missing codec\n"
            b"[INFO] worker still alive\n"
        )

        proxy._read_worker_logs(stream, log)

        assert [call.args[1] for call in log.call_args_list] == [
            "error",
            "error",
            "error",
            "info",
        ]

    def test_remote_proxy_suppresses_known_runtime_log_messages(self) -> None:
        """Verify CEDTS worker logs share Celune's runtime suppressions."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        log = mock.Mock()
        ignored = "\n".join(
            f"{message} additional details"
            for message in remote.RUNTIME_LOG_FILTER_MESSAGES
        )
        stream = io.BytesIO(f"{ignored}\nbackend message\n".encode())

        proxy._read_worker_logs(stream, log)

        log.assert_called_once_with("backend message", "info", loglevel="info")

    def test_remote_proxy_uses_backend_error_for_unknown_worker_exception_types(
        self,
    ) -> None:
        """Verify unknown backend exception classes retain their qualified name."""
        error = remote._worker_exception("backend_pkg.ModelError", "generation failed")

        self.assertIsInstance(error, remote.BackendError)
        self.assertEqual(error.error_code, "backend_worker_error")
        self.assertEqual(error.error_type, "backend_pkg.ModelError")
        self.assertEqual(
            str(error),
            "backend_worker_error (backend_pkg.ModelError): generation failed",
        )

    def test_remote_proxy_does_not_inherit_core_pythonhome(self) -> None:
        """Verify workers receive only safe runtime variables and paths."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BACKEND_MANIFESTS["mini"]
        backend_environment = BackendEnvironment(
            BACKEND_MANIFESTS["mini"], Path("C:/backend")
        )
        process = SimpleNamespace(stderr=None)

        with (
            mock.patch.dict(
                remote.os.environ,
                {
                    "PYTHONHOME": "C:/Python314",
                    "PYTHONPATH": "C:/attacker",
                    "USERNAME": "test-user",
                    "CELUNE_WORKER_SECRET": "do-not-forward",
                    "HF_TOKEN": "do-not-forward",
                    "PATH": "C:/system",
                    "TEMP": "C:/temp",
                    "CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_PATH": "C:/CUDA",
                    "LD_LIBRARY_PATH": "/opt/cuda/lib64",
                },
                clear=True,
            ),
            mock.patch.object(
                remote.subprocess, "Popen", return_value=process
            ) as popen,
            mock.patch(
                "celune.cedts.remote.configure_numba_cache",
                return_value=Path("C:/celune/temp/numba"),
            ) as configure_numba_cache,
            mock.patch(
                "celune.cedts.remote.huggingface_home_dir",
                return_value=Path("C:/celune/huggingface"),
            ),
            mock.patch(
                "celune.cedts.remote.huggingface_hub_cache_dir",
                return_value=Path("C:/celune/huggingface/hub"),
            ),
        ):
            proxy._start_worker(
                backend_environment,
                lambda msg, severity="info", *, loglevel="info": None,
                {},
            )

        self.assertNotIn("PYTHONHOME", popen.call_args.kwargs["env"])
        self.assertNotIn("CELUNE_WORKER_SECRET", popen.call_args.kwargs["env"])
        self.assertNotIn("HF_TOKEN", popen.call_args.kwargs["env"])
        self.assertNotIn("C:/attacker", popen.call_args.kwargs["env"]["PYTHONPATH"])
        self.assertEqual(
            popen.call_args.kwargs["env"]["PYTHONPATH"],
            str(remote.project_root().resolve()),
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["PATH"],
            os.pathsep.join(
                (str(backend_environment.python.resolve().parent), "C:/system")
            ),
        )
        self.assertEqual(popen.call_args.kwargs["env"]["TEMP"], "C:/temp")
        self.assertEqual(
            popen.call_args.kwargs["env"]["NUMBA_CACHE_DIR"],
            str(Path("C:/celune/temp/numba")),
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["HF_HOME"],
            str(Path("C:/celune/huggingface")),
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["HF_HUB_CACHE"],
            str(Path("C:/celune/huggingface/hub")),
        )
        configure_numba_cache.assert_called_once_with()
        self.assertEqual(popen.call_args.kwargs["env"]["USERNAME"], "test-user")
        self.assertEqual(
            popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],
            "1",
        )
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_PATH"], "C:/CUDA")
        self.assertEqual(
            popen.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
            "/opt/cuda/lib64",
        )
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(popen.call_args.args[0][2], "--backend")
        assert proxy._binary_input is not None
        assert proxy._binary_output is not None
        proxy._binary_input.close()
        proxy._binary_output.close()

    @WINDOWS_ONLY
    def test_remote_proxy_windows_launch_allowlists_binary_handles(self) -> None:
        """Verify Windows workers inherit only the two CEDTS binary handles."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BACKEND_MANIFESTS["mini"]
        backend_environment = BackendEnvironment(
            BACKEND_MANIFESTS["mini"], Path("C:/backend")
        )
        process = SimpleNamespace(stderr=None, pid=1234)
        handle_list: list[int] = []
        proxy._process = None

        class ProjectRoot:
            """Mock Path object."""

            def __truediv__(self, value: str) -> "ProjectRoot":
                """Join path elements together."""
                return self

            def resolve(self) -> "ProjectRoot":
                """Resolve the selected path."""
                return self

            def __str__(self) -> str:
                """Return a mock path string."""
                return "/root/VoiceSpeaker/celune/cedts/bootstrap.py"

        class StartupInfo:
            """Mock startup info."""

            def __init__(self) -> None:
                self.lpAttributeList: dict[str, list[int]] = {}

        fake_msvcrt = SimpleNamespace(
            get_osfhandle=lambda descriptor: descriptor + 1000,
        )

        def record_startup_info(command: list[str], **kwargs: object) -> object:
            startup_info = cast(StartupInfo, kwargs["startupinfo"])
            handle_list.extend(startup_info.lpAttributeList["handle_list"])
            self.assertTrue(kwargs["close_fds"])
            self.assertIn("--binary-input-handle", command)
            self.assertIn("--binary-output-handle", command)
            self.assertNotIn("--binary-input-fd", command)
            self.assertNotIn("--binary-output-fd", command)
            return process

        with (
            mock.patch.object(remote, "project_root", return_value=ProjectRoot()),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                remote.os,
                "set_handle_inheritable",
                create=True,
            ),
            mock.patch.object(
                remote.subprocess,
                "STARTUPINFO",
                StartupInfo,
                create=True,
            ),
            mock.patch.object(
                remote.subprocess,
                "Popen",
                side_effect=record_startup_info,
            ),
            mock.patch(
                "celune.cedts.remote.configure_numba_cache",
                return_value=Path("C:/celune/temp/numba"),
            ),
            mock.patch(
                "celune.cedts.remote.huggingface_home_dir",
                return_value=Path("C:/celune/huggingface"),
            ),
            mock.patch(
                "celune.cedts.remote.huggingface_hub_cache_dir",
                return_value=Path("C:/celune/huggingface/hub"),
            ),
        ):
            proxy._start_worker(
                backend_environment,
                lambda msg, severity="info", *, loglevel="info": None,
                {},
            )

        self.assertEqual(len(handle_list), 2)
        assert proxy._binary_input is not None
        assert proxy._binary_output is not None
        proxy._binary_input.close()
        proxy._binary_output.close()

    @WINDOWS_ONLY
    def test_worker_opens_windows_binary_handles_as_streams(self) -> None:
        """Verify Windows workers convert inherited handles into binary streams."""
        args = cast(
            argparse.Namespace,
            SimpleNamespace(binary_input_handle=101, binary_output_handle=202),
        )
        input_stream = io.BytesIO()
        output_stream = io.BytesIO()
        open_osfhandle = mock.Mock(side_effect=(31, 32))
        fake_msvcrt = SimpleNamespace(open_osfhandle=open_osfhandle)

        with (
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                worker.os,
                "fdopen",
                side_effect=(input_stream, output_stream),
            ) as fdopen,
        ):
            streams = worker._open_binary_streams(args)

        self.assertEqual(streams, (input_stream, output_stream))
        self.assertEqual(
            open_osfhandle.call_args_list,
            [
                mock.call(101, worker.os.O_RDONLY | getattr(worker.os, "O_BINARY", 0)),
                mock.call(202, worker.os.O_WRONLY | getattr(worker.os, "O_BINARY", 0)),
            ],
        )
        self.assertEqual(
            fdopen.call_args_list,
            [
                mock.call(31, "rb", buffering=0),
                mock.call(32, "wb", buffering=0),
            ],
        )

    def test_backend_resolution_uses_the_registered_manifest(self) -> None:
        """Verify named resolution always delegates to the CEDTS worker proxy."""
        with mock.patch.object(remote, "RemoteBackendProxy") as proxy:
            resolve_backend("mini")

        proxy.assert_called_once()
        assert proxy.call_args.args[0] == BACKEND_MANIFESTS["mini"]
