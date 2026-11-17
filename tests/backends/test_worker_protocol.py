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
import pytest

from celune.cesn import format_cesn_error
from celune.cedts import remote, worker
from celune.paths import temp_data_dir
from celune.utils import discard, format_error
from celune.exceptions import (
    CESNError,
    CEDTSError,
    BackendError,
    CESNIPAError,
    CESNRangeError,
    CESNValidationError,
    CESNBackendFormatError,
)
from celune.backends.tts import resolve_backend
from celune.typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerRequest,
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
    BackendRuntime,
)
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendManifest,
    BackendEnvironment,
)
from celune.dataclasses.pipeline import VoiceConversionRequest

from ..platform import LINUX_ONLY, WINDOWS_ONLY
from .backend_processes import send_no_payload_packet


def _worker_error_response(error: CESNError) -> tuple[WorkerResponse, str]:
    """Capture a raised CESN exception as a worker response and stderr text."""
    worker_stderr = io.StringIO()
    response: Optional[WorkerResponse] = None
    try:
        raise error
    except CESNError as caught_error:
        with mock.patch.object(worker, "_WORKER_STDERR", worker_stderr):
            response = worker._error_response(caught_error)
    return response, worker_stderr.getvalue()


class TestWorkerProtocol:
    """Exercise the CEDTS worker wire protocol in subprocesses."""

    def test_cedts_preserves_cesn_error_source_spans(self) -> None:
        """Return worker-side CESN diagnostics as source-aware parser errors."""
        stream = io.BytesIO()
        with mock.patch.object(worker, "_WORKER_STDERR", io.StringIO()):
            error = CESNValidationError("unsupported score value")
            error.source_start = 8
            error.source_end = 10
            response = worker._error_response(error)

        send_message(
            stream,
            build_packet(
                "error",
                "call",
                cast(dict[str, WorkerValue], response),
            ),
        )
        stream.seek(0)
        packet = receive_message(stream)
        data = cast(dict[str, WorkerValue], packet["data"])
        restored = remote._worker_exception(
            cast(str, data["error_type"]),
            cast(str, data["error"]),
            source_start=cast(int, data["source_start"]),
            source_end=cast(int, data["source_end"]),
        )

        assert isinstance(restored, CESNValidationError)
        assert restored.source_start == 8
        assert restored.source_end == 10
        assert str(restored) == "unsupported score value"

    def test_worker_cesn_errors_use_notation_diagnostics_without_tracebacks(
        self,
    ) -> None:
        """Keep range and IPA input errors on the concise CESN diagnostic path."""
        cases = (
            (
                "[lang:en][f5:1][x-sampa:A] A",
                CESNRangeError("F5", 77, "Celune", 54, 72),
                "f5",
            ),
            (
                "[lang:en][c4:1][ipa:/ʈa/] word",
                CESNIPAError("ʈ"),
                "ʈ",
            ),
        )
        for notation, error, source_text in cases:
            error.source_start = notation.index(source_text)
            error.source_end = error.source_start + len(source_text)
            response, worker_stderr = _worker_error_response(error)

            assert worker_stderr == ""
            assert worker.WORKER_TRACEBACK_MARKER not in response["error"]

            restored = remote._worker_exception(
                response["error_type"],
                response["error"],
                source_start=response["source_start"],
                source_end=response["source_end"],
            )
            assert isinstance(restored, CESNError)
            diagnostic = format_cesn_error(notation, restored)
            assert diagnostic.startswith("Parse error in CESN notation")
            assert "^" in diagnostic
            assert "Traceback" not in diagnostic

    def test_worker_retains_tracebacks_for_internal_cesn_format_errors(self) -> None:
        """Keep backend-format defects on the backend diagnostic path."""
        error = CESNBackendFormatError("prepared score has an invalid pitch")
        response, worker_stderr = _worker_error_response(error)

        assert "Traceback" in worker_stderr
        assert worker.WORKER_TRACEBACK_MARKER in response["error"]

    def test_remote_proxy_keeps_internal_cesn_format_errors_as_backend_errors(
        self,
    ) -> None:
        """Keep invalid prepared scores on the ordinary backend error path."""
        error = remote._worker_exception(
            f"{CESNBackendFormatError.__module__}.{CESNBackendFormatError.__qualname__}",
            "prepared CESN score has an invalid pitch",
            source_start=4,
            source_end=7,
        )

        assert isinstance(error, BackendError)
        assert error.error_type == "celune.exceptions.CESNBackendFormatError"

    def test_worker_response_serialization_error_reaches_traceback_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Propagate an invalid worker response with its traceback to Celune."""
        stream = io.BytesIO()
        response: WorkerResponse = {
            "ok": True,
            "value": cast(WorkerValue, {"chunk_rate": float("nan")}),
        }
        with mock.patch.object(worker, "_WORKER_STDERR", io.StringIO()):
            worker._send_request_response(
                stream,
                None,
                "response",
                "describe",
                response,
                "describe-request",
                threading.Lock(),
                worker.DEFAULT_CEDTS_LIMITS,
            )

        stream_bytes = stream.getvalue()
        packet = receive_message(io.BytesIO(stream_bytes))
        assert packet["kind"] == "error"
        assert packet["reply_to"] == "describe-request"
        data = cast(dict[str, WorkerValue], packet["data"])
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._received_message_ids = OrderedDict()
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        core_response = proxy._read_response(
            cast(
                subprocess.Popen[bytes],
                SimpleNamespace(stdout=io.BytesIO(stream_bytes)),
            ),
            "describe-request",
        )
        error = remote._worker_exception(
            core_response.get("error_type"),
            cast(str, core_response.get("error")),
        )
        traceback_file = tmp_path / "celune_traceback.txt"
        with mock.patch("celune.utils.traceback_path", return_value=traceback_file):
            format_error(error, "debug")

        traceback_text = traceback_file.read_text(encoding="utf-8")
        assert "Worker traceback:" in traceback_text
        assert "ValueError: JSON number is not finite" in traceback_text
        assert "_send_message" in traceback_text
        assert not cast(bool, data["ok"])

    def test_cedts_accepts_null_cechar_singing_metadata(self) -> None:
        """Keep CECHAR v5 null singing fields valid CEDTS JSON values."""
        stream = io.BytesIO()
        message = build_packet(
            "response",
            "describe",
            {
                "ok": True,
                "value": cast(
                    WorkerValue,
                    {"sing": {"cfg_scale": None, "persona": None}},
                ),
            },
        )
        send_message(stream, message)
        stream.seek(0)

        decoded = receive_message(stream)
        fields = cast(dict[str, WorkerValue], decoded["data"])
        description = cast(dict[str, WorkerValue], fields["value"])
        singing_voice = cast(dict[str, WorkerValue], description["sing"])
        assert (singing_voice["cfg_scale"]) is None
        assert (singing_voice["persona"]) is None

    def test_worker_model_metadata_event_is_validated_and_transmitted(self) -> None:
        """Carry effective runtime metadata from workers to the core."""
        stream = io.BytesIO()
        message = build_packet(
            "event",
            "metadata",
            {"model": "example/model", "type": "gguf", "quant": "Q4_K_M"},
        )
        send_message(stream, message)
        stream.seek(0)

        decoded = receive_message(stream)
        assert decoded["operation"] == "metadata"
        assert decoded["data"] == {
            "model": "example/model",
            "type": "gguf",
            "quant": "Q4_K_M",
        }

        with pytest.raises(CEDTSError):
            send_message(
                io.BytesIO(),
                build_packet(
                    "event",
                    "metadata",
                    {
                        "model": "example/model",
                        "type": "unknown",
                        "quant": "Q4_K_M",
                    },
                ),
            )

    def test_worker_sends_loaded_model_metadata_to_core(self) -> None:
        """Emit runtime metadata as a worker-to-core event after model loading."""

        class MetadataBackend:
            """Minimal worker adapter that publishes one effective runtime."""

            model: Optional[object] = None

            @staticmethod
            def load_model(model_id: str) -> object:
                """Return a placeholder model after checking its identity."""
                assert model_id == "example/model"
                return object()

            @staticmethod
            def inference_metadata(_model: object) -> dict[str, str]:
                """Publish the worker's resolved model runtime metadata."""
                return {
                    "model": "example/model",
                    "type": "gguf",
                    "quant": "Q4_K_M",
                }

        stream = io.BytesIO()
        response, next_model_id = worker._run_request(
            cast(BackendRuntime, MetadataBackend()),
            cast(
                WorkerRequest,
                {
                    "operation": "load_model",
                    "arguments": {"model_id": "example/model"},
                },
            ),
            {},
            7,
            stream,
        )

        assert response["ok"]
        assert next_model_id == 8
        stream.seek(0)
        packet = receive_message(stream)
        assert packet["kind"] == "event"
        assert packet["operation"] == "metadata"
        assert packet["data"] == {
            "model": "example/model",
            "type": "gguf",
            "quant": "Q4_K_M",
        }

    @LINUX_ONLY
    def test_worker_subprocess_completes_cedts_lifecycle_without_backend_dependencies(
        self,
    ) -> None:
        """Verify a real worker process performs handshake, request, and shutdown."""
        child_code = """
from celune.cedts import worker
from celune.utils import discard


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
        discard(release_cuda_cache)


def fake_load_backend(manifest, log, fatal, kwargs):
    discard(manifest)
    discard(log)
    discard(fatal)
    discard(kwargs)
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
                is_ready, _, _ = select.select([process.stdout], [], [], 10.0)
                if not is_ready:
                    pytest.fail("worker did not send the next CEDTS packet")
                return receive_message(process.stdout)

            hello = build_packet(
                "hello",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "versions": [[1, 3]],
                        "capabilities": remote.CORE_CAPABILITIES,
                        "required_capabilities": {"streaming": True},
                    },
                ),
                message_id="hello-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, hello)
            hello_ack = receive_worker_message()
            ready = receive_worker_message()
            assert hello_ack["kind"] == "hello_ack"
            assert hello_ack["reply_to"] == "hello-request"
            assert ready["kind"] == "ready"
            assert ready["reply_to"] == "hello-request"

            request = build_packet(
                "request",
                "describe",
                {"arguments": {}},
                message_id="describe-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, request)
            response = receive_worker_message()
            assert response["kind"] == "response"
            assert response["reply_to"] == "describe-request"
            assert cast(dict, response["data"])["ok"]

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
            assert load_response["kind"] == "response"
            assert load_response["reply_to"] == "load-request"
            assert cast(dict, load_response["data"])["value"] == 1

            shutdown = build_packet(
                "shutdown",
                "shutdown",
                {"active_job_policy": "cancel"},
                message_id="shutdown-request",
            )
            send_no_payload_packet(process.stdin, core_binary_output, shutdown)
            shutdown_ack = receive_worker_message()
            assert shutdown_ack["kind"] == "shutdown_ack"
            assert shutdown_ack["reply_to"] == "shutdown-request"
            assert (
                cast(dict, shutdown_ack["data"])["value"]["active_job_cancelled"]
                is False
            )
            stderr_output = b""
            try:
                _, stderr_output = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                pytest.fail("worker did not exit after shutdown acknowledgement")
            assert process.returncode == 0
            assert "backend load diagnostic" in (
                stderr_output.decode("utf-8", errors="replace")
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
            assert process.stdout.readline() == b"ready\n"
            stderr_thread.join(timeout=5)
            assert not stderr_thread.is_alive()
            assert ([line.rstrip(b"\r\n") for line in stderr_lines]) == (
                [b"backend diagnostic"]
            )
            process.stdin.write(b"\n")
            process.stdin.flush()
            process.wait(timeout=10)
            assert process.returncode == 0
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

    @staticmethod
    def _run_worker_stream_shutdown_policy(
        policy: str, *, ignore_cancellation: bool = False
    ) -> tuple[dict[str, WorkerValue], list[WorkerControlMessage]]:
        """Run an active stream through one CEDTS shutdown policy."""
        child_code = """
import time
from celune.cedts import worker
from celune.utils import discard


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
        discard(model)
        yield {"audio": [0.0], "sample_rate": 48000}
        if IGNORE_CANCELLATION:
            while True:
                time.sleep(0.01)
        time.sleep(3.0)
        yield {"audio": [0.0], "sample_rate": 48000}

    def unload_model(self, release_cuda_cache=True):
        discard(release_cuda_cache)


def fake_load_backend(manifest, log, fatal, kwargs):
    discard(manifest)
    discard(log)
    discard(fatal)
    discard(kwargs)
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
                worker_packet = receive_message(process.stdout)
                descriptors = worker_packet.get("payloads", [])
                receive_payloads(
                    core_binary_input_stream,
                    cast(list[WorkerPayloadDescriptor], descriptors),
                )
                return worker_packet

            send_no_payload_packet(
                process.stdin,
                core_binary_output,
                build_packet(
                    "hello",
                    "handshake",
                    cast(
                        dict[str, WorkerValue],
                        {
                            "versions": [[1, 3]],
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
            assert first_frame["kind"] == "response", (
                f"unexpected worker packet: {first_frame!r}"
            )
            assert first_data.get("stream"), (
                f"unexpected worker packet: {first_frame!r}"
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

        assert value["active_job_policy"] == "finish"
        assert value["active_job_result"] == "finished"
        assert not value["active_job_cancelled"]
        assert value["active_job_completed"]
        assert any(
            cast(dict, packet["data"]).get("stream")
            for packet in packets
            if packet["kind"] == "response"
        )

    @LINUX_ONLY
    def test_worker_shutdown_cancel_interrupts_active_stream(self) -> None:
        """Verify cancel interrupts an active stream and reports cancellation."""
        value, packets = self._run_worker_stream_shutdown_policy("cancel")

        assert value["active_job_policy"] == "cancel"
        assert value["active_job_result"] == "cancelled"
        assert value["active_job_cancelled"]
        assert value["active_job_completed"]
        assert any(
            cast(dict, packet["data"]).get("cancelled")
            for packet in packets
            if packet["kind"] == "response"
        )

    @LINUX_ONLY
    def test_worker_shutdown_cancel_bounds_ignored_stream_join(self) -> None:
        """Verify shutdown acknowledges an active generator that ignores cancellation."""
        value, packets = self._run_worker_stream_shutdown_policy(
            "cancel",
            ignore_cancellation=True,
        )

        assert value["active_job_policy"] == "cancel"
        assert value["active_job_result"] == "timed_out"
        assert value["active_job_cancelled"]
        assert not value["active_job_completed"]
        assert not (
            any(
                cast(dict, packet["data"]).get("done")
                for packet in packets
                if packet["kind"] == "response"
            )
        )

    @staticmethod
    def _run_blocking_worker_control_test(
        packet_kind: str, operation: str = "preload_models"
    ) -> float:
        """Verify control traffic remains responsive during a blocking request."""
        # mock Celune backend for testing purposes
        child_code = """
import os
import threading
from celune.cedts import worker
from celune.utils import discard


release_fd = int(os.environ["CELUNE_TEST_RELEASE_FD"])
started_fd = int(os.environ["CELUNE_TEST_STARTED_FD"])
operation_finished = threading.Event()
response_sent = threading.Event()


def signal_operation_started():
    os.write(started_fd, b"1")


def wait_for_release():
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
        discard(request)
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
        discard(release_cuda_cache)


def fake_load_backend(manifest, log, fatal, kwargs):
    discard(manifest)
    discard(log)
    discard(fatal)
    discard(kwargs)
    return FakeBackend()


worker._load_backend = fake_load_backend

real_send_message = worker._send_message
def controlled_send_message(
    protocol_stream, binary_output, packet, send_lock, *, limits
):
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
                            "versions": [[1, 3]],
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
                pytest.fail(
                    "blocking backend operation did not start "
                    f"(returncode={process.poll()}, stderr={stderr!r})"
                )
            if not os.read(started_read, 1):
                stderr = process.stderr.read() if process.poll() is not None else b""
                pytest.fail(
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
                assert cancellation_ack["kind"] == "cancel_ack"
                assert not cast(dict, cancellation_ack["data"])["cancelled"]
                assert response["kind"] == "response"
                assert cast(dict, response["data"])["ok"]
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
                    pytest.fail("shutdown acknowledgement was not prompt")
                ready, _, _ = select.select([process.stdout], [], [], remaining)
                if not ready:
                    pytest.fail("shutdown acknowledgement was not prompt")
                packet = receive_message(process.stdout)
                if packet["kind"] == "shutdown_ack":
                    shutdown_ack = packet
            elapsed = time.monotonic() - started
            assert shutdown_ack["kind"] == "shutdown_ack"
            assert shutdown_ack["reply_to"] == "shutdown-request"
            shutdown_data = cast(dict[str, WorkerValue], shutdown_ack["data"])
            assert not shutdown_data["ok"]
            shutdown_value = cast(dict[str, WorkerValue], shutdown_data["value"])
            assert shutdown_value["active_job_result"] == "timed_out"
            assert not shutdown_value["active_job_cancelled"]
            release_blocked_operation()
            assert process.wait(timeout=2) == 0
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
            assert self._run_blocking_worker_control_test("cancel", operation) < 1.0

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

            @staticmethod
            def convert(request: VoiceConversionRequest) -> dict[str, bool]:
                """Block conversion until the test releases the backend."""
                discard(request)
                started.set()
                release.wait(timeout=2)
                return {"converted": True}

        def run_conversion() -> None:
            response, _ = worker._run_request(
                cast(BackendRuntime, FakeBackend()),
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
        assert started.wait(timeout=1)
        cancellation.set()
        release.set()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(responses) == 1
        assert responses[0]["ok"]
        assert "cancelled" not in responses[0]

    @LINUX_ONLY
    def test_worker_control_loop_handles_shutdown_during_blocking_operation(
        self,
    ) -> None:
        """Verify shutdown is acknowledged while preload remains blocked."""
        assert self._run_blocking_worker_control_test("shutdown") < 1.0

    @LINUX_ONLY
    def test_worker_rejects_cancel_after_terminal_state_is_marked(self) -> None:
        """Verify a cancel racing with terminal transmission cannot cancel a finished job."""
        child_code = """
import threading
from celune.cedts import worker
from celune.utils import discard


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
        discard(model_id)
        return object()

    def generate_stream(self, model):
        discard(model)
        yield {"audio": [0.0], "sample_rate": 48000}

    def unload_model(self, release_cuda_cache=True):
        discard(release_cuda_cache)


def fake_load_backend(manifest, log, fatal, kwargs):
    discard(manifest)
    discard(log)
    discard(fatal)
    discard(kwargs)
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
                            "versions": [[1, 3]],
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
            assert cast(dict, first_frame["data"])["stream"]
            terminal_line = b""
            for _ in range(20):
                terminal_line = process.stderr.readline()
                if b"TERMINAL_READY" in terminal_line:
                    break
            assert b"TERMINAL_READY" in terminal_line
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
            assert cancel_ack["kind"] == "cancel_ack"
            assert not cast(dict, cancel_ack["data"])["cancelled"]
            assert cast(dict, terminal["data"])["done"]
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
        with pytest.raises(CEDTSError):
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
        assert proxy._read_response(process, "active-request")["value"] == "once"
        with pytest.raises(CEDTSError):
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
        assert response["done"]

    def test_remote_proxy_keeps_worker_builtin_exception_types_inert(self) -> None:
        """Verify worker ValueErrors never select a core exception class."""
        error = remote._worker_exception("builtins.ValueError", "bad input")

        assert isinstance(error, remote.BackendError)
        assert not isinstance(error, ValueError)
        assert error.error_code == "backend_worker_error"
        assert error.error_type == "builtins.ValueError"
        assert str(error) == "backend_worker_error (builtins.ValueError): bad input"

    def test_remote_proxy_rejects_unmapped_worker_exception_names(self) -> None:
        """Verify unknown and nested worker exception names remain backend errors."""
        for error_type in (
            "builtins.Exception",
            "celune.exceptions.BackendError.Nested",
            "backend_pkg.ModelError",
        ):
            error = remote._worker_exception(error_type, "generation failed")

            assert isinstance(error, remote.BackendError)
            assert error.error_code == "backend_worker_error"
            assert error.error_type == error_type
            assert str(error) == (
                f"backend_worker_error ({error_type}): generation failed"
            )

    def test_remote_proxy_does_not_invoke_constructor_sensitive_worker_names(
        self,
    ) -> None:
        """Verify constructor-like wire names cannot select executable attributes."""
        error = remote._worker_exception(
            "builtins.__class_getitem__",
            "generation failed",
        )

        assert isinstance(error, remote.BackendError)
        assert error.error_type == "builtins.__class_getitem__"
        assert str(error) == (
            "backend_worker_error (builtins.__class_getitem__): generation failed"
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
            pytest.raises(BackendError, match="missing codec"),
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

    def test_remote_proxy_suppresses_multiline_tokenizer_warning(self) -> None:
        """Verify a filtered third-party warning and its continuation stay hidden."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        log = mock.Mock()
        stream = io.BytesIO(
            b"The tokenizer is loading with an incorrect regex pattern:\n"
            b"https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84\n"
            b"This will lead to incorrect tokenization. You should set the fix flag.\n"
            b"tokenizer to fix this issue.\n"
            b"[info] backend ready\n"
        )

        proxy._read_worker_logs(stream, log)

        log.assert_called_once_with("backend ready", "info", loglevel="info")

    def test_remote_proxy_suppresses_torchao_windows_worker_warnings(self) -> None:
        """Verify harmless TorchAO and Windows worker diagnostics stay hidden."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._worker_stderr = deque()
        proxy._worker_stderr_lock = threading.Lock()
        log = mock.Mock()
        stream = io.BytesIO(
            b"qwen3/torchao/quant_api.py:1745: SyntaxWarning: "
            b"invalid escape sequence '\\.'\n"
            b"  * regex for parameter names, must start with `re:`\n"
            b"W0920 redirects.py:29] NOTE: Redirects are currently not "
            b"supported in Windows or MacOs.\n"
            b"[info] backend ready\n"
        )

        proxy._read_worker_logs(stream, log)

        log.assert_called_once_with("backend ready", "info", loglevel="info")

    def test_remote_proxy_uses_backend_error_for_unknown_worker_exception_types(
        self,
    ) -> None:
        """Verify unknown backend exception classes retain their qualified name."""
        error = remote._worker_exception("backend_pkg.ModelError", "generation failed")

        assert isinstance(error, remote.BackendError)
        assert error.error_code == "backend_worker_error"
        assert error.error_type == "backend_pkg.ModelError"
        assert str(error) == (
            "backend_worker_error (backend_pkg.ModelError): generation failed"
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

        assert "PYTHONHOME" not in popen.call_args.kwargs["env"]
        assert "CELUNE_WORKER_SECRET" not in popen.call_args.kwargs["env"]
        assert "HF_TOKEN" not in popen.call_args.kwargs["env"]
        assert "C:/attacker" not in popen.call_args.kwargs["env"]["PYTHONPATH"]
        assert popen.call_args.kwargs["env"]["PYTHONPATH"] == (
            str(remote.project_root().resolve())
        )
        assert popen.call_args.kwargs["env"]["PATH"] == (
            os.pathsep.join(
                (str(backend_environment.python.resolve().parent), "C:/system")
            )
        )
        expected_temp_directory = str(temp_data_dir())
        for variable in ("TMPDIR", "TEMP", "TMP"):
            assert popen.call_args.kwargs["env"][variable] == (expected_temp_directory)
        assert (popen.call_args.kwargs["env"]["NUMBA_CACHE_DIR"]) == (
            str(Path("C:/celune/temp/numba"))
        )
        assert (popen.call_args.kwargs["env"]["HF_HOME"]) == (
            str(Path("C:/celune/huggingface"))
        )
        assert (popen.call_args.kwargs["env"]["HF_HUB_CACHE"]) == (
            str(Path("C:/celune/huggingface/hub"))
        )
        configure_numba_cache.assert_called_once_with()
        assert popen.call_args.kwargs["env"]["USERNAME"] == "test-user"
        assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert popen.call_args.kwargs["env"]["CUDA_PATH"] == "C:/CUDA"
        assert popen.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"
        assert popen.call_args.kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert popen.call_args.args[0][2] == "--backend"
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
            assert kwargs["close_fds"]
            assert "--binary-input-handle" in command
            assert "--binary-output-handle" in command
            assert "--binary-input-fd" not in command
            assert "--binary-output-fd" not in command
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

        assert len(handle_list) == 2
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

        assert streams == (input_stream, output_stream)
        assert open_osfhandle.call_args_list == (
            [
                mock.call(101, worker.os.O_RDONLY | getattr(worker.os, "O_BINARY", 0)),
                mock.call(202, worker.os.O_WRONLY | getattr(worker.os, "O_BINARY", 0)),
            ]
        )
        assert fdopen.call_args_list == (
            [
                mock.call(31, "rb", buffering=0),
                mock.call(32, "wb", buffering=0),
            ]
        )

    def test_backend_resolution_uses_the_registered_manifest(self) -> None:
        """Verify named resolution always delegates to the CEDTS worker proxy."""
        with mock.patch.object(remote, "RemoteBackendProxy") as proxy:
            resolve_backend("mini")

        proxy.assert_called_once()
        assert proxy.call_args.args[0] == BACKEND_MANIFESTS["mini"]
