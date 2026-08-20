# SPDX-License-Identifier: Apache-2.0
"""Tests for isolated backend environment metadata and installation."""

import io
import os
import re
import sys
import json
import time
import select
import argparse
import tempfile
import unittest
import threading
import subprocess
from pathlib import Path
from unittest import mock
from contextlib import suppress
from types import SimpleNamespace
from typing import IO, Optional, cast
from collections import OrderedDict, deque
from collections.abc import Callable, Generator

import numpy as np
from celune.exceptions import BackendError
from celune.backends.tts import resolve_backend
from celune.backends import remote, worker, environment
from celune.dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from celune.typing.backends import BackendModel, BackendArgumentValue, _BackendRuntime
from celune.typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerRequest,
    WorkerResponse,
    WorkerControlMessage,
    WorkerPayloadDescriptor,
)
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendManifest,
    BackendEnvironment,
    BackendEnvironmentError,
    BackendEnvironmentManager,
    _exclusive_lock,
    backend_manifest,
)
from celune.backends.worker_protocol import (
    CEDTSLimits,
    WorkerPayload,
    WorkerProtocolError,
    build_packet,
    send_message,
    send_payloads,
    decode_message,
    encode_message,
    receive_message,
    receive_payloads,
    limits_from_capabilities,
    validate_payload_descriptors,
)

from .support import CeluneTestCase


def send_no_payload_packet(
    control_stream: IO[bytes], binary_fd: int, packet: WorkerMessage
) -> None:
    """Send a control packet and its empty binary-channel boundary in tests."""
    control, payloads = encode_message(packet)
    if payloads:
        raise AssertionError("test helper only supports packets without payloads")
    send_message(control_stream, control)
    os.write(binary_fd, b"\x00\x00\x00\x00")


class _ShutdownProcess:
    """Small process stand-in for proxy shutdown lifecycle tests."""

    def __init__(self, stdout: IO[bytes], waits_before_exit: int = 0) -> None:
        self.pid = 1234
        self.stdin = io.BytesIO()
        self.stdout = stdout
        self.stderr = io.BytesIO()
        self.returncode: Optional[int] = None
        self.waits_before_exit = waits_before_exit
        self.terminated = False
        self.killed = False

    def poll(self) -> Optional[int]:
        """Return the stand-in process exit state."""
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        """Exit after the configured number of simulated timeout waits."""
        if self.waits_before_exit:
            self.waits_before_exit -= 1
            raise subprocess.TimeoutExpired("worker", timeout or 1.0)
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        """Record graceful escalation to process termination."""
        self.terminated = True

    def kill(self) -> None:
        """Record final process termination escalation."""
        self.killed = True


class TestBackendEnvironment(CeluneTestCase):
    """Verify backend environment paths and installation transactions."""

    def _make_shutdown_proxy(
        self,
        *,
        active_request_id: Optional[str] = None,
        waits_before_exit: int = 0,
        shutdown_ok: bool = True,
    ) -> tuple[remote.RemoteBackendProxy, _ShutdownProcess]:
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
        process = _ShutdownProcess(stream, waits_before_exit)
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

    def test_worker_diagnostic_localization_keys_exist(self) -> None:
        """Ensure every localized IPC diagnostic key has an English default."""
        source = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in (
                "celune/backends/worker_protocol.py",
                "celune/backends/worker.py",
                "celune/backends/remote.py",
            )
        )
        referenced_keys = set(
            re.findall(
                r"backends\.worker_(?:protocol|runtime|proxy)\.[a-z0-9_]+",
                source,
            )
        )
        translations = json.loads(
            Path("celune/lang/en.json").read_text(encoding="utf-8")
        )

        self.assertTrue(referenced_keys)
        self.assertTrue(referenced_keys.issubset(translations))

    def test_manifests_cover_installed_backend_extras(self) -> None:
        """Verify every supported optional backend has a manifest."""
        assert set(BACKEND_MANIFESTS) == {
            "mini",
            "qwen3",
            "dotstts",
            "voxcpm2",
            "gpt-sovits",
            "seed-vc",
        }

    def test_manifest_lookup_normalizes_backend_id(self) -> None:
        """Verify manifest lookup accepts surrounding whitespace and case changes."""
        assert backend_manifest(" QWEN3 ") is BACKEND_MANIFESTS["qwen3"]

    def test_worker_registry_contains_only_approved_backends(self) -> None:
        """Verify worker construction is limited to the six CEDTS backends."""
        self.assertEqual(
            set(worker._BACKEND_REGISTRY),
            {"mini", "qwen3", "dotstts", "voxcpm2", "gpt-sovits", "seed-vc"},
        )
        self.assertEqual(
            {
                backend_id: kind
                for backend_id, (kind, _loader) in worker._BACKEND_REGISTRY.items()
            },
            {
                "mini": "tts",
                "qwen3": "tts",
                "dotstts": "tts",
                "voxcpm2": "tts",
                "gpt-sovits": "tts",
                "seed-vc": "vc",
            },
        )

    def test_worker_registry_rejects_unregistered_backend_ids(self) -> None:
        """Verify unregistered manifest IDs cannot select a constructor."""
        manifest = BackendManifest(
            "unregistered",
            "tts",
            (),
            "attacker.module",
            "AttackerBackend",
        )
        with self.assertRaises(worker.WorkerProtocolError):
            worker._load_backend(manifest, mock.Mock(), mock.Mock(), {})

    def test_worker_registry_ignores_manifest_constructor_strings(self) -> None:
        """Verify approved IDs use their static constructor instead of manifest strings."""
        constructor = mock.Mock(return_value=mock.sentinel.backend)
        manifest = BackendManifest(
            "mini",
            "tts",
            (),
            "attacker.module",
            "AttackerBackend",
        )
        with mock.patch.dict(
            worker._BACKEND_REGISTRY,
            {"mini": ("tts", lambda: constructor)},
            clear=False,
        ):
            log = mock.Mock()
            fatal = mock.Mock()
            result = worker._load_backend(manifest, log, fatal, {"setting": True})
        self.assertIs(result, mock.sentinel.backend)
        constructor.assert_called_once_with(log=log, fatal=fatal, setting=True)

    def test_manifests_use_the_cuda_pytorch_index(self) -> None:
        """Verify isolated backends use the main branch's CUDA 12.8 stack."""
        expected_requirements = {
            "torch==2.11.0+cu128",
            "torchaudio==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
        }
        for manifest in BACKEND_MANIFESTS.values():
            assert "https://download.pytorch.org/whl/cu128" in manifest.index_urls
            assert expected_requirements.issubset(manifest.requirements)

    def test_manifests_use_the_main_branch_huggingface_versions(self) -> None:
        """Verify isolated backends use the main branch's Hugging Face ranges."""
        expected_requirements = {
            "huggingface-hub>=0.36,<1.0.0",
            "transformers>=4.56,<5.0.0",
        }
        for manifest in BACKEND_MANIFESTS.values():
            assert expected_requirements.issubset(manifest.requirements)

    def test_manifests_pin_the_main_branch_librosa_stack(self) -> None:
        """Verify isolated backends pin the compatible librosa dependency chain."""
        expected_requirements = {
            "librosa==0.11.0",
            "llvmlite==0.47.0",
            "numba==0.65.1",
        }
        for manifest in BACKEND_MANIFESTS.values():
            assert expected_requirements.issubset(manifest.requirements)

    def test_dotstts_uses_the_celune_fork(self) -> None:
        """Verify dots.tts is installed from Celune's maintained fork."""
        assert (
            "dots.tts @ git+https://github.com/celunah/dots.tts"
            in BACKEND_MANIFESTS["dotstts"].requirements
        )

    def test_backend_dependency_list_matches_main_backend_normalizers(self) -> None:
        """Verify the backend dependency list follows the main branch declarations."""
        assert "WeTextProcessing" not in BACKEND_MANIFESTS["dotstts"].requirements
        assert "jieba" in BACKEND_MANIFESTS["gpt-sovits"].requirements
        assert "split-lang" in BACKEND_MANIFESTS["gpt-sovits"].requirements
        assert "matplotlib" in BACKEND_MANIFESTS["gpt-sovits"].requirements
        assert "torchcodec" in BACKEND_MANIFESTS["gpt-sovits"].requirements

    def test_fingerprint_changes_when_requirements_change(self) -> None:
        """Verify dependency changes select a different environment directory."""
        first = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
        second = BackendManifest("test", "tts", ("demo==2",), "module", "Backend")
        assert first.fingerprint() != second.fingerprint()

    def test_ensure_installs_backend_requirements_with_dependencies(self) -> None:
        """Verify backend packages are installed with their declared dependencies."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest(
                "test",
                "tts",
                ("wrapper==1",),
                "module",
                "Backend",
            )

            def fake_run(command: list[str], **_kwargs) -> None:
                if command[1] == "venv":
                    virtualenv = Path(command[-1])
                    virtualenv_python = (
                        virtualenv / "Scripts" / "python.exe"
                        if os.name == "nt"
                        else virtualenv / "bin" / "python"
                    )
                    virtualenv_python.parent.mkdir(parents=True, exist_ok=True)
                    virtualenv_python.touch()

            with mock.patch(
                "celune.backends.environment.subprocess.run", side_effect=fake_run
            ) as run:
                manager.ensure(manifest)

            assert run.call_count == 2
            assert "--no-config" in run.call_args_list[1].args[0]
            assert "--no-cache" in run.call_args_list[1].args[0]
            assert "--no-deps" not in run.call_args_list[1].args[0]

    def test_ensure_installs_into_a_temporary_environment_then_publishes_it(
        self,
    ) -> None:
        """Verify uv commands and metadata are written only after installation succeeds."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest(
                "test",
                "tts",
                ("demo==1",),
                "module",
                "Backend",
                index_urls=("https://pypi.org/simple", "https://example.com/simple"),
            )
            backend_environment = manager.environment_for(manifest)

            def fake_run(command: list[str], **_kwargs) -> None:
                if command[1] == "venv":
                    virtualenv = Path(command[-1])
                    virtualenv_python = (
                        virtualenv / "Scripts" / "python.exe"
                        if os.name == "nt"
                        else virtualenv / "bin" / "python"
                    )
                    virtualenv_python.parent.mkdir(parents=True, exist_ok=True)
                    virtualenv_python.touch()

            with mock.patch(
                "celune.backends.environment.subprocess.run", side_effect=fake_run
            ) as run:
                result = manager.ensure(manifest)

            self.assertEqual(result, backend_environment)
            self.assertTrue(result.is_ready)
            self.assertTrue(
                json.loads(result.metadata_path.read_text(encoding="utf-8"))[
                    "fingerprint"
                ]
                == manifest.fingerprint()
            )
            assert run.call_count == 2
            install_command = run.call_args_list[1].args[0]
            strategy_index = install_command.index("--index-strategy")
            self.assertEqual(
                install_command[strategy_index + 1],
                "unsafe-best-match",
            )
            self.assertIn("--index-url", install_command)
            self.assertFalse(
                any(
                    path.name.startswith(f"{backend_environment.root.name}.install-")
                    for path in root.rglob("*")
                )
            )

    def test_ensure_reuses_a_ready_environment_without_running_uv(self) -> None:
        """Verify an already-installed environment is reused."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = BackendEnvironmentManager(
                root=Path(temporary_directory), uv_executable="uv"
            )
            manifest = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
            backend_environment = manager.environment_for(manifest)
            backend_environment.python.parent.mkdir(parents=True)
            backend_environment.python.touch()
            backend_environment.metadata_path.write_text("{}", encoding="utf-8")

            with mock.patch("celune.backends.environment.subprocess.run") as run:
                self.assertEqual(manager.ensure(manifest), backend_environment)

            run.assert_not_called()

    def test_ensure_uses_the_backend_python_by_default(self) -> None:
        """Verify backend environments do not inherit the core interpreter."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")

            def fake_run(command: list[str], **_kwargs) -> None:
                if command[1] == "venv":
                    virtualenv = Path(command[-1])
                    virtualenv_python = (
                        virtualenv / "Scripts" / "python.exe"
                        if os.name == "nt"
                        else virtualenv / "bin" / "python"
                    )
                    virtualenv_python.parent.mkdir(parents=True, exist_ok=True)
                    virtualenv_python.touch()

            with (
                mock.patch(
                    "celune.backends.environment.subprocess.run",
                    side_effect=fake_run,
                ) as run,
            ):
                manager.ensure(manifest)

            assert run.call_args_list[0].args[0][3] == "3.13"

    def test_exclusive_lock_times_out_while_another_handle_owns_it(self) -> None:
        """Verify the operating-system lock blocks a second installer."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "install.lock"
            with (
                _exclusive_lock(lock_path, timeout=1.0),
                self.assertRaises(BackendEnvironmentError),
                _exclusive_lock(lock_path, timeout=0.01),
            ):
                pass

    def test_uv_timeout_becomes_backend_environment_error(self) -> None:
        """Verify stalled uv operations release installation with a clear error."""
        manager = BackendEnvironmentManager(uv_executable="uv", uv_timeout=1.5)
        with (
            mock.patch(
                "celune.backends.environment.subprocess.run",
                side_effect=subprocess.TimeoutExpired("uv", 1.5),
            ) as run,
            self.assertRaisesRegex(
                BackendEnvironmentError,
                "uv operation timed out",
            ),
        ):
            manager._run_uv("venv")

        assert run.call_args.kwargs["timeout"] == 1.5
        assert "PYTHONHOME" not in run.call_args.kwargs["env"]

    def test_uv_does_not_inherit_core_package_manager_settings(self) -> None:
        """Verify uv cannot inherit core environment resolution constraints."""
        manager = BackendEnvironmentManager(uv_executable="uv")
        with (
            mock.patch.dict(
                environment.os.environ,
                {
                    "PIP_CONSTRAINT": "C:/core/constraints.txt",
                    "UV_INDEX_URL": "https://core.example/simple",
                    "PYTHONHOME": "C:/Python314",
                    "PYTHONPATH": "C:/core/site-packages",
                    "PYTHONUSERBASE": "C:/core/userbase",
                    "PYTHONNOUSERSITE": "1",
                    "VIRTUAL_ENV": "C:/core/.venv",
                },
                clear=True,
            ),
            mock.patch("celune.backends.environment.subprocess.run") as run,
        ):
            manager._run_uv("venv")

        child_environment = run.call_args.kwargs["env"]
        assert not any(name.startswith(("PIP_", "UV_")) for name in child_environment)
        for variable in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
        ):
            assert variable not in child_environment

    def test_worker_protocol_round_trips_messages(self) -> None:
        """Verify JSON-compatible worker messages survive CEDTS framing."""
        stream = io.BytesIO()
        message = build_packet(
            "request",
            "describe",
            cast(dict[str, WorkerValue], {"arguments": {}}),
        )
        send_message(stream, message)
        stream.seek(0)
        assert receive_message(stream) == message

    def test_worker_protocol_uses_utf8_json_control_framing(self) -> None:
        """Verify control frames contain only length-prefixed UTF-8 JSON."""
        stream = io.BytesIO()
        message = build_packet(
            "event",
            "ready",
            cast(dict[str, WorkerValue], {"state": "ready", "message": "こんにちは"}),
            message_id="wire-message",
        )

        send_message(stream, message)

        frame = stream.getvalue()
        payload_size = int.from_bytes(frame[:4], "big")
        payload = frame[4:]
        self.assertEqual(payload_size, len(payload))
        self.assertEqual(json.loads(payload.decode("utf-8")), message)

        stream.seek(0)
        self.assertEqual(receive_message(stream), message)

    def test_worker_protocol_rejects_non_json_control_bytes(self) -> None:
        """Verify non-JSON control bytes become a CEDTS protocol error."""
        payload = b"not a CEDTS control object"
        stream = io.BytesIO(len(payload).to_bytes(4, "big") + payload)

        with self.assertRaises(WorkerProtocolError):
            receive_message(stream)

    def test_worker_protocol_normalizes_malformed_control_frames(self) -> None:
        """Verify malformed CEDTS frames consistently raise protocol errors."""

        def frame(payload: bytes) -> bytes:
            """Build one length-prefixed control frame for the test."""
            return len(payload).to_bytes(4, "big") + payload

        malformed_frames = (
            b"",
            b"\x00\x00\x00",
            frame(b""),
            frame(b"\xff"),
            frame(b"{"),
            frame(b"[]"),
            (1024 * 1024 + 1).to_bytes(4, "big"),
        )
        for raw_frame in malformed_frames:
            with (
                self.subTest(raw_frame=raw_frame),
                self.assertRaises(WorkerProtocolError),
            ):
                receive_message(io.BytesIO(raw_frame))

    def test_worker_protocol_rejects_bounded_json_limit_violations(self) -> None:
        """Verify nesting, collection, string, and packet-schema limits fail safely."""

        def frame(payload: bytes) -> bytes:
            """Build one length-prefixed control frame for the test."""
            return len(payload).to_bytes(4, "big") + payload

        nested_packet = (
            b'{"cedts_version":1,"kind":"request","message_id":"nested",'
            b'"reply_to":null,"operation":"describe","data":{"arguments":'
            + b'{"value":'
            + b"[" * 65
            + b"null"
            + b"]" * 65
            + b"}}}"
        )
        oversized_collection = {
            "cedts_version": 1,
            "kind": "request",
            "message_id": "collection",
            "reply_to": None,
            "operation": "describe",
            "data": {"arguments": {"value": list(range(1025))}},
        }
        unknown_packet_field = {
            "cedts_version": 1,
            "kind": "request",
            "message_id": "unknown-field",
            "reply_to": None,
            "operation": "describe",
            "data": {"arguments": {}},
            "unexpected": True,
        }
        for value in (
            nested_packet,
            json.dumps(oversized_collection).encode(),
            json.dumps(unknown_packet_field).encode(),
        ):
            with self.subTest(value=value), self.assertRaises(WorkerProtocolError):
                receive_message(io.BytesIO(frame(value)))

        oversized_string = json.dumps(
            {
                "cedts_version": 1,
                "kind": "request",
                "message_id": "string",
                "reply_to": None,
                "operation": "describe",
                "data": {"arguments": {"value": "x" * (1024 * 1024 + 1)}},
            }
        ).encode()
        with self.assertRaises(WorkerProtocolError):
            receive_message(io.BytesIO(frame(oversized_string)))

    def test_worker_request_validation_rejects_unknown_operations_and_methods(
        self,
    ) -> None:
        """Verify worker dispatch accepts only known operations and callback methods."""

        class FakeBackend:
            """Backend stand-in for dispatch validation."""

            def describe_secret(self) -> str:
                """Represent a method that must never be remotely exposed."""
                return "secret"

        backend = cast(_BackendRuntime, FakeBackend())
        with self.assertRaises(WorkerProtocolError):
            worker._run_request(
                backend,
                {"operation": "unknown", "arguments": {}},
                {},
                1,
                io.BytesIO(),
            )
        with self.assertRaises(WorkerProtocolError):
            worker._run_request(
                backend,
                {
                    "operation": "call",
                    "arguments": {"method": "describe_secret"},
                },
                {},
                1,
                io.BytesIO(),
            )

    def test_worker_request_validation_normalizes_malformed_arguments(self) -> None:
        """Verify non-object and invalid operation arguments become protocol errors."""
        backend = cast(_BackendRuntime, object())
        malformed_requests = (
            {"operation": "describe", "arguments": []},
            {"operation": "model_is_available_locally", "arguments": {}},
            {
                "operation": "unload_model",
                "arguments": {"release_cuda_cache": "yes"},
            },
        )
        for request in malformed_requests:
            with self.subTest(request=request), self.assertRaises(WorkerProtocolError):
                worker._run_request(
                    backend,
                    cast(WorkerRequest, request),
                    {},
                    1,
                    io.BytesIO(),
                )

    def test_worker_protocol_rejects_kind_specific_schema_mismatches(self) -> None:
        """Verify each CEDTS packet kind accepts only its declared data shape."""
        invalid_packets = (
            build_packet(
                "hello",
                "handshake",
                cast(
                    dict[str, WorkerValue],
                    {
                        "versions": [1],
                        "capabilities": remote.CORE_CAPABILITIES,
                        "unexpected": True,
                    },
                ),
            ),
            build_packet("ready", "ready", {}),
            build_packet(
                "response",
                "describe",
                {"ok": True, "unexpected": True},
            ),
            build_packet("event", "fatal", {"fatal": "yes"}),
            build_packet("cancel", "cancel", {"target_message_id": 1}),
            build_packet("shutdown", "shutdown", {}),
            build_packet(
                "error",
                "protocol",
                {"ok": True, "error": "bad", "error_type": "ValueError"},
            ),
        )
        for packet in invalid_packets:
            with self.subTest(packet=packet), self.assertRaises(WorkerProtocolError):
                send_message(io.BytesIO(), packet)

    def test_worker_operation_schemas_require_handles_and_bound_backend_kwargs(
        self,
    ) -> None:
        """Verify model operations require handles while preserving bounded JSON kwargs."""
        valid_wire_packet = build_packet(
            "request",
            "load_model",
            cast(
                dict[str, WorkerValue],
                {
                    "arguments": {
                        "model_id": "mini",
                        "settings": {"voice": "celune", "enabled": True},
                    }
                },
            ),
        )
        send_message(io.BytesIO(), valid_wire_packet)
        with self.assertRaises(WorkerProtocolError):
            send_message(
                io.BytesIO(),
                build_packet(
                    "request",
                    "load_model",
                    cast(
                        dict[str, WorkerValue],
                        {"arguments": {"settings": {"voice": "celune"}}},
                    ),
                ),
            )

        class FakeModel:
            """Opaque model stand-in for operation schema validation."""

        class FakeBackend:
            """Backend stand-in that records validated keyword arguments."""

            model: Optional[BackendModel] = None

            loaded: tuple[str, dict[str, BackendArgumentValue]]
            generated: tuple[BackendModel, dict[str, BackendArgumentValue]]

            def load_model(
                self, model_id: str, **kwargs: BackendArgumentValue
            ) -> FakeModel:
                """Record model-loading arguments and return an opaque model."""
                self.loaded = (model_id, kwargs)
                return FakeModel()

            def generate_stream(
                self, model: BackendModel, **kwargs: BackendArgumentValue
            ):
                """Record generation arguments without emitting audio frames."""
                self.generated = (model, kwargs)
                yield from ()

        backend = FakeBackend()
        models: dict[int, BackendModel] = {}
        loaded, next_model_id = worker._run_request(
            cast(_BackendRuntime, backend),
            cast(
                WorkerRequest,
                {
                    "operation": "load_model",
                    "arguments": {
                        "model_id": "mini",
                        "temperature": 0.7,
                        "settings": {"voice": "celune", "enabled": True},
                    },
                },
            ),
            models,
            1,
            io.BytesIO(),
        )
        self.assertTrue(loaded["ok"])
        self.assertEqual(backend.loaded[0], "mini")
        loaded_settings = cast(
            dict[str, BackendArgumentValue], backend.loaded[1]["settings"]
        )
        self.assertEqual(loaded_settings["voice"], "celune")

        generated, _ = worker._run_request(
            cast(_BackendRuntime, backend),
            cast(
                WorkerRequest,
                {
                    "operation": "generate_stream",
                    "arguments": {
                        "model_id": cast(int, loaded["value"]),
                        "temperature": 0.4,
                        "decoding": {"top_k": 20},
                    },
                },
            ),
            models,
            next_model_id,
            io.BytesIO(),
        )
        self.assertTrue(generated["done"])
        generated_decoding = cast(
            dict[str, BackendArgumentValue], backend.generated[1]["decoding"]
        )
        self.assertEqual(generated_decoding["top_k"], 20)

        malformed_arguments = (
            {"operation": "load_model", "arguments": {"temperature": 0.7}},
            {
                "operation": "generate_stream",
                "arguments": {"model_id": True},
            },
            {
                "operation": "load_model",
                "arguments": {"model_id": "mini", "bad": object()},
            },
        )
        for request in malformed_arguments:
            with self.subTest(request=request), self.assertRaises(WorkerProtocolError):
                worker._run_request(
                    cast(_BackendRuntime, backend),
                    cast(WorkerRequest, request),
                    models,
                    next_model_id,
                    io.BytesIO(),
                )

    def test_worker_stream_cancellation_during_blocked_next_is_terminal(self) -> None:
        """Verify cancellation during a blocked generator next is terminal."""
        started = threading.Event()
        release = threading.Event()
        cancellation = threading.Event()
        responses: list[WorkerResponse] = []

        class FakeBackend:
            """Backend stand-in whose stream blocks before exhausting."""

            def generate_stream(self, model: BackendModel):
                """Block the generator until cancellation has been requested."""
                del model
                started.set()
                release.wait(timeout=2)
                yield {"audio": [0.0], "sample_rate": 48000}

        def run_stream() -> None:
            response, _ = worker._run_request(
                cast(_BackendRuntime, FakeBackend()),
                {
                    "operation": "generate_stream",
                    "arguments": {"model_id": 1},
                },
                {1: cast(BackendModel, object())},
                2,
                io.BytesIO(),
                cancellation_event=cancellation,
            )
            responses.append(response)

        thread = threading.Thread(target=run_stream)
        thread.start()
        self.assertTrue(started.wait(timeout=1))
        cancellation.set()
        release.set()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(responses, [{"ok": False, "cancelled": True, "done": True}])

    def test_worker_stream_cancellation_after_final_chunk_is_terminal(self) -> None:
        """Verify cancellation requested after the final chunk cannot complete normally."""
        cancellation = threading.Event()

        class FakeBackend:
            """Backend stand-in that requests cancellation while exhausting."""

            def generate_stream(self, model: BackendModel):
                """Yield one chunk, then request cancellation before exhaustion."""
                del model
                yield {"audio": [0.0], "sample_rate": 48000}
                cancellation.set()

        response, _ = worker._run_request(
            cast(_BackendRuntime, FakeBackend()),
            {
                "operation": "generate_stream",
                "arguments": {"model_id": 1},
            },
            {1: cast(BackendModel, object())},
            2,
            io.BytesIO(),
            cancellation_event=cancellation,
        )

        self.assertEqual(response, {"ok": False, "cancelled": True, "done": True})

    def test_worker_protocol_round_trips_audio_on_the_binary_channel(self) -> None:
        """Verify audio metadata stays in JSON while samples use binary framing."""
        audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        message = build_packet(
            "response",
            "convert",
            {"ok": True, "value": AudioOutput(audio, 48000)},
        )
        control, payloads = encode_message(message)
        control_stream = io.BytesIO()
        binary_stream = io.BytesIO()
        send_message(control_stream, control)
        send_payloads(binary_stream, payloads)

        self.assertEqual(
            cast(list, control["payloads"])[0]["media_type"], "audio/pcm_f32le"
        )
        self.assertNotIn(audio.tobytes(), control_stream.getvalue())
        control_stream.seek(0)
        binary_stream.seek(0)
        received_control = receive_message(control_stream)
        received_payloads = receive_payloads(
            binary_stream,
            cast(list, received_control["payloads"]),
        )
        decoded = decode_message(received_control, received_payloads)
        output = cast(AudioOutput, cast(dict, decoded["data"])["value"])
        np.testing.assert_array_equal(output.audio, audio)
        self.assertEqual(output.sample_rate, 48000)

    def test_worker_protocol_preflights_payload_limits_before_materializing_arrays(
        self,
    ) -> None:
        """Verify aggregate and descriptor limits reject before byte conversion."""
        limits = CEDTSLimits(max_aggregate_payload_size=8, max_payload_descriptors=2)
        array = np.zeros(3, dtype=np.float32)
        with (
            mock.patch.object(
                np, "ascontiguousarray", wraps=np.ascontiguousarray
            ) as contiguous,
            self.assertRaises(WorkerProtocolError),
        ):
            encode_message({"value": array}, limits=limits)
        contiguous.assert_not_called()

        limits = CEDTSLimits(max_aggregate_payload_size=8, max_payload_descriptors=1)
        first = np.zeros(1, dtype=np.float32)
        second = np.zeros(2, dtype=np.float32)
        with (
            mock.patch.object(
                np, "ascontiguousarray", wraps=np.ascontiguousarray
            ) as contiguous,
            self.assertRaises(WorkerProtocolError),
        ):
            encode_message(
                {"values": cast(list[WorkerValue], [first, second])}, limits=limits
            )
        self.assertEqual(contiguous.call_count, 1)

        limits = CEDTSLimits(max_aggregate_payload_size=8, max_payload_descriptors=2)
        audio = AudioOutput(np.zeros(3, dtype=np.float32), 48000)
        with (
            mock.patch.object(
                np, "ascontiguousarray", wraps=np.ascontiguousarray
            ) as contiguous,
            self.assertRaises(WorkerProtocolError),
        ):
            encode_message({"value": audio}, limits=limits)
        contiguous.assert_not_called()

        limits = CEDTSLimits(max_binary_frame_size=22)
        with (
            mock.patch.object(
                np, "ascontiguousarray", wraps=np.ascontiguousarray
            ) as contiguous,
            self.assertRaises(WorkerProtocolError),
        ):
            encode_message({"value": np.zeros(1, dtype=np.float32)}, limits=limits)
        contiguous.assert_not_called()

    def test_worker_protocol_rejects_duplicate_typed_tensor_references(self) -> None:
        """Verify one tensor payload cannot be copied more than once."""
        array = np.zeros(2, dtype=np.float32)
        control, payloads = encode_message({"value": array})
        typed_array = cast(dict, control["value"])
        control["value"] = {
            "__cedts_type__": "tuple",
            "items": [typed_array, typed_array],
        }
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                control,
                {payload.descriptor["id"]: payload for payload in payloads},
            )

    def test_worker_protocol_rejects_duplicate_typed_audio_references(self) -> None:
        """Verify one audio payload cannot be normalized into multiple copies."""
        audio = np.zeros(2, dtype=np.float32)
        control, payloads = encode_message({"value": AudioOutput(audio, 48000)})
        typed_audio = cast(dict, control["value"])
        control["value"] = {
            "__cedts_type__": "tuple",
            "items": [typed_audio, typed_audio],
        }
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                control,
                {payload.descriptor["id"]: payload for payload in payloads},
            )

    def test_worker_protocol_rejects_extra_fields_in_typed_payload_references(
        self,
    ) -> None:
        """Verify every typed wrapper accepts only its payload ID reference."""
        audio = np.zeros(2, dtype=np.float32)
        cases = (
            ("numpy_array", {"value": audio}, "payload"),
            ("audio_output", {"value": AudioOutput(audio, 48000)}, "audio"),
            (
                "voice_conversion_request",
                {"value": VoiceConversionRequest(audio, 24000)},
                "source_audio",
            ),
            (
                "backend_generation",
                {"value": (audio, 48000, None)},
                "audio",
            ),
        )
        for wrapper_name, message, reference_field in cases:
            with self.subTest(wrapper=wrapper_name):
                control, payloads = encode_message(
                    cast(dict[str, WorkerValue], message)
                )
                wrapper = cast(dict, control["value"])
                reference = cast(dict, wrapper[reference_field])
                reference["unexpected"] = "rejected"
                with self.assertRaises(WorkerProtocolError):
                    decode_message(
                        control,
                        {payload.descriptor["id"]: payload for payload in payloads},
                    )

    def test_worker_protocol_rejects_duplicate_typed_mixed_wrapper_references(
        self,
    ) -> None:
        """Verify mixed typed wrappers cannot share one audio payload."""
        audio = np.zeros(2, dtype=np.float32)
        control, payloads = encode_message({"value": AudioOutput(audio, 48000)})
        typed_audio = cast(dict, control["value"])
        control["value"] = {
            "__cedts_type__": "tuple",
            "items": [
                typed_audio,
                {
                    "__cedts_type__": "voice_conversion_request",
                    "source_audio": typed_audio["audio"],
                    "sample_rate": 48000,
                    "target_voice": None,
                    "target_character": None,
                    "target_references": [],
                    "label": "audio input",
                    "pitch_shift": None,
                    "f0_condition": None,
                },
            ],
        }
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                control,
                {payload.descriptor["id"]: payload for payload in payloads},
            )

    def test_worker_protocol_enforces_decoded_payload_allocation_limit(self) -> None:
        """Verify decoded tensor and normalized audio bytes use one aggregate bound."""
        arrays = [np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)]
        control, payloads = encode_message({"value": cast(list[WorkerValue], arrays)})
        payload_map = {payload.descriptor["id"]: payload for payload in payloads}
        decode_message(
            control,
            payload_map,
            limits=CEDTSLimits(max_aggregate_payload_size=8),
        )
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                control,
                payload_map,
                limits=CEDTSLimits(max_aggregate_payload_size=7),
            )

        audio_control, audio_payloads = encode_message(
            {
                "value": AudioOutput(
                    cast(np.ndarray, np.zeros(2, dtype=np.int16)),
                    48000,
                )
            }
        )
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                audio_control,
                {payload.descriptor["id"]: payload for payload in audio_payloads},
                limits=CEDTSLimits(max_aggregate_payload_size=4),
            )

    def test_worker_protocol_round_trips_voice_conversion_request_payload(self) -> None:
        """Verify conversion requests reconstruct audio without Python object JSON."""
        request = VoiceConversionRequest(
            source_audio=np.array([0.1, -0.2], dtype=np.float32),
            sample_rate=24000,
            target_voice="celune",
        )
        control, payloads = encode_message(
            build_packet(
                "request",
                "convert",
                cast(
                    dict[str, WorkerValue],
                    {"arguments": {"request": request}},
                ),
            )
        )
        control_stream = io.BytesIO()
        binary_stream = io.BytesIO()
        send_message(control_stream, control)
        send_payloads(binary_stream, payloads)
        control_stream.seek(0)
        binary_stream.seek(0)
        received_control = receive_message(control_stream)
        decoded = decode_message(
            received_control,
            receive_payloads(binary_stream, cast(list, received_control["payloads"])),
        )
        decoded_request = cast(
            VoiceConversionRequest,
            cast(dict, cast(dict, decoded["data"])["arguments"])["request"],
        )
        np.testing.assert_array_equal(
            decoded_request.source_audio, request.source_audio
        )
        self.assertEqual(decoded_request.sample_rate, request.sample_rate)
        self.assertEqual(decoded_request.target_voice, request.target_voice)

    def test_worker_protocol_rejects_binary_after_zero_payload_message(self) -> None:
        """Verify an undeclared binary frame cannot desynchronize later messages."""
        unexpected_frame = (10).to_bytes(4, "big") + (b"x" * 10)
        binary_stream = io.BytesIO(unexpected_frame + (0).to_bytes(4, "big"))

        with self.assertRaises(WorkerProtocolError):
            receive_payloads(binary_stream, [])

        self.assertEqual(receive_payloads(binary_stream, []), {})

    def test_worker_protocol_round_trips_empty_and_multiple_payload_boundaries(
        self,
    ) -> None:
        """Verify empty and multi-payload messages keep their binary boundaries."""
        payloads = (
            WorkerPayload(
                {
                    "id": "first",
                    "media_type": "application/octet-stream",
                    "byte_length": 3,
                },
                b"one",
            ),
            WorkerPayload(
                {
                    "id": "second",
                    "media_type": "application/octet-stream",
                    "byte_length": 3,
                },
                b"two",
            ),
        )
        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "rb", buffering=0)
        writer = os.fdopen(write_fd, "wb", buffering=0)
        try:
            send_payloads(writer, ())
            send_payloads(writer, payloads)
            self.assertEqual(receive_payloads(reader, []), {})
            received = receive_payloads(
                reader, [payload.descriptor for payload in payloads]
            )
        finally:
            reader.close()
            writer.close()

        self.assertEqual(
            {payload_id: payload.data for payload_id, payload in received.items()},
            {"first": b"one", "second": b"two"},
        )

    def test_worker_protocol_validates_audio_wrapper_metadata(self) -> None:
        """Verify audio wrappers agree with their binary payload descriptors."""
        audio = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
        cases = (
            (
                AudioOutput(audio, 48000),
                "response",
                cast(
                    dict[str, WorkerValue],
                    {"value": AudioOutput(audio, 48000)},
                ),
            ),
            (
                VoiceConversionRequest(audio, 24000),
                "request",
                cast(
                    dict[str, WorkerValue],
                    {
                        "arguments": cast(
                            dict[str, WorkerValue],
                            {"request": VoiceConversionRequest(audio, 24000)},
                        )
                    },
                ),
            ),
        )
        for value, packet_kind, data in cases:
            with self.subTest(value_type=type(value).__name__):
                control, payloads = encode_message(
                    build_packet(packet_kind, "convert", data)
                )
                decoded = decode_message(
                    control,
                    {payload.descriptor["id"]: payload for payload in payloads},
                )
                decoded_data = cast(dict[str, WorkerValue], decoded["data"])
                if isinstance(value, AudioOutput):
                    decoded_value = cast(AudioOutput, decoded_data["value"])
                    self.assertEqual(decoded_value.sample_rate, 48000)
                    self.assertEqual(decoded_value.audio.shape, audio.shape)
                else:
                    decoded_arguments = cast(
                        dict[str, WorkerValue], decoded_data["arguments"]
                    )
                    decoded_value = cast(
                        VoiceConversionRequest,
                        decoded_arguments["request"],
                    )
                    self.assertEqual(decoded_value.sample_rate, 24000)
                    self.assertEqual(decoded_value.source_audio.shape, audio.shape)

        generation_control, generation_payloads = encode_message(
            build_packet(
                "response",
                "generate_stream",
                {"value": (audio, 48000, None)},
            )
        )
        encoded_generation = cast(
            dict[str, WorkerValue],
            cast(dict[str, WorkerValue], generation_control["data"])["value"],
        )
        self.assertEqual(encoded_generation["channels"], 2)
        self.assertEqual(encoded_generation["shape"], [2, 2])
        decoded_generation_value = cast(
            tuple[np.ndarray, int, WorkerValue],
            cast(
                dict[str, WorkerValue],
                decode_message(
                    generation_control,
                    {
                        payload.descriptor["id"]: payload
                        for payload in generation_payloads
                    },
                )["data"],
            )["value"],
        )
        self.assertEqual(decoded_generation_value[1], 48000)
        self.assertEqual(decoded_generation_value[0].shape, audio.shape)

        for field_name, invalid_value in (
            ("sample_rate", 44100),
            ("channels", 1),
            ("shape", [4]),
        ):
            with self.subTest(backend_generation_field=field_name):
                invalid_control, invalid_payloads = encode_message(
                    build_packet(
                        "response",
                        "generate_stream",
                        {"value": (audio, 48000, None)},
                    )
                )
                invalid_value_wrapper = cast(
                    dict[str, WorkerValue],
                    cast(dict[str, WorkerValue], invalid_control["data"])["value"],
                )
                invalid_value_wrapper[field_name] = cast(WorkerValue, invalid_value)
                with self.assertRaises(WorkerProtocolError):
                    decode_message(
                        invalid_control,
                        {
                            payload.descriptor["id"]: payload
                            for payload in invalid_payloads
                        },
                    )

        output_control, output_payloads = encode_message(
            build_packet(
                "response",
                "convert",
                {"value": AudioOutput(audio, 48000)},
            )
        )
        cast(dict, cast(dict, output_control["data"])["value"])["sample_rate"] = 44100
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                output_control,
                {payload.descriptor["id"]: payload for payload in output_payloads},
            )

        request_control, request_payloads = encode_message(
            build_packet(
                "request",
                "convert",
                cast(
                    dict[str, WorkerValue],
                    {
                        "arguments": cast(
                            dict[str, WorkerValue],
                            {"request": VoiceConversionRequest(audio, 24000)},
                        )
                    },
                ),
            )
        )
        cast(
            dict,
            cast(dict, cast(dict, request_control["data"])["arguments"])["request"],
        )["sample_rate"] = 22050
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                request_control,
                {payload.descriptor["id"]: payload for payload in request_payloads},
            )

    def test_worker_protocol_rejects_cross_media_typed_wrappers(self) -> None:
        """Verify typed wrappers accept only their declared media family."""
        audio = np.array([0.1, -0.2], dtype=np.float32)
        audio_cases = (
            (
                "response",
                {"value": AudioOutput(audio, 48000)},
            ),
            (
                "request",
                {
                    "arguments": {
                        "request": VoiceConversionRequest(audio, 24000),
                    }
                },
            ),
            (
                "response",
                {"value": (audio, 48000, None)},
            ),
        )
        for packet_kind, data in audio_cases:
            with self.subTest(packet_kind=packet_kind, wrapper=data):
                control, payloads = encode_message(
                    build_packet(packet_kind, "convert", cast(dict, data))
                )
                payload = payloads[0]
                descriptor = dict(payload.descriptor)
                descriptor["media_type"] = "application/x-tensor"
                with self.assertRaises(WorkerProtocolError):
                    decode_message(
                        control,
                        {
                            payload.descriptor["id"]: WorkerPayload(
                                cast(WorkerPayloadDescriptor, descriptor), payload.data
                            )
                        },
                    )

        control, payloads = encode_message(
            build_packet("response", "convert", {"value": audio})
        )
        payload = payloads[0]
        descriptor = dict(payload.descriptor)
        descriptor.update(
            {
                "media_type": "audio/pcm_f32le",
                "sample_rate": 48000,
                "channels": 1,
            }
        )
        with self.assertRaises(WorkerProtocolError):
            decode_message(
                control,
                {
                    payload.descriptor["id"]: WorkerPayload(
                        cast(WorkerPayloadDescriptor, descriptor), payload.data
                    )
                },
            )

    def test_worker_protocol_normalizes_signed_pcm_audio_to_float32(self) -> None:
        """Verify signed PCM audio is normalized at the CEDTS decode boundary."""
        audio = np.array([-32768, 0, 32767], dtype=np.int16)
        control, payloads = encode_message(
            build_packet(
                "response",
                "convert",
                {"ok": True, "value": AudioOutput(cast(np.ndarray, audio), 48000)},
            )
        )
        control_stream = io.BytesIO()
        binary_stream = io.BytesIO()
        send_message(control_stream, control)
        send_payloads(binary_stream, payloads)
        control_stream.seek(0)
        binary_stream.seek(0)

        received_control = receive_message(control_stream)
        decoded = decode_message(
            received_control,
            receive_payloads(binary_stream, cast(list, received_control["payloads"])),
        )
        output = cast(AudioOutput, cast(dict, decoded["data"])["value"])

        self.assertEqual(output.audio.dtype, np.float32)
        np.testing.assert_allclose(
            output.audio,
            np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32),
        )

    def test_worker_protocol_requires_matching_audio_media_type_and_dtype(
        self,
    ) -> None:
        """Verify each audio media type accepts only its matching dtype."""
        invalid_descriptors = (
            {
                "id": "float-audio",
                "media_type": "audio/pcm_f32le",
                "byte_length": 4,
                "dtype": "int16",
                "shape": [1],
                "sample_rate": 48000,
                "channels": 1,
            },
            {
                "id": "pcm-audio",
                "media_type": "audio/pcm_s16le",
                "byte_length": 4,
                "dtype": "float32",
                "shape": [1],
                "sample_rate": 48000,
                "channels": 1,
            },
        )
        for descriptor in invalid_descriptors:
            with (
                self.subTest(descriptor=descriptor),
                self.assertRaises(WorkerProtocolError),
            ):
                validate_payload_descriptors(cast(list, [descriptor]))

    def test_worker_protocol_rejects_invalid_float_audio_samples(self) -> None:
        """Verify decoded float audio is finite and within normalized bounds."""
        invalid_audio = (
            np.array([np.nan], dtype=np.float32),
            np.array([np.inf], dtype=np.float32),
            np.array([-np.inf], dtype=np.float32),
            np.array([1.000001], dtype=np.float32),
            np.array([-1.000001], dtype=np.float32),
        )
        for audio in invalid_audio:
            with self.subTest(audio=audio):
                control, payloads = encode_message(
                    build_packet(
                        "response",
                        "convert",
                        {
                            "ok": True,
                            "value": AudioOutput(audio, 48000),
                        },
                    )
                )
                with self.assertRaises(WorkerProtocolError):
                    decode_message(
                        control,
                        {payload.descriptor["id"]: payload for payload in payloads},
                    )

    def test_worker_protocol_rejects_binary_length_mismatch(self) -> None:
        """Verify a binary frame cannot disagree with its declared payload length."""
        descriptor = {
            "id": "audio-1",
            "media_type": "audio/pcm_f32le",
            "byte_length": 8,
            "dtype": "float32",
            "shape": [2],
            "sample_rate": 48000,
            "channels": 1,
        }
        raw = bytearray()
        raw.extend((18).to_bytes(4, "big"))
        raw.extend((7).to_bytes(2, "big"))
        raw.extend((4).to_bytes(8, "big"))
        raw.extend(b"audio-1")
        raw.extend(b"\x00\x00\x00\x00")
        with self.assertRaises(WorkerProtocolError):
            receive_payloads(io.BytesIO(raw), [descriptor])

    def test_worker_protocol_rejects_unexpected_binary_payload_identity(self) -> None:
        """Verify binary frames must match one declared payload ID exactly once."""
        descriptor = {
            "id": "expected",
            "media_type": "application/octet-stream",
            "byte_length": 3,
        }
        payload_id = b"unexpected"
        payload = b"abc"
        binary_frame = (
            (10 + len(payload_id) + len(payload)).to_bytes(4, "big")
            + len(payload_id).to_bytes(2, "big")
            + len(payload).to_bytes(8, "big")
            + payload_id
            + payload
        )

        with self.assertRaises(WorkerProtocolError):
            receive_payloads(io.BytesIO(binary_frame), [descriptor])

    def test_worker_protocol_rejects_invalid_binary_metadata_and_limits(self) -> None:
        """Verify invalid dtype, shape, audio metadata, and aggregate sizes fail early."""
        invalid_descriptors = (
            {
                "id": "tensor",
                "media_type": "application/x-tensor",
                "byte_length": 3,
                "dtype": "float32",
                "shape": [1],
            },
            {
                "id": "audio",
                "media_type": "audio/pcm_f32le",
                "byte_length": 4,
                "dtype": "float32",
                "shape": [1],
                "channels": 1,
            },
            {
                "id": "unknown",
                "media_type": "application/x-unknown",
                "byte_length": 0,
            },
        )
        for descriptor in invalid_descriptors:
            with (
                self.subTest(descriptor=descriptor),
                self.assertRaises(WorkerProtocolError),
            ):
                validate_payload_descriptors(cast(list, [descriptor]))

        descriptors = [
            {
                "id": f"payload-{index}",
                "media_type": "application/octet-stream",
                "byte_length": 7 * 1024 * 1024,
            }
            for index in range(10)
        ]
        with self.assertRaises(WorkerProtocolError):
            validate_payload_descriptors(descriptors)

    def test_worker_protocol_rejects_payload_data_length_before_transmission(
        self,
    ) -> None:
        """Verify send-side descriptors cannot claim bytes that are not present."""
        payload = WorkerPayload(
            {
                "id": "payload-1",
                "media_type": "application/octet-stream",
                "byte_length": 4,
            },
            b"short",
        )
        with self.assertRaises(WorkerProtocolError):
            send_payloads(io.BytesIO(), [payload])

    def test_worker_stream_uses_protocol_stdout_during_backend_redirects(self) -> None:
        """Verify backend stdout redirection cannot discard streamed protocol frames."""

        class FakeBackend:
            """Backend whose stream represents one generated audio frame."""

            @staticmethod
            def generate_stream(model: object, **kwargs: object):
                """Yield one protocol-compatible fake audio frame."""
                del model, kwargs
                yield {"audio": [0.0], "sample_rate": 48000}

        protocol_stream = io.BytesIO()
        with (
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch.object(worker, "_WORKER_STDERR", io.StringIO()),
        ):
            response, _ = worker._run_request(
                cast(_BackendRuntime, FakeBackend()),
                {"operation": "generate_stream", "arguments": {"model_id": 1}},
                {1: object()},
                2,
                protocol_stream,
            )
            send_message(
                protocol_stream,
                build_packet(
                    "response",
                    "generate_stream",
                    cast(dict[str, WorkerValue], response),
                ),
            )

        assert response["done"]
        protocol_stream.seek(0)
        frame = receive_message(protocol_stream)
        self.assertTrue(cast(dict, frame["data"])["stream"])
        self.assertEqual(
            cast(dict, frame["data"])["value"],
            {"audio": [0.0], "sample_rate": 48000},
        )
        self.assertTrue(cast(dict, receive_message(protocol_stream)["data"])["done"])

    def test_worker_stream_stops_before_backend_next_chunk_when_cancelled(self) -> None:
        """Verify a request-scoped cancellation event terminates a worker stream."""

        class FakeBackend:
            """Backend stand-in for cancellation-aware streaming."""

            @staticmethod
            def generate_stream(model: object, **kwargs: object):
                """Yield a chunk only when the request was not already cancelled."""
                del model, kwargs
                yield {"audio": [0.0], "sample_rate": 48000}

        cancellation = threading.Event()
        cancellation.set()
        response, _ = worker._run_request(
            cast(_BackendRuntime, FakeBackend()),
            {"operation": "generate_stream", "arguments": {"model_id": 1}},
            {1: object()},
            2,
            io.BytesIO(),
            cancellation_event=cancellation,
        )

        self.assertTrue(response["cancelled"])
        self.assertTrue(response["done"])

    def test_remote_cancellation_targets_active_request_without_protocol_lock(
        self,
    ) -> None:
        """Verify cancellation delivery bypasses the lock held by stream consumption."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._process = cast(
            subprocess.Popen[bytes],
            SimpleNamespace(stdin=io.BytesIO(), poll=lambda: None),
        )
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "active-request"
        proxy._cancel_target = None
        proxy._cancel_packet_id = None
        proxy._cancel_ack_event = None
        proxy._cancel_ack_result = None
        proxy._cancel_sent = False
        proxy._protocol_lock = threading.Lock()
        proxy._send_packet = mock.Mock(return_value="cancel-packet")

        proxy._protocol_lock.acquire()
        try:
            self.assertTrue(
                proxy.cancel_active_request(
                    "active-request",
                    wait_for_ack=False,
                )
            )
        finally:
            proxy._protocol_lock.release()

        proxy._send_packet.assert_called_once_with(
            proxy._process.stdin,
            "cancel",
            "cancel",
            {"target_message_id": "active-request"},
            message_id=mock.ANY,
        )

    def test_remote_stale_cancellation_cannot_target_new_stream(self) -> None:
        """Verify an old request ID cannot cancel a newer active stream."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._active_request_lock = threading.Lock()
        proxy._active_request_id = "new-request"
        proxy._cancel_target = None
        proxy._cancel_packet_id = None
        proxy._cancel_ack_event = None
        proxy._cancel_ack_result = None
        proxy._cancel_sent = False
        proxy._send_packet = mock.Mock()

        self.assertFalse(proxy.cancel_active_request("old-request"))
        proxy._send_packet.assert_not_called()

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
            TimeoutError,
            "worker operation 'preload_models' timed out",
        ):
            proxy._request("preload_models", response_timeout=0.01)

        proxy.abort.assert_called_once_with()
        proxy._read_response.assert_called_once_with(
            proxy._process,
            "request-id",
            timeout=0.01,
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
        process = _ShutdownProcess(worker_output)
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
            raise WorkerProtocolError("stream read interrupted by shutdown")

        generator = proxy._stream_request("generate_stream")
        generator_errors: list[Exception] = []

        def consume_generator() -> None:
            """Consume the test generator and retain expected shutdown errors."""
            try:
                next(generator)
            except WorkerProtocolError as error:
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
        process = _ShutdownProcess(io.BytesIO())
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

    def test_remote_proxy_startup_failure_closes_partial_worker(self) -> None:
        """Verify a partial stream setup terminates the worker and closes ownership."""
        process = _ShutdownProcess(io.BytesIO())
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

                        self.assertIsInstance(reader_error, WorkerProtocolError)
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
        reader_error = WorkerProtocolError("worker reader failed")
        proxy._reader_error = None
        waiter_started = threading.Event()
        errors: list[Exception] = []

        def wait_for_event() -> None:
            waiter_started.set()
            try:
                proxy.get_worker_event(timeout=2)
            except WorkerProtocolError as error:
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
        proxy, reader, writer = self._make_packet_reader_proxy(
            event_callback=event_callback,
        )
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

            self.assertIsInstance(reader_error, WorkerProtocolError)
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

            self.assertIsInstance(reader_error, WorkerProtocolError)
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
        )
        negotiated = worker._negotiate_hello(hello)
        self.assertEqual(negotiated["streaming"], True)
        self.assertEqual(negotiated["cancellation"], True)

        unsupported_version = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [2],
                    "capabilities": remote.CORE_CAPABILITIES,
                    "required_capabilities": {"streaming": True},
                },
            ),
        )
        with self.assertRaises(WorkerProtocolError):
            worker._negotiate_hello(unsupported_version)

        incompatible = build_packet(
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [1],
                    "capabilities": remote.CORE_CAPABILITIES,
                    "required_capabilities": {"callback": True},
                },
            ),
        )
        with self.assertRaises(WorkerProtocolError):
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
                    "versions": [1],
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
        with self.assertRaises(WorkerProtocolError):
            send_message(io.BytesIO(), oversized_control, limits=limits)
        control_stream = io.BytesIO()
        send_message(control_stream, oversized_control)
        control_stream.seek(0)
        with self.assertRaises(WorkerProtocolError):
            receive_message(control_stream, limits=limits)

        oversized_payload = WorkerPayload(
            {
                "id": "payload-1",
                "media_type": "application/octet-stream",
                "byte_length": 9,
            },
            b"x" * 9,
        )
        with self.assertRaises(WorkerProtocolError):
            send_payloads(io.BytesIO(), [oversized_payload], limits=limits)
        binary_stream = io.BytesIO()
        send_payloads(binary_stream, [oversized_payload])
        binary_stream.seek(0)
        with self.assertRaises(WorkerProtocolError):
            receive_payloads(
                binary_stream,
                [oversized_payload.descriptor],
                limits=limits,
            )

        with self.assertRaises(WorkerProtocolError):
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
                WorkerProtocolError("unsupported CEDTS version"),
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
                {"cedts_version": 1, "capabilities": remote.CORE_CAPABILITIES},
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
                {"cedts_version": 1, "capabilities": remote.CORE_CAPABILITIES},
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
                {
                    "cedts_version": 1,
                    "capabilities": remote.CORE_CAPABILITIES,
                },
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
        with self.assertRaises(WorkerProtocolError):
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
                {"cedts_version": 1, "capabilities": small_capabilities},
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

    def test_worker_subprocess_completes_cedts_lifecycle_without_backend_dependencies(
        self,
    ) -> None:
        """Verify a real worker process performs handshake, request, and shutdown."""
        if os.name == "nt":
            self.skipTest("the subprocess fixture uses POSIX descriptor inheritance")

        child_code = """
from celune.backends import worker


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
                pass_fds=(worker_binary_input, worker_binary_output),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            worker_binary_input = -1
            worker_binary_output = -1
            assert process.stdin is not None
            assert process.stdout is not None

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
            hello_ack = receive_message(process.stdout)
            ready = receive_message(process.stdout)
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
            response = receive_message(process.stdout)
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
            load_response = receive_message(process.stdout)
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
            shutdown_ack = receive_message(process.stdout)
            self.assertEqual(shutdown_ack["kind"], "shutdown_ack")
            self.assertEqual(shutdown_ack["reply_to"], "shutdown-request")
            self.assertEqual(
                cast(dict, shutdown_ack["data"])["value"]["active_job_cancelled"],
                False,
            )
            self.assertEqual(process.wait(timeout=5), 0)
            assert process.stderr is not None
            self.assertIn(
                "backend load diagnostic",
                process.stderr.read().decode("utf-8", errors="replace"),
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

    def _run_worker_stream_shutdown_policy(
        self, policy: str, *, ignore_cancellation: bool = False
    ) -> tuple[dict[str, WorkerValue], list[WorkerControlMessage]]:
        """Run an active stream through one CEDTS shutdown policy."""
        if os.name == "nt":
            self.skipTest("the subprocess fixture uses POSIX descriptor inheritance")

        child_code = """
import time
from celune.backends import worker


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
                        {"arguments": {"model_id": "test-model"}},
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
                    cast(
                        dict[str, WorkerValue],
                        {"arguments": {"model_id": model_id}},
                    ),
                    message_id="stream-request",
                ),
            )
            first_frame = receive_message(process.stdout)
            self.assertTrue(cast(dict, first_frame["data"])["stream"])
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
                packet = receive_message(process.stdout)
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
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

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
        if os.name == "nt":
            self.skipTest("the subprocess fixture uses POSIX descriptor inheritance")

        child_code = """
import os
import threading
from celune.backends import worker


release_fd = int(os.environ["CELUNE_TEST_RELEASE_FD"])
operation_finished = threading.Event()
response_sent = threading.Event()


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
        print("BLOCKING_STARTED", file=worker._WORKER_STDERR, flush=True)
        try:
            wait_for_release()
        finally:
            operation_finished.set()

    def load_model(self, model_id):
        print("BLOCKING_STARTED", file=worker._WORKER_STDERR, flush=True)
        try:
            wait_for_release()
            return model_id
        finally:
            operation_finished.set()

    def convert(self, request):
        del request
        print("BLOCKING_STARTED", file=worker._WORKER_STDERR, flush=True)
        try:
            wait_for_release()
            return {"converted": True}
        finally:
            operation_finished.set()

    def resolve_generation_language(self, lang):
        print("BLOCKING_STARTED", file=worker._WORKER_STDERR, flush=True)
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
                pass_fds=(worker_binary_input, worker_binary_output, release_read),
            )
            os.close(worker_binary_input)
            os.close(worker_binary_output)
            os.close(release_read)
            worker_binary_input = -1
            worker_binary_output = -1
            release_read = -1
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
            stderr_lines: list[bytes] = []
            while True:
                remaining = blocking_deadline - time.monotonic()
                if remaining <= 0:
                    stderr = b""
                    if process.poll() is not None:
                        stderr = process.stderr.read()
                    self.fail(
                        "blocking backend operation did not start "
                        f"(returncode={process.poll()}, stderr={stderr!r})"
                    )
                ready, _, _ = select.select([process.stderr], [], [], remaining)
                if not ready:
                    self.fail(
                        "blocking backend operation did not start "
                        f"(returncode={process.poll()}, stderr={stderr_lines!r})"
                    )
                line = process.stderr.readline()
                stderr_lines.append(line)
                if b"BLOCKING_STARTED" in line:
                    break
                if not line:
                    self.fail(
                        "blocking backend operation did not start "
                        f"(returncode={process.poll()}, stderr={stderr_lines!r})"
                    )

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

    def test_worker_control_loop_handles_shutdown_during_blocking_operation(
        self,
    ) -> None:
        """Verify shutdown is acknowledged while preload remains blocked."""
        self.assertLess(self._run_blocking_worker_control_test("shutdown"), 1.0)

    def test_worker_rejects_cancel_after_terminal_state_is_marked(self) -> None:
        """Verify a cancel racing with terminal transmission cannot cancel a finished job."""
        if os.name == "nt":
            self.skipTest("the subprocess fixture uses POSIX descriptor inheritance")

        child_code = """
import threading
from celune.backends import worker


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
        with self.assertRaises(WorkerProtocolError):
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
        with self.assertRaises(WorkerProtocolError):
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
                return "/root/VoiceSpeaker/backend_worker_bootstrap.py"

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
            mock.patch.object(remote.os, "name", "nt"),
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

    def test_isolated_backend_resolution_uses_the_registered_manifest(self) -> None:
        """Verify isolated resolution delegates construction to the worker proxy."""
        with mock.patch.object(remote, "RemoteBackendProxy") as proxy:
            resolve_backend("mini", isolated=True)

        proxy.assert_called_once()
        assert proxy.call_args.args[0] == BACKEND_MANIFESTS["mini"]


if __name__ == "__main__":
    unittest.main()
