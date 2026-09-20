# SPDX-License-Identifier: Apache-2.0
"""Tests for isolated backend environment metadata and installation."""

import io
import os
import json
import tempfile
import threading
import subprocess
from types import SimpleNamespace
from typing import IO, Optional, cast
from pathlib import Path
from unittest import mock
from collections import OrderedDict, deque
from collections.abc import Iterator

import numpy as np

from celune.cedts import remote, worker
from celune.backends import environment
from celune.exceptions import (
    CEDTSError,
    CEDTSEOFError,
    CEDTSStreamError,
    CEDTSPayloadError,
    CEDTSTimeoutError,
    CEDTSProtocolError,
    BackendEnvironmentError,
)
from celune.typing.worker import (
    WorkerValue,
    WorkerRequest,
    WorkerResponse,
    WorkerPayloadDescriptor,
)
from celune.cedts.protocol import (
    CEDTSLimits,
    WorkerPayload,
    build_packet,
    send_message,
    send_payloads,
    decode_message,
    encode_message,
    receive_message,
    receive_payloads,
    validate_payload_descriptors,
)
from celune.typing.backends import (
    BackendModel,
    BackendGeneration,
    BackendArgumentValue,
    _BackendRuntime,
)
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendManifest,
    BackendEnvironmentManager,
    _exclusive_lock,
    backend_manifest,
)
from celune.dataclasses.pipeline import AudioOutput, VoiceConversionRequest

from .support import CeluneTestCase
from .backend_processes import ShutdownProcess

_ShutdownProcess = ShutdownProcess


class TestBackendEnvironment(CeluneTestCase):
    """Verify backend environment paths and installation transactions."""

    def test_remote_bundle_voice_uses_worker_model_for_custom_name(self) -> None:
        """Verify pack-local voice names do not use the static model map."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy.uses_voice_bundles = True
        proxy.clone_model_id = None
        proxy.model_name = "shared-model"
        proxy.voice_models = {"balanced": "shared-model"}
        proxy.default_voice = "balanced"

        self.assertEqual(proxy.model_id_for_voice("Standard"), "shared-model")

    def test_remote_bundle_voice_switch_passes_active_pack_to_worker(self) -> None:
        """Verify switching from a pack default voice to another voice keeps the worker on that pack."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy.uses_voice_bundles = True
        proxy._stream_request = mock.Mock(return_value=iter(()))
        model = remote.RemoteModelHandle(1)
        bundle_path = Path("custom.cevoice")

        with mock.patch(
            "celune.cedts.remote.active_bundle_path", return_value=bundle_path
        ):
            list(proxy.generate_stream(model, text="A", voice="Standard"))
            list(proxy.generate_stream(model, text="A", voice="Alternate"))

        calls = proxy._stream_request.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["voice"], "Standard")
        self.assertEqual(calls[1].kwargs["voice"], "Alternate")
        self.assertEqual(calls[1].kwargs["voice_bundle"], str(bundle_path))
        self.assertEqual(calls[1].kwargs["model_id"], 1)

    def test_worker_switches_from_pack_default_to_non_default_voice(self) -> None:
        """Verify the worker selects the active pack before each requested voice."""

        class FakeBackend:
            """Backend stand-in that records the requested pack voices."""

            generated_voices: list[str]

            def __init__(self) -> None:
                self.generated_voices = []

            def generate_stream(
                self,
                model: BackendModel,
                **kwargs: BackendArgumentValue,
            ) -> Iterator[BackendGeneration]:
                """Record one voice request without generating audio."""
                del model
                self.generated_voices.append(cast(str, kwargs["voice"]))
                yield from ()

        backend = FakeBackend()
        request_arguments = {
            "model_id": 1,
            "text": "A",
            "voice_bundle": str(Path("custom.cevoice")),
        }

        with mock.patch("celune.cedts.worker.select_voice_bundle") as select_bundle:
            for voice in ("Standard", "Alternate"):
                arguments = dict(request_arguments, voice=voice)
                response, _ = worker._run_request(
                    cast(_BackendRuntime, backend),
                    cast(
                        WorkerRequest,
                        {"operation": "generate_stream", "arguments": arguments},
                    ),
                    {1: cast(BackendModel, object())},
                    2,
                    io.BytesIO(),
                )
                self.assertTrue(response["done"])

        self.assertEqual(backend.generated_voices, ["Standard", "Alternate"])
        self.assertEqual(
            select_bundle.call_args_list,
            [mock.call("custom.cevoice"), mock.call("custom.cevoice")],
        )

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

    def test_worker_diagnostic_text_is_not_localized(self) -> None:
        """Keep protocol diagnostics out of the user-facing translation table."""
        translations = json.loads(
            Path("celune/lang/en.json").read_text(encoding="utf-8")
        )

        self.assertFalse(
            any(
                key.startswith(("backends.cedts", "backends.worker"))
                for key in translations
            )
        )

    def test_manifests_cover_installed_backend_extras(self) -> None:
        """Verify every supported optional backend has a manifest."""
        assert set(BACKEND_MANIFESTS) == {
            "mini",
            "qwen3",
            "fireredtts3",
            "dotstts",
            "voxcpm2",
            "luxtts",
            "seed-vc",
        }

    def test_manifest_lookup_normalizes_backend_id(self) -> None:
        """Verify manifest lookup accepts surrounding whitespace and case changes."""
        assert backend_manifest(" QWEN3 ") is BACKEND_MANIFESTS["qwen3"]

    def test_worker_registry_contains_only_approved_backends(self) -> None:
        """Verify worker construction is limited to the seven CEDTS backends."""
        self.assertEqual(
            set(worker._BACKEND_REGISTRY),
            {
                "mini",
                "qwen3",
                "fireredtts3",
                "dotstts",
                "voxcpm2",
                "luxtts",
                "seed-vc",
            },
        )
        self.assertEqual(
            {
                backend_id: kind
                for backend_id, (kind, _loader) in worker._BACKEND_REGISTRY.items()
            },
            {
                "mini": "tts",
                "qwen3": "tts",
                "fireredtts3": "tts",
                "dotstts": "tts",
                "voxcpm2": "tts",
                "luxtts": "tts",
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
        with self.assertRaises(CEDTSProtocolError):
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

    def test_manifests_use_the_main_branch_pytorch_stack(self) -> None:
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
        """Verify standard isolated backends use the main Hugging Face ranges."""
        expected_requirements = {
            "huggingface-hub>=0.36,<1.0.0",
            "hf-xet",
            "transformers>=4.56,<5.0.0",
        }
        standard_manifests = (
            manifest
            for backend_id, manifest in BACKEND_MANIFESTS.items()
            if backend_id != "fireredtts3"
        )
        for manifest in standard_manifests:
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

    def test_fireredtts3_uses_its_isolated_huggingface_contract(self) -> None:
        """Verify FireRedTTS3 gets an isolated Transformers 5 worker contract."""
        requirements = BACKEND_MANIFESTS["fireredtts3"].requirements

        assert "huggingface-hub>=1.5.0,<2.0.0" in requirements
        assert "huggingface-hub>=0.36,<1.0.0" not in requirements
        assert "transformers==5.6.2" in requirements
        assert "transformers>=4.56,<5.0.0" not in requirements
        assert "einops==0.8.2" in requirements
        assert "flash_attn==2.8.3" not in requirements
        assert "torchcodec==0.16.0" in requirements

    def test_backend_dependency_list_matches_main_backend_normalizers(self) -> None:
        """Verify the backend dependency list follows the main branch declarations."""
        assert "WeTextProcessing" not in BACKEND_MANIFESTS["dotstts"].requirements
        luxtts_requirements = BACKEND_MANIFESTS["luxtts"].requirements
        assert "onnxruntime" in luxtts_requirements
        assert (
            "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"
            in luxtts_requirements
        )
        assert (
            "linacodec @ git+https://github.com/ysharma3501/LinaCodec.git"
            in luxtts_requirements
        )
        assert {
            "torch==2.11.0+cu128",
            "torchaudio==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
        }.issubset(luxtts_requirements)
        assert BACKEND_MANIFESTS["luxtts"].ignore_uv_sources
        assert BACKEND_MANIFESTS["luxtts"].find_links == (
            "https://k2-fsa.github.io/icefall/piper_phonemize.html",
        )

    def test_manifest_find_links_are_forwarded_to_uv(self) -> None:
        """Verify a backend can provide wheel links outside package indexes."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest(
                "test",
                "tts",
                ("demo==1",),
                "module",
                "Backend",
                find_links=("https://example.com/wheels.html",),
            )

            def fake_run(command: list[str], **_kwargs) -> None:
                if command[1] == "venv":
                    backend_environment = manager.environment_for(manifest)
                    relative_python = backend_environment.python.relative_to(
                        backend_environment.virtualenv
                    )
                    virtualenv_python = Path(command[-1]) / relative_python
                    virtualenv_python.parent.mkdir(parents=True, exist_ok=True)
                    virtualenv_python.touch()

            with mock.patch(
                "celune.backends.environment.subprocess.run", side_effect=fake_run
            ) as run:
                manager.ensure(manifest)

            install_command = run.call_args_list[1].args[0]
            find_links_index = install_command.index("--find-links")
            assert install_command[find_links_index + 1] == (
                "https://example.com/wheels.html"
            )

    def test_fingerprint_changes_when_requirements_change(self) -> None:
        """Verify dependency changes select a different environment directory."""
        first = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
        second = BackendManifest("test", "tts", ("demo==2",), "module", "Backend")
        assert first.fingerprint() != second.fingerprint()

    def test_fingerprint_changes_when_uv_sources_are_ignored(self) -> None:
        """Verify resolver source policy selects a distinct environment."""
        first = BackendManifest("test", "tts", (), "module", "Backend")
        second = BackendManifest(
            "test",
            "tts",
            (),
            "module",
            "Backend",
            ignore_uv_sources=True,
        )
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
                ignore_uv_sources=True,
            )

            def fake_run(command: list[str], **_kwargs) -> None:
                if command[1] == "venv":
                    backend_environment = manager.environment_for(manifest)
                    relative_python = backend_environment.python.relative_to(
                        backend_environment.virtualenv
                    )
                    virtualenv_python = Path(command[-1]) / relative_python
                    virtualenv_python.parent.mkdir(parents=True, exist_ok=True)
                    virtualenv_python.touch()

            with mock.patch(
                "celune.backends.environment.subprocess.run", side_effect=fake_run
            ) as run:
                manager.ensure(manifest)

            assert run.call_count == 2
            assert "--no-config" in run.call_args_list[1].args[0]
            assert "--no-cache" in run.call_args_list[1].args[0]
            assert "--no-sources" in run.call_args_list[1].args[0]
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
                    backend_environment = manager.environment_for(manifest)
                    relative_python = backend_environment.python.relative_to(
                        backend_environment.virtualenv
                    )
                    virtualenv_python = Path(command[-1]) / relative_python
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
                    backend_environment = manager.environment_for(manifest)
                    relative_python = backend_environment.python.relative_to(
                        backend_environment.virtualenv
                    )
                    virtualenv_python = Path(command[-1]) / relative_python
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

    def test_uv_diagnostics_are_decoded_as_utf8(self) -> None:
        """Verify uv's Unicode resolver diagnostics survive Windows decoding."""
        manager = BackendEnvironmentManager(uv_executable="uv")
        diagnostic = "× No solution found\n╰─▶ Because the requirements conflict."
        with (
            mock.patch(
                "celune.backends.environment.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "uv", stderr=diagnostic),
            ) as run,
            self.assertRaisesRegex(BackendEnvironmentError, "× No solution found"),
        ):
            manager._run_uv("pip", "install")

        assert run.call_args.kwargs["encoding"] == "utf-8"
        assert run.call_args.kwargs["errors"] == "replace"

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

    def test_cedts_errors_share_a_common_typed_base(self) -> None:
        """Verify every public CEDTS error is catchable as CEDTSError."""
        errors = (
            CEDTSEOFError(),
            CEDTSTimeoutError("load_model", 180.0),
            CEDTSProtocolError(packet_name="hello"),
            CEDTSPayloadError(packet_name="generate_stream"),
            CEDTSStreamError("stream closed"),
        )

        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.assertIsInstance(error, CEDTSError)
        self.assertIsInstance(errors[1], TimeoutError)
        self.assertIsInstance(errors[4], OSError)
        self.assertEqual(str(errors[0]), "unexpected EOF while reading stream")
        self.assertEqual(str(errors[1]), "load_model timed out after 180 seconds")
        self.assertEqual(str(errors[2]), "invalid packet hello")
        self.assertEqual(
            str(errors[3]),
            "invalid binary payload received while processing generate_stream",
        )

    def test_worker_protocol_distinguishes_eof_protocol_and_payload_errors(
        self,
    ) -> None:
        """Verify CEDTS classifies its three packet-boundary failure families."""
        with self.assertRaises(CEDTSEOFError):
            receive_message(io.BytesIO())
        with self.assertRaises(CEDTSProtocolError):
            build_packet("invalid", "protocol")
        with self.assertRaises(CEDTSPayloadError):
            validate_payload_descriptors(
                cast(
                    list[WorkerPayloadDescriptor],
                    [
                        {
                            "id": "audio",
                            "media_type": "audio/pcm_f32le",
                            "byte_length": 4,
                            "dtype": "float32",
                            "shape": [2],
                            "sample_rate": 48000,
                            "channels": 1,
                        }
                    ],
                )
            )

    def test_worker_protocol_wraps_stream_write_failures(self) -> None:
        """Verify a closed CEDTS output pipe becomes a stream error."""

        class _ClosedStream:
            def write(self, value: bytes) -> int:
                """Reject every write as a closed-pipe failure."""
                del value
                raise OSError("pipe is closed")

            def flush(self) -> None:
                """Provide the flush method required by the CEDTS writer."""
                return

        with self.assertRaises(CEDTSStreamError) as context:
            send_message(
                cast(IO[bytes], _ClosedStream()),
                build_packet("ping", "protocol"),
            )
        self.assertIn("pipe is closed", str(context.exception))

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

        with self.assertRaises(CEDTSError):
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
                self.assertRaises(CEDTSError),
            ):
                receive_message(io.BytesIO(raw_frame))

    def test_worker_protocol_rejects_bounded_json_limit_violations(self) -> None:
        """Verify nesting, collection, string, and packet-schema limits fail safely."""

        def frame(payload: bytes) -> bytes:
            """Build one length-prefixed control frame for the test."""
            return len(payload).to_bytes(4, "big") + payload

        nested_packet = (
            b'{"cedts_version":[1,1],"kind":"request","message_id":"nested",'
            b'"reply_to":null,"operation":"describe","data":{"arguments":'
            + b'{"value":'
            + b"[" * 65
            + b"null"
            + b"]" * 65
            + b"}}}"
        )
        oversized_collection = {
            "cedts_version": [1, 1],
            "kind": "request",
            "message_id": "collection",
            "reply_to": None,
            "operation": "describe",
            "data": {"arguments": {"value": list(range(1025))}},
        }
        unknown_packet_field = {
            "cedts_version": [1, 1],
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
            with self.subTest(value=value), self.assertRaises(CEDTSError):
                receive_message(io.BytesIO(frame(value)))

        oversized_string = json.dumps(
            {
                "cedts_version": [1, 1],
                "kind": "request",
                "message_id": "string",
                "reply_to": None,
                "operation": "describe",
                "data": {"arguments": {"value": "x" * (1024 * 1024 + 1)}},
            }
        ).encode()
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
            worker._run_request(
                backend,
                {"operation": "unknown", "arguments": {}},
                {},
                1,
                io.BytesIO(),
            )
        with self.assertRaises(CEDTSError):
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

    def test_protocol_accepts_runtime_quantization_callbacks(self) -> None:
        """Verify quantization control callbacks pass the parent packet validator."""
        for method in (
            "runtime_quantization_active",
            "disable_runtime_quantization",
        ):
            with self.subTest(method=method):
                arguments = cast(dict[str, WorkerValue], {"method": method})
                packet = build_packet(
                    "request",
                    "call",
                    {"arguments": arguments},
                )
                send_message(io.BytesIO(), packet)

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
            with self.subTest(request=request), self.assertRaises(CEDTSError):
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
                        "versions": [[1, 1]],
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
            with self.subTest(packet=packet), self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
            with self.subTest(request=request), self.assertRaises(CEDTSError):
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
            self.assertRaises(CEDTSError),
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
            self.assertRaises(CEDTSError),
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
            self.assertRaises(CEDTSError),
        ):
            encode_message({"value": audio}, limits=limits)
        contiguous.assert_not_called()

        limits = CEDTSLimits(max_binary_frame_size=22)
        with (
            mock.patch.object(
                np, "ascontiguousarray", wraps=np.ascontiguousarray
            ) as contiguous,
            self.assertRaises(CEDTSError),
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
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
                with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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

        with self.assertRaises(CEDTSError):
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
                with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
                with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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
                self.assertRaises(CEDTSError),
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
                            "value": AudioOutput(np.zeros(1, dtype=np.float32), 48000),
                        },
                    )
                )
                payload = payloads[0]
                invalid_payload = WorkerPayload(
                    payload.descriptor,
                    audio.tobytes(),
                )
                with self.assertRaises(CEDTSError):
                    decode_message(
                        control,
                        {invalid_payload.descriptor["id"]: invalid_payload},
                    )

    def test_worker_protocol_normalizes_float_audio_before_transmission(self) -> None:
        """Verify worker-produced float audio cannot poison the response stream."""
        control, payloads = encode_message(
            {
                "value": AudioOutput(
                    np.array([np.nan, 2.0, -2.0], dtype=np.float32),
                    48000,
                )
            }
        )

        decoded = decode_message(
            control,
            {payload.descriptor["id"]: payload for payload in payloads},
        )
        audio = cast(AudioOutput, decoded["value"]).audio

        assert np.all(np.isfinite(audio))
        assert float(np.max(np.abs(audio))) <= 0.95

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
        with self.assertRaises(CEDTSError):
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

        with self.assertRaises(CEDTSError):
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
                self.assertRaises(CEDTSError),
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
        with self.assertRaises(CEDTSError):
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
        with self.assertRaises(CEDTSError):
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

        with proxy._protocol_lock:
            self.assertTrue(
                proxy.cancel_active_request(
                    "active-request",
                    wait_for_ack=False,
                )
            )

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
