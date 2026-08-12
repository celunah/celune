# SPDX-License-Identifier: MIT
"""Tests for isolated backend environment metadata and installation."""

import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from celune.backends import environment, remote, worker
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendEnvironment,
    BackendEnvironmentManager,
    BackendEnvironmentError,
    BackendManifest,
    _exclusive_lock,
    backend_manifest,
)
from celune.backends.tts import resolve_backend
from celune.backends.worker_protocol import receive_message, send_message
from celune.typing.backends import BackendModel, _BackendRuntime
from celune.typing import WorkerMessage


class BackendEnvironmentTests(unittest.TestCase):
    """Verify backend environment paths and installation transactions."""

    def test_manifests_cover_installed_backend_extras(self) -> None:
        """Verify every supported optional backend has a manifest."""
        self.assertEqual(
            set(BACKEND_MANIFESTS),
            {"mini", "qwen3", "dotstts", "voxcpm2", "gpt-sovits", "seed-vc"},
        )

    def test_manifest_lookup_normalizes_backend_id(self) -> None:
        """Verify manifest lookup accepts surrounding whitespace and case changes."""
        self.assertIs(backend_manifest(" QWEN3 "), BACKEND_MANIFESTS["qwen3"])

    def test_manifests_use_the_cuda_pytorch_index(self) -> None:
        """Verify isolated backends use the main branch's CUDA 12.8 stack."""
        expected_requirements = {
            "torch==2.11.0+cu128",
            "torchaudio==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
        }
        for manifest in BACKEND_MANIFESTS.values():
            self.assertIn(
                "https://download.pytorch.org/whl/cu128",
                manifest.index_urls,
            )
            self.assertTrue(expected_requirements.issubset(manifest.requirements))

    def test_manifests_use_the_main_branch_huggingface_versions(self) -> None:
        """Verify isolated backends use the main branch's Hugging Face ranges."""
        expected_requirements = {
            "huggingface-hub>=0.36,<1.0.0",
            "transformers>=4.56,<5.0.0",
        }
        for manifest in BACKEND_MANIFESTS.values():
            self.assertTrue(expected_requirements.issubset(manifest.requirements))

    def test_manifests_pin_the_main_branch_librosa_stack(self) -> None:
        """Verify isolated backends pin the compatible librosa dependency chain."""
        expected_requirements = {
            "librosa==0.11.0",
            "llvmlite==0.47.0",
            "numba==0.65.1",
        }
        for manifest in BACKEND_MANIFESTS.values():
            self.assertTrue(expected_requirements.issubset(manifest.requirements))

    def test_dotstts_uses_the_celune_fork(self) -> None:
        """Verify dots.tts is installed from Celune's maintained fork."""
        self.assertIn(
            "dots.tts @ git+https://github.com/celunah/dots.tts",
            BACKEND_MANIFESTS["dotstts"].requirements,
        )

    def test_backend_dependency_list_matches_main_backend_normalizers(self) -> None:
        """Verify the backend dependency list follows the main branch declarations."""
        self.assertNotIn("WeTextProcessing", BACKEND_MANIFESTS["dotstts"].requirements)
        self.assertIn("jieba", BACKEND_MANIFESTS["gpt-sovits"].requirements)
        self.assertIn("split-lang", BACKEND_MANIFESTS["gpt-sovits"].requirements)
        self.assertIn("matplotlib", BACKEND_MANIFESTS["gpt-sovits"].requirements)
        self.assertIn("torchcodec", BACKEND_MANIFESTS["gpt-sovits"].requirements)

    def test_fingerprint_changes_when_requirements_change(self) -> None:
        """Verify dependency changes select a different environment directory."""
        first = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
        second = BackendManifest("test", "tts", ("demo==2",), "module", "Backend")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

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

            self.assertEqual(run.call_count, 2)
            self.assertIn("--no-config", run.call_args_list[1].args[0])
            self.assertIn("--no-cache", run.call_args_list[1].args[0])
            self.assertNotIn("--no-deps", run.call_args_list[1].args[0])

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
            environment = manager.environment_for(manifest)

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

            self.assertEqual(result, environment)
            self.assertTrue(result.is_ready)
            self.assertEqual(
                json.loads(result.metadata_path.read_text(encoding="utf-8"))[
                    "fingerprint"
                ],
                manifest.fingerprint(),
            )
            self.assertEqual(run.call_count, 2)
            install_command = run.call_args_list[1].args[0]
            strategy_index = install_command.index("--index-strategy")
            self.assertEqual(
                install_command[strategy_index + 1],
                "unsafe-best-match",
            )
            self.assertIn("--index-url", install_command)
            self.assertFalse(
                any(
                    path.name.startswith(f"{environment.root.name}.install-")
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
            environment = manager.environment_for(manifest)
            environment.python.parent.mkdir(parents=True)
            environment.python.touch()
            environment.metadata_path.write_text("{}", encoding="utf-8")

            with mock.patch("celune.backends.environment.subprocess.run") as run:
                self.assertEqual(manager.ensure(manifest), environment)

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

            self.assertEqual(run.call_args_list[0].args[0][3], "3.13")

    def test_exclusive_lock_times_out_while_another_handle_owns_it(self) -> None:
        """Verify the operating-system lock blocks a second installer."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "install.lock"
            with _exclusive_lock(lock_path, timeout=1.0):
                with self.assertRaises(BackendEnvironmentError):
                    with _exclusive_lock(lock_path, timeout=0.01):
                        pass

    def test_uv_timeout_becomes_backend_environment_error(self) -> None:
        """Verify stalled uv operations release installation with a clear error."""
        manager = BackendEnvironmentManager(uv_executable="uv", uv_timeout=1.5)
        with mock.patch(
            "celune.backends.environment.subprocess.run",
            side_effect=subprocess.TimeoutExpired("uv", 1.5),
        ) as run:
            with self.assertRaisesRegex(
                BackendEnvironmentError,
                "uv operation timed out",
            ):
                manager._run_uv("venv")

        self.assertEqual(run.call_args.kwargs["timeout"], 1.5)
        self.assertNotIn("PYTHONHOME", run.call_args.kwargs["env"])

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
        self.assertFalse(
            any(name.startswith(("PIP_", "UV_")) for name in child_environment)
        )
        for variable in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
        ):
            self.assertNotIn(variable, child_environment)

    def test_worker_protocol_round_trips_messages(self) -> None:
        """Verify worker messages survive framing and serialization."""
        stream = io.BytesIO()
        message = cast(WorkerMessage, {"operation": "describe", "value": ["mini", 1]})
        send_message(stream, message)
        stream.seek(0)
        self.assertEqual(receive_message(stream), message)

    def test_worker_stream_uses_protocol_stdout_during_backend_redirects(self) -> None:
        """Verify backend stdout redirection cannot discard streamed protocol frames."""

        class FakeBackend:
            """Backend whose stream represents one generated audio frame."""

            @staticmethod
            def generate_stream(model: object, **kwargs: object):
                """Yield one protocol-compatible fake audio frame."""
                del model, kwargs
                yield (b"audio", 48000, None)

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
            send_message(protocol_stream, cast(WorkerMessage, response))

        self.assertTrue(response["done"])
        protocol_stream.seek(0)
        frame = receive_message(protocol_stream)
        self.assertTrue(frame["stream"])
        self.assertEqual(frame["value"], (b"audio", 48000, None))
        self.assertTrue(receive_message(protocol_stream)["done"])

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

            def unload_model(self) -> None:
                """Leave model-table cleanup to the worker."""

            @staticmethod
            def load_model(**_kwargs: object) -> FakeModel:
                """Return one runtime owned by the worker model table."""
                return FakeModel()

        models: dict[int, BackendModel] = {}
        loaded, next_model_id = worker._run_request(
            cast(_BackendRuntime, FakeBackend()),
            {"operation": "load_model", "arguments": {}},
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

        self.assertTrue(loaded["ok"])
        self.assertTrue(unloaded["ok"])
        self.assertTrue(model.closed)
        self.assertEqual(models, {})

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

        self.assertEqual(models, {})

    def test_remote_proxy_handles_fatal_frames_out_of_band(self) -> None:
        """Verify fatal worker notifications do not consume the response frame."""
        stream = io.BytesIO()
        send_message(stream, cast(WorkerMessage, {"ok": True, "fatal": True}))
        send_message(stream, cast(WorkerMessage, {"ok": True, "value": "ready"}))
        stream.seek(0)
        callback = mock.Mock()
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._fatal_callback = callback

        response = proxy._read_response(
            cast(subprocess.Popen[bytes], SimpleNamespace(stdout=stream))
        )

        callback.assert_called_once_with()
        self.assertEqual(response["value"], "ready")

    def test_remote_proxy_recreates_worker_builtin_exception_types(self) -> None:
        """Verify worker ValueErrors remain ValueErrors across the proxy boundary."""
        error = remote._worker_exception("builtins.ValueError", "bad input")

        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "bad input")

    def test_remote_proxy_does_not_drain_after_consuming_worker_error_frame(
        self,
    ) -> None:
        """Verify worker errors reach the caller without waiting for a done frame."""
        stream = io.BytesIO()
        send_message(
            stream,
            cast(
                WorkerMessage,
                {
                    "ok": False,
                    "error": "missing codec",
                    "error_type": "builtins.ImportError",
                },
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

        with mock.patch.object(proxy, "_drain_stream") as drain:
            with self.assertRaisesRegex(ImportError, "missing codec"):
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

        self.assertEqual(
            [call.args[1] for call in log.call_args_list],
            ["error", "error", "error", "info"],
        )

    def test_remote_proxy_uses_backend_error_for_unknown_worker_exception_types(
        self,
    ) -> None:
        """Verify unknown backend exception classes retain their qualified name."""
        error = remote._worker_exception("backend_pkg.ModelError", "generation failed")

        self.assertIsInstance(error, remote.BackendError)
        self.assertEqual(
            str(error),
            "backend_pkg.ModelError: generation failed",
        )

    def test_remote_proxy_does_not_inherit_core_pythonhome(self) -> None:
        """Verify workers do not mix the core and backend Python runtimes."""
        proxy = object.__new__(remote.RemoteBackendProxy)
        proxy._manifest = BACKEND_MANIFESTS["mini"]
        environment = BackendEnvironment(BACKEND_MANIFESTS["mini"], Path("C:/backend"))
        process = SimpleNamespace(stderr=None)

        with (
            mock.patch.dict(
                remote.os.environ,
                {"PYTHONHOME": "C:/Python314", "PYTHONPATH": "C:/existing"},
                clear=True,
            ),
            mock.patch.object(
                remote.subprocess, "Popen", return_value=process
            ) as popen,
        ):
            proxy._start_worker(
                environment,
                lambda msg, severity="info", *, loglevel="info": None,
                {},
            )

        self.assertNotIn("PYTHONHOME", popen.call_args.kwargs["env"])
        self.assertIn("C:/existing", popen.call_args.kwargs["env"]["PYTHONPATH"])
        self.assertEqual(popen.call_args.args[0][2], "--backend")

    def test_isolated_backend_resolution_uses_the_registered_manifest(self) -> None:
        """Verify isolated resolution delegates construction to the worker proxy."""
        with mock.patch.object(remote, "RemoteBackendProxy") as proxy:
            resolve_backend("mini", isolated=True)

        proxy.assert_called_once()
        self.assertEqual(proxy.call_args.args[0], BACKEND_MANIFESTS["mini"])


if __name__ == "__main__":
    unittest.main()
