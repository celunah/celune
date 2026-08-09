# SPDX-License-Identifier: MIT
"""Tests for isolated backend environment metadata and installation."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from celune.backends import remote
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

    def test_audio_compatibility_is_enabled_for_librosa_backends(self) -> None:
        """Verify affected workers opt into the isolated audio compatibility layer."""
        self.assertEqual(
            {
                backend_id
                for backend_id, manifest in BACKEND_MANIFESTS.items()
                if manifest.install_librosa_compat
            },
            {"qwen3", "voxcpm2", "dotstts"},
        )

    def test_manifests_use_the_cuda_pytorch_index(self) -> None:
        """Verify isolated backends resolve the project's CUDA PyTorch wheels."""
        for backend_id, manifest in BACKEND_MANIFESTS.items():
            if backend_id == "mini":
                continue
            self.assertIn(
                "https://download.pytorch.org/whl/cu130",
                manifest.index_urls,
            )

    def test_dotstts_uses_the_celune_fork(self) -> None:
        """Verify dots.tts is installed from Celune's maintained fork."""
        self.assertEqual(
            BACKEND_MANIFESTS["dotstts"].no_deps_requirements,
            ("dots.tts @ git+https://github.com/celunah/dots.tts@main",),
        )

    def test_text_normalizer_builds_are_not_required_by_backends(self) -> None:
        """Verify disabled backend normalizers cannot pull native build dependencies."""
        self.assertNotIn("WeTextProcessing", BACKEND_MANIFESTS["dotstts"].requirements)
        self.assertNotIn("split-lang", BACKEND_MANIFESTS["gpt-sovits"].requirements)

    def test_fingerprint_changes_when_requirements_change(self) -> None:
        """Verify dependency changes select a different environment directory."""
        first = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
        second = BackendManifest("test", "tts", ("demo==2",), "module", "Backend")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_ensure_installs_no_dependency_requirements_separately(self) -> None:
        """Verify selected packages can be installed without transitive dependencies."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest(
                "test",
                "tts",
                ("runtime==1",),
                "module",
                "Backend",
                no_deps_requirements=("wrapper==1",),
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

            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[2].args[0][5], "--no-deps")

    def test_ensure_installs_into_a_temporary_environment_then_publishes_it(
        self,
    ) -> None:
        """Verify uv commands and metadata are written only after installation succeeds."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
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

    def test_worker_protocol_round_trips_messages(self) -> None:
        """Verify worker messages survive framing and serialization."""
        stream = io.BytesIO()
        message = cast(WorkerMessage, {"operation": "describe", "value": ["mini", 1]})
        send_message(stream, message)
        stream.seek(0)
        self.assertEqual(receive_message(stream), message)

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
            proxy._start_worker(environment, lambda _message, _severity: None, {})

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
