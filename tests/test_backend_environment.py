# SPDX-License-Identifier: MIT
"""Tests for isolated backend environment metadata and installation."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from celune.backends import remote
from celune.backends.environment import (
    BACKEND_MANIFESTS,
    BackendEnvironmentManager,
    BackendManifest,
    backend_manifest,
)
from celune.backends.tts import resolve_backend
from celune.backends.worker_protocol import receive_message, send_message


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

    def test_fingerprint_changes_when_requirements_change(self) -> None:
        """Verify dependency changes select a different environment directory."""
        first = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
        second = BackendManifest("test", "tts", ("demo==2",), "module", "Backend")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

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

    def test_ensure_uses_the_core_venv_interpreter_by_default(self) -> None:
        """Verify compiled launchers do not pass their executable to uv as Python."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = BackendEnvironmentManager(root=root, uv_executable="uv")
            manifest = BackendManifest("test", "tts", ("demo==1",), "module", "Backend")
            core_python = root / "core" / "python.exe"

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
                    "celune.backends.environment.core_python_executable",
                    return_value=core_python,
                ),
                mock.patch(
                    "celune.backends.environment.subprocess.run",
                    side_effect=fake_run,
                ) as run,
            ):
                manager.ensure(manifest)

            self.assertEqual(run.call_args_list[0].args[0][3], str(core_python))

    def test_worker_protocol_round_trips_messages(self) -> None:
        """Verify worker messages survive framing and serialization."""
        stream = io.BytesIO()
        message = {"operation": "describe", "value": ["mini", 1]}
        send_message(stream, message)
        stream.seek(0)
        self.assertEqual(receive_message(stream), message)

    def test_isolated_backend_resolution_uses_the_registered_manifest(self) -> None:
        """Verify isolated resolution delegates construction to the worker proxy."""
        with mock.patch.object(remote, "RemoteBackendProxy") as proxy:
            resolve_backend("mini", isolated=True)

        proxy.assert_called_once()
        self.assertEqual(proxy.call_args.args[0], BACKEND_MANIFESTS["mini"])


if __name__ == "__main__":
    unittest.main()
