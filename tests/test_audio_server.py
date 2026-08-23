# SPDX-License-Identifier: Apache-2.0
"""Tests for host audio-server restart helpers."""

import subprocess
from unittest import mock

import pytest

from celune.audio import server

from .support import CeluneTestCase


class TestAudioServer(CeluneTestCase):
    """Verify platform-specific audio-server restart commands."""

    @staticmethod
    def _completed(
        returncode: int = 0, stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        """Build one completed-process fixture."""
        return subprocess.CompletedProcess([], returncode, "", stderr)

    def test_windows_restarts_windows_audio_service(self) -> None:
        """Verify Windows uses PowerShell to restart the Windows Audio service."""
        with (
            mock.patch.object(server.os, "name", "nt"),
            mock.patch.object(server.shutil, "which", return_value="powershell.exe"),
            mock.patch.object(
                server,
                "_run_command",
                return_value=self._completed(),
            ) as run_command,
        ):
            server.restart_audio_server()

        command = run_command.call_args.args[0]
        assert command[0] == "powershell.exe"
        script = command[-1]
        assert "Start-Process" in script
        assert "-Verb RunAs" in script
        assert "-WindowStyle Hidden" in script
        assert "Restart-Service -Name Audiosrv -Force" in script

    def test_linux_restarts_active_user_audio_units(self) -> None:
        """Verify Linux restarts active PipeWire-related user units together."""
        responses = [
            self._completed(0),
            self._completed(0),
            self._completed(3),
            self._completed(3),
            self._completed(0),
        ]
        with (
            mock.patch.object(server.os, "name", "posix"),
            mock.patch.object(server.sys, "platform", "linux"),
            mock.patch.object(
                server.shutil,
                "which",
                side_effect=lambda name: "systemctl" if name == "systemctl" else None,
            ),
            mock.patch.object(
                server,
                "_run_command",
                side_effect=responses,
            ) as run_command,
        ):
            server.restart_audio_server()

        assert run_command.call_args.args[0][2:] == (
            "restart",
            "pipewire.service",
            "pipewire-pulse.service",
        )
        assert run_command.call_args.args[0][0:3] == ("systemctl", "--user", "restart")

    def test_linux_uses_pulseaudio_fallback(self) -> None:
        """Verify Linux can stop a running PulseAudio server without systemd user units."""
        with (
            mock.patch.object(server.os, "name", "posix"),
            mock.patch.object(server.sys, "platform", "linux"),
            mock.patch.object(
                server.shutil,
                "which",
                side_effect=lambda name: (
                    "/usr/bin/pulseaudio" if name == "pulseaudio" else None
                ),
            ),
            mock.patch.object(
                server,
                "_run_command",
                side_effect=[
                    self._completed(0),
                    self._completed(0),
                ],
            ) as run_command,
        ):
            server.restart_audio_server()

        assert run_command.call_args.args[0] == ("/usr/bin/pulseaudio", "--kill")

    def test_unsupported_platform_reports_error(self) -> None:
        """Verify unsupported platforms fail explicitly."""
        with (
            mock.patch.object(server.os, "name", "posix"),
            mock.patch.object(server.sys, "platform", "darwin"),
            pytest.raises(RuntimeError, match="unsupported platform"),
        ):
            server.restart_audio_server()
