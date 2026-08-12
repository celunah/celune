# SPDX-License-Identifier: MIT
"""Tests for host audio-server restart helpers."""

import subprocess
from unittest import TestCase, mock

from celune import audio


class AudioServerTests(TestCase):
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
            mock.patch.object(audio.os, "name", "nt"),
            mock.patch.object(audio.shutil, "which", return_value="powershell.exe"),
            mock.patch.object(
                audio,
                "_run_command",
                return_value=self._completed(),
            ) as run_command,
        ):
            audio.restart_audio_server()

        command = run_command.call_args.args[0]
        self.assertEqual(command[0], "powershell.exe")
        script = command[-1]
        self.assertIn("Start-Process", script)
        self.assertIn("-Verb RunAs", script)
        self.assertIn("-WindowStyle Hidden", script)
        self.assertIn("Restart-Service -Name Audiosrv -Force", script)

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
            mock.patch.object(audio.os, "name", "posix"),
            mock.patch.object(audio.sys, "platform", "linux"),
            mock.patch.object(
                audio.shutil,
                "which",
                side_effect=lambda name: "systemctl" if name == "systemctl" else None,
            ),
            mock.patch.object(
                audio,
                "_run_command",
                side_effect=responses,
            ) as run_command,
        ):
            audio.restart_audio_server()

        self.assertEqual(
            run_command.call_args.args[0][2:],
            ("restart", "pipewire.service", "pipewire-pulse.service"),
        )
        self.assertEqual(
            run_command.call_args.args[0][0:3],
            ("systemctl", "--user", "restart"),
        )

    def test_linux_uses_pulseaudio_fallback(self) -> None:
        """Verify Linux can stop a running PulseAudio server without systemd user units."""
        with (
            mock.patch.object(audio.os, "name", "posix"),
            mock.patch.object(audio.sys, "platform", "linux"),
            mock.patch.object(
                audio.shutil,
                "which",
                side_effect=lambda name: (
                    "/usr/bin/pulseaudio" if name == "pulseaudio" else None
                ),
            ),
            mock.patch.object(
                audio,
                "_run_command",
                side_effect=[
                    self._completed(0),
                    self._completed(0),
                ],
            ) as run_command,
        ):
            audio.restart_audio_server()

        self.assertEqual(
            run_command.call_args.args[0], ("/usr/bin/pulseaudio", "--kill")
        )

    def test_unsupported_platform_reports_error(self) -> None:
        """Verify unsupported platforms fail explicitly."""
        with (
            mock.patch.object(audio.os, "name", "posix"),
            mock.patch.object(audio.sys, "platform", "darwin"),
            self.assertRaisesRegex(RuntimeError, "unsupported platform"),
        ):
            audio.restart_audio_server()
