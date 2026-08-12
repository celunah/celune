# SPDX-License-Identifier: Apache-2.0

import os
from unittest import TestCase, mock

from celune import watchdog
from celune.constants import ExitCodes


class WatchdogTests(TestCase):
    """Verify the launcher-loss watchdog's process-facing behavior."""

    def setUp(self) -> None:
        """Clear the process-wide launcher-loss request before each test."""
        watchdog._LAUNCHER_LOST.clear()

    def test_launcher_lost_exit_code_is_eight(self) -> None:
        """Verify launcher loss uses the reserved exit code eight."""
        self.assertEqual(ExitCodes.EXIT_LAUNCHER_LOST.value, 8)

    def test_posix_pipe_eof_exits_with_launcher_lost_code(self) -> None:
        """Verify EOF on the launcher pipe invokes the immediate exit path."""
        read_fd, write_fd = os.pipe()
        os.close(write_fd)

        with mock.patch.object(
            watchdog,
            "_request_launcher_lost",
            side_effect=RuntimeError("launcher lost"),
        ) as exit_launcher_lost:
            with self.assertRaisesRegex(RuntimeError, "launcher lost"):
                watchdog._watch_posix_pipe(str(read_fd))

        exit_launcher_lost.assert_called_once_with()

    def test_pipe_loss_requests_coordinated_shutdown_without_hard_exit(self) -> None:
        """Verify pipe loss sets the shutdown request instead of aborting the process."""
        read_fd, write_fd = os.pipe()
        os.close(write_fd)

        watchdog._watch_posix_pipe(str(read_fd))

        self.assertTrue(watchdog.launcher_loss_requested())

    @staticmethod
    def test_watchdog_starts_for_configured_posix_pipe() -> None:
        """Verify a configured POSIX pipe is handed to a daemon watcher thread."""
        with (
            mock.patch.dict(
                watchdog.os.environ,
                {"CELUNE_LAUNCHER_PIPE_FD": "17"},
                clear=True,
            ),
            mock.patch.object(watchdog.os, "name", "posix"),
            mock.patch.object(watchdog.threading, "Thread") as thread,
        ):
            watchdog.start_watchdog()

        thread.assert_called_once_with(
            target=watchdog._watch_posix_pipe,
            args=("17",),
            daemon=True,
            name="celune-launcher-watchdog",
        )
        thread.return_value.start.assert_called_once_with()
