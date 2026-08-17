# SPDX-License-Identifier: MIT
"""Tests for the CI process runner's interruption cleanup."""

from unittest import TestCase, mock

from scripts import run_ci


class RunCiTests(TestCase):
    """Verify CI process cleanup behavior."""

    def test_windows_start_keeps_console_interrupts_on_runner(self) -> None:
        """Verify the child is not isolated from the runner's console group."""
        process = mock.Mock()

        with (
            mock.patch.object(run_ci.os, "name", "nt"),
            mock.patch.object(
                run_ci.subprocess, "Popen", return_value=process
            ) as popen,
            mock.patch.object(run_ci, "_attach_windows_job"),
        ):
            self.assertIs(run_ci._start_process(), process)

        popen.assert_called_once_with(run_ci.POE_COMMAND, text=True)

    def test_windows_cleanup_kills_tree_after_wrapper_exits(self) -> None:
        """Verify fallback cleanup still runs when the wrapper has exited."""
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = 0

        with (
            mock.patch.object(run_ci.os, "name", "nt"),
            mock.patch.object(run_ci, "_close_windows_job", return_value=False),
            mock.patch.object(run_ci.subprocess, "run") as taskkill,
            mock.patch.object(run_ci.subprocess, "CREATE_NO_WINDOW", 0, create=True),
        ):
            run_ci.stop_process_tree(process)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            stdout=run_ci.subprocess.DEVNULL,
            stderr=run_ci.subprocess.DEVNULL,
            check=False,
            creationflags=0,
        )
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=run_ci.GRACE_PERIOD)

    def test_posix_cleanup_uses_immediate_process_group_kill(self) -> None:
        """Verify POSIX cleanup does not let Poe advance to another task."""
        process = mock.Mock()
        process.pid = 1234

        with (
            mock.patch.object(run_ci.os, "name", "posix"),
            mock.patch.object(
                run_ci.os, "getpgid", return_value=5678, create=True
            ) as getpgid,
            mock.patch.object(run_ci.os, "killpg", create=True) as killpg,
        ):
            run_ci.stop_process_tree(process)

        getpgid.assert_called_once_with(1234)
        sigkill = getattr(run_ci.signal, "SIGKILL", run_ci.signal.SIGTERM)
        killpg.assert_called_once_with(5678, sigkill)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=run_ci.GRACE_PERIOD)

    def test_wait_for_process_polls_until_child_finishes(self) -> None:
        """Verify the runner does not block signal handling in one long wait."""
        process = mock.Mock()
        process.args = ["uv", "run", "poe", "ci"]
        process.wait.side_effect = [
            run_ci.subprocess.TimeoutExpired(process.args, 0.1),
            0,
        ]

        with mock.patch.object(run_ci.time, "monotonic", side_effect=(0.0, 0.0, 0.2)):
            exit_code = run_ci._wait_for_process(process, 600.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(process.wait.call_count, 2)
