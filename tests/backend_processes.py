# SPDX-License-Identifier: Apache-2.0
"""Reusable process fixtures for backend protocol tests."""

import io
import os
import subprocess
from typing import IO, Optional

from celune.cedts.protocol import encode_message, send_message
from celune.typing.worker import WorkerMessage


def send_no_payload_packet(
    control_stream: IO[bytes], binary_fd: int, packet: WorkerMessage
) -> None:
    """Send a control packet and its empty binary-channel boundary in tests."""
    control, payloads = encode_message(packet)
    if payloads:
        raise AssertionError("test helper only supports packets without payloads")
    send_message(control_stream, control)
    os.write(binary_fd, b"\x00\x00\x00\x00")


class ShutdownProcess:
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
