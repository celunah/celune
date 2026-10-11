# SPDX-License-Identifier: Apache-2.0
"""Tests for the managed Cloudflare quick-tunnel process."""

import io
import threading
from unittest import mock

from celune.cloudflared import CloudflaredQuickTunnel


def test_quick_tunnel_uses_the_local_api_and_closes_its_process() -> None:
    """Verify the public tunnel points at loopback and is stopped on close."""
    messages: list[tuple[str, str]] = []
    startup_reported = threading.Event()

    def log(message: str, severity: str) -> None:
        messages.append((message, severity))
        if "API has started on https://violet-moon.trycloudflare.com" in message:
            startup_reported.set()

    process = mock.Mock()
    process.stdout = io.StringIO(
        "Quick Tunnel ready at https://violet-moon.trycloudflare.com\n"
    )
    process.wait.return_value = 0
    process.poll.return_value = None
    tunnel = CloudflaredQuickTunnel(log)

    with (
        mock.patch("celune.cloudflared.shutil.which", return_value="cloudflared"),
        mock.patch(
            "celune.cloudflared.subprocess.Popen", return_value=process
        ) as popen,
    ):
        assert tunnel.start(2060)
        assert startup_reported.wait(timeout=1.0)
        tunnel.stop()

    assert popen.call_args.args[0] == [
        "cloudflared",
        "tunnel",
        "--url",
        "http://127.0.0.1:2060",
    ]
    process.terminate.assert_called_once()
    assert len(messages) == 1


def test_quick_tunnel_falls_back_to_the_local_api_url() -> None:
    """Verify a tunnel that exits without a URL reports the loopback API URL."""
    messages: list[tuple[str, str]] = []
    startup_reported = threading.Event()

    def log(message: str, severity: str) -> None:
        messages.append((message, severity))
        if "API has started on http://127.0.0.1:2060" in message:
            startup_reported.set()

    process = mock.Mock()
    process.stdout = io.StringIO()
    process.wait.return_value = 0
    process.poll.return_value = None
    tunnel = CloudflaredQuickTunnel(log)

    with (
        mock.patch("celune.cloudflared.shutil.which", return_value="cloudflared"),
        mock.patch("celune.cloudflared.subprocess.Popen", return_value=process),
    ):
        assert tunnel.start(2060)
        assert startup_reported.wait(timeout=1.0)
        tunnel.stop()

    assert len(messages) == 1


def test_quick_tunnel_reports_missing_cloudflared() -> None:
    """Verify public mode remains closed when cloudflared is unavailable."""
    messages: list[tuple[str, str]] = []
    tunnel = CloudflaredQuickTunnel(
        lambda message, severity: messages.append((message, severity))
    )

    with mock.patch("celune.cloudflared.shutil.which", return_value=None):
        assert not tunnel.start(2060)

    assert len(messages) == 1
    assert messages[0][1] == "warning"
