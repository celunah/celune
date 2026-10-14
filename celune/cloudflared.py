# SPDX-License-Identifier: Apache-2.0
"""Manage a temporary Cloudflare quick tunnel for Celune's public mode."""

import os
import re
import atexit
import shutil
import threading
import contextlib
import subprocess
from collections.abc import Callable
from typing import Optional

from .i18n import string
from .constants import APP_NAME

_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b", re.IGNORECASE)
_STARTUP_ANNOUNCEMENT_TIMEOUT = 10.0


class CloudflaredQuickTunnel:
    """Own one cloudflared quick-tunnel process for Celune's lifetime."""

    def __init__(self, log: Callable[[str, str], None]) -> None:
        """Create a tunnel manager.

        Args:
            log: Callback used to report tunnel status and its public URL.
        """
        self._log = log
        self._process: Optional[subprocess.Popen[str]] = None
        self._reader: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._port: Optional[int] = None
        self._startup_announced = False
        self._startup_timer: Optional[threading.Timer] = None

    def start(self, port: int) -> bool:
        """Start the public quick tunnel and watch for its endpoint URL.

        Args:
            port: Local API port to expose through Cloudflare.

        Returns:
            bool: ``True`` when cloudflared started, otherwise ``False``.
        """
        with self._lock:
            if self._closed:
                return False
            if self._process is not None and self._process.poll() is None:
                return True

            executable = shutil.which("cloudflared")
            if executable is None:
                self._log(
                    string("celune.cloudflared_missing", app_name=APP_NAME),
                    "warning",
                )
                return False

            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            )
            try:
                process = subprocess.Popen(  # pylint: disable=R1732
                    [executable, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
            except OSError:
                self._log(
                    string("celune.cloudflared_failed", app_name=APP_NAME),
                    "warning",
                )
                return False

            self._stopping.clear()
            self._process = process
            self._port = port
            self._startup_announced = False
            atexit.register(self.stop)
            self._startup_timer = threading.Timer(
                _STARTUP_ANNOUNCEMENT_TIMEOUT,
                self._announce_local_start,
            )
            self._startup_timer.daemon = True
            self._startup_timer.start()
            self._reader = threading.Thread(
                target=self._read_output,
                args=(process,),
                daemon=True,
                name=f"{APP_NAME}Cloudflared",
            )
            self._reader.start()
        return True

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        """Read cloudflared output without blocking Celune's API thread."""
        output = process.stdout
        if output is not None:
            for line in output:
                match = _TUNNEL_URL.search(line)
                if match is not None:
                    url = match.group(0)
                    if not self._announce_startup(url):
                        self._log(
                            string(
                                "celune.public_api_endpoint",
                                app_name=APP_NAME,
                                url=url,
                            ),
                            "info",
                        )

        return_code = process.wait()
        if not self._stopping.is_set():
            self._announce_local_start()
        if return_code != 0 and not self._stopping.is_set():
            self._log(string("celune.cloudflared_failed", app_name=APP_NAME), "warning")

    def _announce_local_start(self) -> None:
        """Report the local API URL when the tunnel URL is not ready in time."""
        with self._lock:
            port = self._port
        if port is not None:
            self._announce_startup(f"http://127.0.0.1:{port}")

    def _announce_startup(self, url: str) -> bool:
        """Log the first usable API URL, returning whether this call announced it."""
        with self._lock:
            if self._startup_announced or self._stopping.is_set():
                return False
            self._startup_announced = True
            timer = self._startup_timer
            self._startup_timer = None

        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        self._log(
            string("celune.api_started", app_name=APP_NAME, url=url),
            "info",
        )
        return True

    def stop(self) -> None:
        """Terminate the quick tunnel and wait for its process to exit."""
        with self._lock:
            self._closed = True
            process = self._process
            timer = self._startup_timer
            self._startup_timer = None
            if timer is not None:
                timer.cancel()
            if process is None:
                return

            atexit.unregister(self.stop)
            self._stopping.set()
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)

            reader = self._reader
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=1)
            self._process = None
            self._reader = None
