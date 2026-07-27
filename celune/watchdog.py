# SPDX-License-Identifier: MIT
"""Request coordinated runtime shutdown when the native launcher disappears."""

import os
import threading

_LAUNCHER_LOST = threading.Event()


def _request_launcher_lost() -> None:
    """Request coordinated shutdown because the native launcher disappeared."""
    _LAUNCHER_LOST.set()


def launcher_loss_requested() -> bool:
    """Return whether the native launcher has disconnected."""
    return _LAUNCHER_LOST.is_set()


def _watch_posix_pipe(pipe_fd_text: str) -> None:
    """Block until the launcher closes the inherited POSIX pipe."""
    pipe_fd = -1
    try:
        pipe_fd = int(pipe_fd_text)
        while os.read(pipe_fd, 1):
            pass
    except (OSError, ValueError):
        pass
    finally:
        if pipe_fd >= 0:
            try:
                os.close(pipe_fd)
            except OSError:
                pass

    _request_launcher_lost()


def _watch_windows_pipe(pipe_name: str) -> None:
    """Block until the launcher closes the Windows named pipe."""
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "WinDLL"):
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.WaitNamedPipeW.restype = wintypes.BOOL

    invalid_handle = ctypes.c_void_p(-1).value
    handle = None
    while handle in (None, invalid_handle):
        handle = kernel32.CreateFileW(
            pipe_name,
            0x80000000,  # GENERIC_READ
            0,
            None,
            3,  # OPEN_EXISTING
            0,
            None,
        )
        if handle not in (None, invalid_handle):
            break

        if not kernel32.WaitNamedPipeW(pipe_name, 5000):
            _request_launcher_lost()
            return

    try:
        buffer = ctypes.create_string_buffer(1)
        bytes_read = wintypes.DWORD()  # pylint: disable=E1120
        while (
            kernel32.ReadFile(
                handle,
                buffer,
                1,
                ctypes.byref(bytes_read),
                None,
            )
            and bytes_read.value > 0
        ):
            pass
    finally:
        kernel32.CloseHandle(handle)

    _request_launcher_lost()


def start_watchdog() -> None:
    """Start a daemon thread that watches the launcher-owned pipe, if configured."""
    if os.name != "nt":
        pipe_fd_text = os.getenv("CELUNE_LAUNCHER_PIPE_FD")
        if pipe_fd_text is None:
            return
        watcher = _watch_posix_pipe
        argument = pipe_fd_text
    else:
        pipe_name = os.getenv("CELUNE_LAUNCHER_PIPE")
        if pipe_name is None:
            return
        watcher = _watch_windows_pipe
        argument = pipe_name

    threading.Thread(
        target=watcher,
        args=(argument,),
        daemon=True,
        name="celune-launcher-watchdog",
    ).start()
