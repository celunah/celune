# SPDX-License-Identifier: Apache-2.0
"""Tests for SoulX-Singer's pinned source checkout."""

from pathlib import Path
from unittest import mock
from tempfile import TemporaryDirectory

from celune.backends.tts import soulxsinger


def test_soulxsinger_source_publication_retries_windows_sharing_violation() -> None:
    """Retry a transient Windows lock while publishing the source checkout."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        staging_root = root / "staging"
        source_root = root / "source"
        staging_root.mkdir()
        (staging_root / "checkout.py").touch()
        attempts = 0
        replace = Path.replace

        def replace_after_transient_lock(source: Path, destination: Path) -> Path:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                error = PermissionError("sharing violation")
                error.winerror = 32
                raise error
            return replace(source, destination)

        with (
            mock.patch.object(Path, "replace", replace_after_transient_lock),
            mock.patch.object(soulxsinger.time, "sleep") as sleep,
        ):
            soulxsinger._publish_upstream_source(staging_root, source_root)

        assert attempts == 2
        sleep.assert_called_once_with(soulxsinger._SOURCE_PUBLICATION_RETRY_DELAY)
        assert (source_root / "checkout.py").is_file()
