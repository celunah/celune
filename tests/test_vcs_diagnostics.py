# SPDX-License-Identifier: Apache-2.0
"""Tests for Git checkout diagnostics shown by Celune."""

from typing import cast
from unittest import mock
from types import SimpleNamespace

from celune import runtime
from celune.i18n import string
from celune.vcs import git_checkout_hint

from .support import CeluneTestCase


class TestRuntimeGitDiagnostics(CeluneTestCase):
    """Verify source launches explain when Git metadata is unavailable."""

    def test_source_runtime_banner_shows_upstream_clone_instructions(self) -> None:
        """Verify a ZIP-based source tree receives a warning without blocking start."""
        messages: list[tuple[str, str]] = []
        backend = cast(
            runtime.CeluneBackend[str],
            SimpleNamespace(is_fake=False, name="test"),
        )
        with (
            mock.patch.object(runtime, "running_compiled", return_value=False),
            mock.patch.object(runtime, "is_git_checkout", return_value=False),
        ):
            runtime.log_runtime_banner(
                lambda message, level: messages.append((message, level)),
                backend,
            )

        assert (string("git.checkout_missing"), "warning") in messages
        assert (git_checkout_hint(), "warning") in messages

    def test_compiled_runtime_does_not_require_local_git_metadata(self) -> None:
        """Verify release bundles do not warn about their absent source checkout."""
        messages: list[tuple[str, str]] = []
        backend = cast(
            runtime.CeluneBackend[str],
            SimpleNamespace(is_fake=False, name="test"),
        )
        with (
            mock.patch.object(runtime, "running_compiled", return_value=True),
            mock.patch.object(runtime, "is_git_checkout", return_value=False),
        ):
            runtime.log_runtime_banner(
                lambda message, level: messages.append((message, level)),
                backend,
            )

        assert (string("git.checkout_missing"), "warning") not in messages
