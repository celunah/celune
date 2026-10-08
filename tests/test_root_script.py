# SPDX-License-Identifier: Apache-2.0
"""Tests for the source-build repository marker helper."""

import io
import contextlib
import subprocess
from pathlib import Path
from unittest import mock

from scripts import root
from celune.i18n import string
from celune.vcs import git_checkout_hint

from .support import CeluneTestCase


class TestRootScript(CeluneTestCase):
    """Verify Git-dependent helper commands fail with clone instructions."""

    def test_revision_reports_clone_instructions_when_git_checkout_is_missing(
        self,
    ) -> None:
        """Verify the build helper replaces a Git traceback with recovery steps."""
        error_output = io.StringIO()
        with (
            mock.patch.object(root, "_repository_root", return_value=Path(".")),
            mock.patch.object(
                root,
                "_git_revision",
                side_effect=subprocess.CalledProcessError(128, "git rev-parse HEAD"),
            ),
            contextlib.redirect_stderr(error_output),
        ):
            exit_code = root.main(["--revision"])

        assert exit_code == 1
        assert string("git.checkout_required") in error_output.getvalue()
        assert git_checkout_hint() in error_output.getvalue()

    def test_revision_prints_the_commit_for_build_metadata(self) -> None:
        """Verify build scripts can retrieve the checked-out full revision."""
        output = io.StringIO()
        revision = "a" * 40
        with (
            mock.patch.object(root, "_repository_root", return_value=Path(".")),
            mock.patch.object(root, "_git_revision", return_value=revision),
            contextlib.redirect_stdout(output),
        ):
            exit_code = root.main(["--revision"])

        assert exit_code == 0
        assert output.getvalue().strip() == revision
