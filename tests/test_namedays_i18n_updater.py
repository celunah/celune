# SPDX-License-Identifier: MIT
"""Tests for lightweight data, localization, and update helpers."""

import datetime
import json
import subprocess
import tempfile
from unittest import mock, TestCase
from pathlib import Path

from celune import i18n, namedays, updater


class NameDayTests(TestCase):
    """Tests for name-day lookup helpers."""

    def test_lookup_helpers_cover_supported_inputs(self) -> None:
        """Verify date lookup helpers and invalid input handling.

        Raises:
            AssertionError: Name-day lookup behavior changes unexpectedly.
        """
        self.assertEqual(namedays.get_names(5, 16), ["Andrew", "Simon"])
        self.assertEqual(
            namedays.get_names_for_date(datetime.date(2026, 5, 16)),
            ["Andrew", "Simon"],
        )
        self.assertEqual(namedays.get_names_for_date("2026-05-16"), ["Andrew", "Simon"])
        self.assertEqual(namedays.get_names_for_date("05-16"), ["Andrew", "Simon"])
        self.assertEqual(namedays.has_name_day("andrew", "05-16"), True)
        self.assertIn("10-21", namedays.find_dates_for_name("Celine"))
        with self.assertRaises(TypeError):
            namedays.get_names_for_date(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            namedays.get_names_for_date("bad-date")


class I18nTests(TestCase):
    """Tests for localization fallback behavior."""

    def tearDown(self) -> None:
        i18n.set_locale("en")

    def test_string_falls_back_and_formats_values(self) -> None:
        """Verify fallback strings and interpolation.

        Raises:
            AssertionError: Localization behavior changes unexpectedly.
        """
        original = dict(i18n.STRINGS)
        try:
            i18n.STRINGS["en"] = {"hello": "Hello {name}"}
            i18n.STRINGS["pl"] = {}
            i18n.set_locale("pl")
            self.assertEqual(i18n.string("hello", name="Celune"), "Hello Celune")
            self.assertEqual(i18n.string("missing"), "missing")
        finally:
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original)


class UpdaterTests(TestCase):
    """Tests for pure updater decision logic."""

    def test_version_helpers_order_tags(self) -> None:
        """Verify version normalization and ordering helpers.

        Raises:
            AssertionError: Version helper behavior changes unexpectedly.
        """
        self.assertEqual(updater._normalize_tag("refs/tags/v4.0.0"), "4.0.0")
        self.assertEqual(updater._short_revision("abcdef123"), "abcdef1")
        self.assertEqual(updater._short_revision(""), "unknown")
        self.assertEqual(updater._is_newer_version_tag("9.9.9", "4.0.0"), True)
        self.assertEqual(updater._is_newer_version_tag("4.0.0", "4.0.0"), False)

    def test_check_for_update_returns_none_for_dirty_worktree(self) -> None:
        """Verify dirty repositories suppress update prompts.

        Raises:
            AssertionError: Update suppression behavior changes unexpectedly.
        """
        with (
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._current_branch", return_value="main"),
            mock.patch("celune.updater._has_local_changes", return_value=True),
        ):
            self.assertIsNone(updater.check_for_update())

    def test_check_for_update_builds_update_info(self) -> None:
        """Verify update metadata assembly for a newer revision.

        Raises:
            AssertionError: Update metadata behavior changes unexpectedly.
        """
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._current_branch", return_value="main"),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch("celune.updater._local_revision", return_value="a" * 40),
            mock.patch("celune.updater._local_tag", return_value="3.5.0"),
            mock.patch(
                "celune.updater._remote_branch_revision",
                return_value="b" * 40,
            ),
            mock.patch(
                "celune.updater._latest_remote_tag",
                return_value=("4.0.0", "c" * 40),
            ),
            mock.patch(
                "celune.updater._git_succeeds",
                side_effect=[False, True],
            ),
        ):
            update = updater.check_for_update()
        self.assertIsNotNone(update)
        if update is not None:
            self.assertEqual(update.local_revision, "aaaaaaa")
            self.assertEqual(update.latest_revision, "bbbbbbb")
            self.assertEqual(update.latest_version, "4.0.0")

    def test_check_for_update_ignores_local_commits_ahead_of_remote(self) -> None:
        """Verify unpushed local commits do not show as available updates.

        Raises:
            AssertionError: Ahead-of-remote repositories should not prompt updates.
        """
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._current_branch", return_value="main"),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch("celune.updater._local_revision", return_value="a" * 40),
            mock.patch("celune.updater._local_tag", return_value="3.5.0"),
            mock.patch(
                "celune.updater._remote_branch_revision",
                return_value="b" * 40,
            ),
            mock.patch(
                "celune.updater._latest_remote_tag",
                return_value=("3.5.0", "b" * 40),
            ),
            mock.patch(
                "celune.updater._git_succeeds",
                side_effect=[True],
            ),
        ):
            self.assertIsNone(updater.check_for_update())

    def test_has_new_remote_revision_only_for_fast_forward_updates(self) -> None:
        """Verify revision comparison distinguishes ahead vs behind states.

        Raises:
            AssertionError: Fast-forward detection behavior changes unexpectedly.
        """
        with mock.patch(
            "celune.updater._git_succeeds",
            side_effect=[True],
        ):
            self.assertFalse(updater._has_new_remote_revision("a" * 40, "b" * 40))

        with mock.patch(
            "celune.updater._git_succeeds",
            side_effect=[False, True],
        ):
            self.assertTrue(updater._has_new_remote_revision("a" * 40, "b" * 40))

        with mock.patch(
            "celune.updater._git_succeeds",
            side_effect=[False, False],
        ):
            self.assertFalse(updater._has_new_remote_revision("a" * 40, "b" * 40))

    def test_check_for_update_compiled_uses_bundle_checksums(self) -> None:
        """Verify compiled update detection compares bundle checksums against artifact metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            (bundle_dir / "celune.exe").write_bytes(b"launcher-old")
            (bundle_dir / "celune-bin.exe").write_bytes(b"runtime-old")
            manifest = {
                "version": "4.1.0",
                "revision": "a" * 40,
                "artifact": "Celune-win-x64",
                "files": {
                    "celune.exe": updater._sha256_file(bundle_dir / "celune.exe"),
                    "celune-bin.exe": updater._sha256_file(
                        bundle_dir / "celune-bin.exe"
                    ),
                },
            }
            (bundle_dir / updater.UPDATE_MANIFEST_NAME).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            remote = updater.BundleManifest(
                version="4.2.0",
                revision="b" * 40,
                artifact="Celune-win-x64",
                files={
                    "celune.exe": "1" * 64,
                    "celune-bin.exe": "2" * 64,
                },
            )

            with (
                mock.patch("celune.updater.running_compiled", return_value=True),
                mock.patch("celune.updater._bundle_dir", return_value=bundle_dir),
                mock.patch(
                    "celune.updater._read_remote_bundle_manifest", return_value=remote
                ),
                mock.patch("celune.updater._is_git_checkout", return_value=False),
            ):
                update = updater.check_for_update()

        self.assertIsNotNone(update)
        if update is not None:
            self.assertEqual(update.local_revision, "aaaaaaa")
            self.assertEqual(update.latest_revision, "bbbbbbb")
            self.assertEqual(update.latest_version, "4.2.0")

    def test_check_for_update_compiled_returns_none_when_bundle_matches_remote(
        self,
    ) -> None:
        """Verify compiled update checks stay quiet when the local bundle already matches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            (bundle_dir / "celune.exe").write_bytes(b"launcher")
            (bundle_dir / "celune-bin.exe").write_bytes(b"runtime")
            local_files = {
                "celune.exe": updater._sha256_file(bundle_dir / "celune.exe"),
                "celune-bin.exe": updater._sha256_file(bundle_dir / "celune-bin.exe"),
            }
            manifest = {
                "version": "4.1.0",
                "revision": "a" * 40,
                "artifact": "Celune-win-x64",
                "files": local_files,
            }
            (bundle_dir / updater.UPDATE_MANIFEST_NAME).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            remote = updater.BundleManifest(
                version="4.1.0",
                revision="a" * 40,
                artifact="Celune-win-x64",
                files=local_files,
            )

            with (
                mock.patch("celune.updater.running_compiled", return_value=True),
                mock.patch("celune.updater._bundle_dir", return_value=bundle_dir),
                mock.patch(
                    "celune.updater._read_remote_bundle_manifest", return_value=remote
                ),
                mock.patch("celune.updater._is_git_checkout", return_value=False),
            ):
                self.assertIsNone(updater.check_for_update())

    def test_update_to_latest_rejects_unsafe_states(self) -> None:
        """Verify unsafe repository states reject automatic updates.

        Raises:
            AssertionError: Update safety behavior changes unexpectedly.
        """
        with (
            mock.patch("celune.updater._is_git_checkout", return_value=False),
            self.assertRaisesRegex(updater.UpdateError, "did not find"),
        ):
            updater.update_to_latest()

        with (
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._has_local_changes", return_value=True),
            self.assertRaisesRegex(updater.UpdateError, "not committed"),
        ):
            updater.update_to_latest()

        with (
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch(
                "celune.updater._current_branch",
                side_effect=subprocess.TimeoutExpired("git", 5),
            ),
            self.assertRaisesRegex(updater.UpdateError, "timed out"),
        ):
            updater.update_to_latest()
