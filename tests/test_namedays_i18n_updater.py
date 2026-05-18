# SPDX-License-Identifier: MIT
"""Tests for lightweight data, localization, and update helpers."""

import datetime
import subprocess
from unittest import mock, TestCase

from celune import i18n, namedays, updater


class NameDayTests(TestCase):
    """Tests for name-day lookup helpers."""

    def test_lookup_helpers_cover_supported_inputs(self) -> None:
        """Verify date lookup helpers and invalid input handling.

        Returns:
            None: Assertions verify lookup behavior.

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

        Returns:
            None: Assertions verify localization behavior.

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

        Returns:
            None: Assertions verify version helper behavior.

        Raises:
            AssertionError: Version helper behavior changes unexpectedly.
        """
        self.assertEqual(updater._normalize_tag("refs/tags/v3.5.0"), "3.5.0")
        self.assertEqual(updater._short_revision("abcdef123"), "abcdef1")
        self.assertEqual(updater._short_revision(""), "unknown")
        self.assertEqual(updater._is_newer_version_tag("3.5.1", "3.5.0"), True)
        self.assertEqual(updater._is_newer_version_tag("3.5.0", "3.5.1"), False)

    def test_check_for_update_returns_none_for_dirty_worktree(self) -> None:
        """Verify dirty repositories suppress update prompts.

        Returns:
            None: Assertions verify update suppression behavior.

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

        Returns:
            None: Assertions verify update metadata behavior.

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
                return_value=("3.5.1", "c" * 40),
            ),
        ):
            update = updater.check_for_update()
        self.assertIsNotNone(update)
        if update is not None:
            self.assertEqual(update.local_revision, "aaaaaaa")
            self.assertEqual(update.latest_revision, "bbbbbbb")
            self.assertEqual(update.latest_version, "3.5.1")

    def test_update_to_latest_rejects_unsafe_states(self) -> None:
        """Verify unsafe repository states reject automatic updates.

        Returns:
            None: Assertions verify update safety behavior.

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
