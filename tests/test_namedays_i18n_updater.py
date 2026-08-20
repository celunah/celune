# SPDX-License-Identifier: MIT
"""Tests for lightweight data, localization, and update helpers."""

import json
import stat
import zipfile
import datetime
import tempfile
import subprocess
from pathlib import Path
from unittest import TestCase, mock

from celune import i18n, updater, namedays


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

    def test_string_falls_back_from_specific_locale_to_base_language(self) -> None:
        """Verify locale variants can reuse a base-language translation table.

        Raises:
            AssertionError: Locale normalization behavior changes unexpectedly.
        """
        original = dict(i18n.STRINGS)
        try:
            i18n.STRINGS["en"] = {"hello": "Hello"}
            i18n.set_locale("en-US")
            self.assertEqual(i18n.string("hello"), "Hello")
        finally:
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original)

    def test_get_system_locale_prefers_loaded_base_language_candidate(self) -> None:
        """Verify system locale detection tries specific and base candidates before English."""
        original = dict(i18n.STRINGS)
        try:
            i18n.STRINGS.clear()
            i18n.STRINGS["en"] = {"hello": "Hello"}
            i18n.STRINGS["pl"] = {"hello": "Cześć"}
            with mock.patch(
                "celune.i18n._locale.getlocale", return_value=("pl_PL", None)
            ):
                self.assertEqual(i18n.get_system_locale(), "pl")
        finally:
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original)

    def test_get_system_locale_returns_english_after_trying_missing_candidates(
        self,
    ) -> None:
        """Verify missing locale files are tried before English fallback is used."""
        original = dict(i18n.STRINGS)
        try:
            i18n.STRINGS.clear()
            i18n.STRINGS["en"] = {
                "hello": "Hello",
                "celune.locale_not_found": "Missing locale: {locale}",
            }
            with (
                mock.patch(
                    "celune.i18n._locale.getlocale", return_value=("fr_CA", None)
                ),
                mock.patch("sys.stderr.write") as stderr_write,
            ):
                self.assertEqual(i18n.get_system_locale(), "en")
            stderr_write.assert_not_called()
        finally:
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original)


class UpdaterTests(TestCase):
    """Tests for pure updater decision logic."""

    def test_missing_update_metadata_uses_active_locale(self) -> None:
        """Verify missing update metadata is reported through the locale table."""
        original_strings = dict(i18n.STRINGS)
        original_locale = i18n.get_locale()
        try:
            i18n.STRINGS["zz"] = {
                "cli.update_missing_metadata": "localized metadata missing",
            }
            i18n.set_locale("zz")
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                zip_path = root / "artifact.zip"
                destination = root / "destination"
                destination.mkdir()
                with zipfile.ZipFile(zip_path, "w"):
                    pass

                with self.assertRaisesRegex(
                    updater.UpdateError,
                    "localized metadata missing",
                ):
                    updater._extract_artifact_root(zip_path, destination)
        finally:
            i18n.set_locale(original_locale)
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original_strings)

    def test_extract_artifact_rejects_unsafe_zip_members(self) -> None:
        """Reject traversal, absolute, and symbolic-link ZIP members before extraction."""
        unsafe_members = (
            "../escape.txt",
            "/absolute.txt",
            "C:/absolute.txt",
            r"\\server\share\absolute.txt",
            "release/../../escape.txt",
            r"release\..\escape.txt",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for member_name in unsafe_members:
                zip_path = root / "artifact.zip"
                destination = root / "destination"
                destination.mkdir()
                with zipfile.ZipFile(zip_path, "w") as archive:
                    archive.writestr(member_name, b"unsafe")

                with self.subTest(member_name=member_name):
                    with self.assertRaises(updater.UpdateError):
                        updater._extract_artifact_root(zip_path, destination)
                    self.assertFalse((root / "escape.txt").exists())
                destination.rmdir()
                zip_path.unlink()

            zip_path = root / "symlink-artifact.zip"
            destination = root / "symlink-destination"
            destination.mkdir()
            symlink = zipfile.ZipInfo("release/link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(symlink, "../../escape.txt")

            with self.assertRaises(updater.UpdateError):
                updater._extract_artifact_root(zip_path, destination)
            self.assertFalse((root / "escape.txt").exists())

    def test_extract_artifact_rejects_existing_symlink_escape(self) -> None:
        """Reject a member whose destination path traverses an existing symlink."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "destination"
            outside = root / "outside"
            destination.mkdir()
            outside.mkdir()
            try:
                (destination / "release").symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("symbolic links are not supported")

            zip_path = root / "artifact.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("release/escaped.txt", b"unsafe")

            with self.assertRaises(updater.UpdateError):
                updater._extract_artifact_root(zip_path, destination)
            self.assertFalse((outside / "escaped.txt").exists())

    def test_zip_safety_errors_use_localized_messages(self) -> None:
        """Verify ZIP safety failures resolve their user-facing messages through i18n."""
        symlink = zipfile.ZipInfo("release/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = (
            (
                zipfile.ZipInfo("../escape.txt"),
                "cli.update_unsafe_zip_member_path",
            ),
            (symlink, "cli.update_unsafe_zip_symbolic_link"),
        )

        for member, key in cases:
            with self.subTest(key=key):
                with (
                    mock.patch.object(
                        updater, "string", wraps=i18n.string
                    ) as localized_string,
                    self.assertRaises(updater.UpdateError) as raised,
                ):
                    updater._validate_zip_member(member, Path("destination"))

                self.assertEqual(
                    str(raised.exception),
                    i18n.string(key, member=repr(member.filename)),
                )
                localized_string.assert_called_once_with(
                    key, member=repr(member.filename)
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "destination"
            outside = root / "outside"
            destination.mkdir()
            outside.mkdir()
            try:
                (destination / "release").symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("symbolic links are not supported")

            member = zipfile.ZipInfo("release/file.txt")
            with (
                mock.patch.object(
                    updater, "string", wraps=i18n.string
                ) as localized_string,
                self.assertRaises(updater.UpdateError) as raised,
            ):
                updater._validate_zip_member(member, destination)

            self.assertEqual(
                str(raised.exception),
                i18n.string(
                    "cli.update_zip_member_escape", member=repr(member.filename)
                ),
            )
            localized_string.assert_called_once_with(
                "cli.update_zip_member_escape", member=repr(member.filename)
            )

    def test_extract_artifact_preserves_valid_nested_archives(self) -> None:
        """Extract a valid nested release archive and locate its update manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "artifact.zip"
            destination = root / "destination"
            destination.mkdir()
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("release/nested/celune-update.json", b"{}")
                archive.writestr("release/nested/bin/celune.exe", b"binary")

            extracted_root = updater._extract_artifact_root(zip_path, destination)

            self.assertEqual(extracted_root, destination / "release" / "nested")
            self.assertEqual(
                (extracted_root / "bin" / "celune.exe").read_bytes(), b"binary"
            )

    def test_version_helpers_order_tags(self) -> None:
        """Verify version normalization and ordering helpers.

        Raises:
            AssertionError: Version helper behavior changes unexpectedly.
        """
        self.assertEqual(updater.normalize_tag("refs/tags/v4.0.0"), "4.0.0")
        self.assertEqual(updater.short_revision("abcdef123"), "abcdef1")
        self.assertEqual(updater.short_revision(""), "unknown")
        self.assertEqual(updater.is_newer_version_tag("9.9.9", "4.0.0"), True)
        self.assertEqual(updater.is_newer_version_tag("4.0.0", "4.0.0"), False)
        self.assertEqual(updater.is_newer_version_tag("nightly", "4.0.0"), False)
        self.assertEqual(updater.is_newer_version_tag("4.0.0", "4.0.0-rc.1"), True)

    def test_latest_release_ignores_non_semver_releases(self) -> None:
        """Verify only published SemVer releases with no draft flag are considered."""
        release_info = updater.ReleaseInfo(
            tag="v4.5.0",
            version="4.5.0",
            revision="c" * 40,
            asset_url="https://example.com/celune.zip",
        )

        with mock.patch("celune.updater._latest_release", return_value=release_info):
            release = updater._latest_release()
        self.assertIsNotNone(release)
        if release is not None:
            self.assertEqual(release.version, "4.5.0")
            self.assertEqual(release.asset_url, "https://example.com/celune.zip")

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

    def test_check_for_update_builds_update_info_from_release(self) -> None:
        """Verify update metadata comes from a newer SemVer release with an asset."""
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("celune.updater.__version__", "3.5.0"),
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._current_branch", return_value="main"),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch("celune.updater._local_revision", return_value="a" * 40),
            mock.patch("celune.updater._local_tag", return_value="3.5.0"),
            mock.patch(
                "celune.updater._get_latest_release",
                return_value=updater.ReleaseInfo(
                    tag="v4.4.0",
                    version="4.4.0",
                    revision="b" * 40,
                    asset_url="https://example.com/celune.zip",
                ),
            ),
        ):
            update = updater.check_for_update()

        if not updater.FORCE_DISABLE_UPDATES:
            self.assertIsNotNone(update)
            if update is not None:
                self.assertEqual(update.local_revision, "aaaaaaa")
                self.assertEqual(update.latest_revision, "bbbbbbb")
                self.assertEqual(update.latest_version, "4.4.0")

    def test_check_for_update_ignores_release_without_platform_zip(self) -> None:
        """Verify a release without the current-platform ZIP does not prompt updates."""
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._current_branch", return_value="main"),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch("celune.updater._local_revision", return_value="a" * 40),
            mock.patch("celune.updater._local_tag", return_value="3.5.0"),
            mock.patch(
                "celune.updater._get_latest_release",
                return_value=updater.ReleaseInfo(
                    tag="v4.0.0",
                    version="4.0.0",
                    revision="b" * 40,
                    asset_url="",
                ),
            ),
        ):
            self.assertIsNone(updater.check_for_update())

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
                    "celune.exe": updater.sha256_file(bundle_dir / "celune.exe"),
                    "celune-bin.exe": updater.sha256_file(
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
                    "celune.updater._latest_release",
                    return_value=updater.ReleaseInfo(
                        tag="v4.2.0",
                        version="4.2.0",
                        revision="b" * 40,
                        asset_url="https://example.com/celune.zip",
                    ),
                ),
                mock.patch(
                    "celune.updater._read_remote_bundle_manifest", return_value=remote
                ),
                mock.patch("celune.updater._is_git_checkout", return_value=False),
            ):
                update = updater.check_for_update()

        if not updater.FORCE_DISABLE_UPDATES:
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
                "celune.exe": updater.sha256_file(bundle_dir / "celune.exe"),
                "celune-bin.exe": updater.sha256_file(bundle_dir / "celune-bin.exe"),
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
                    "celune.updater._latest_release",
                    return_value=updater.ReleaseInfo(
                        tag="v4.2.0",
                        version="4.2.0",
                        revision="b" * 40,
                        asset_url="https://example.com/celune.zip",
                    ),
                ),
                mock.patch(
                    "celune.updater._read_remote_bundle_manifest", return_value=remote
                ),
                mock.patch("celune.updater._is_git_checkout", return_value=False),
            ):
                self.assertIsNone(updater.check_for_update())

    def test_check_for_update_compiled_ignores_rebuilt_same_release(self) -> None:
        """Verify a rebuilt artifact does not prompt for the same local release."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            (bundle_dir / "celune.exe").write_bytes(b"launcher")
            (bundle_dir / "celune-bin.exe").write_bytes(b"runtime")
            local_files = {
                "celune.exe": updater.sha256_file(bundle_dir / "celune.exe"),
                "celune-bin.exe": updater.sha256_file(bundle_dir / "celune-bin.exe"),
            }
            manifest = {
                "version": "4.3.0",
                "revision": "a" * 40,
                "artifact": "Celune-win-x64",
                "files": local_files,
            }
            (bundle_dir / updater.UPDATE_MANIFEST_NAME).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            remote = updater.BundleManifest(
                version="4.3.0",
                revision="a" * 40,
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
                    "celune.updater._latest_release",
                    return_value=updater.ReleaseInfo(
                        tag="v4.3.0",
                        version="4.3.0",
                        revision="a" * 40,
                        asset_url="https://example.com/celune.zip",
                    ),
                ),
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
