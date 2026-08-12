# SPDX-License-Identifier: Apache-2.0
"""Tests for lightweight data, localization, and update helpers."""

import datetime
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from celune import i18n, namedays, updater

from .support import CeluneTestCase


class TestNameDay(CeluneTestCase):
    """Tests for name-day lookup helpers."""

    def test_lookup_helpers_cover_supported_inputs(self) -> None:
        """Verify date lookup helpers and invalid input handling.

        Raises:
            AssertionError: Name-day lookup behavior changes unexpectedly.
        """
        assert namedays.get_names(5, 16) == ["Andrew", "Simon"]
        assert namedays.get_names_for_date(datetime.date(2026, 5, 16)) == [
            "Andrew",
            "Simon",
        ]
        assert namedays.get_names_for_date("2026-05-16") == ["Andrew", "Simon"]
        assert namedays.get_names_for_date("05-16") == ["Andrew", "Simon"]
        assert namedays.has_name_day("andrew", "05-16")
        assert "10-21" in namedays.find_dates_for_name("Celine")
        with pytest.raises(TypeError):
            namedays.get_names_for_date(123)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            namedays.get_names_for_date("bad-date")


class TestI18n(CeluneTestCase):
    """Tests for localization fallback behavior."""

    def tearDown(self) -> None:
        """Restore the process locale after each localization test."""
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
            assert i18n.string("hello", name="Celune") == "Hello Celune"
            assert i18n.string("missing") == "missing"
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
            assert i18n.string("hello") == "Hello"
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
                assert i18n.get_system_locale() == "pl"
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
                assert i18n.get_system_locale() == "en"
            stderr_write.assert_not_called()
        finally:
            i18n.STRINGS.clear()
            i18n.STRINGS.update(original)


class TestUpdater(CeluneTestCase):
    """Tests for pure updater decision logic."""

    def test_version_helpers_order_tags(self) -> None:
        """Verify version normalization and ordering helpers.

        Raises:
            AssertionError: Version helper behavior changes unexpectedly.
        """
        assert updater.normalize_tag("refs/tags/v4.0.0") == "4.0.0"
        assert updater.short_revision("abcdef123") == "abcdef1"
        assert updater.short_revision("") == "unknown"
        assert updater.is_newer_version_tag("9.9.9", "4.0.0")
        assert not updater.is_newer_version_tag("4.0.0", "4.0.0")
        assert not updater.is_newer_version_tag("nightly", "4.0.0")
        assert updater.is_newer_version_tag("4.0.0", "4.0.0-rc.1")

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
        assert release is not None
        if release is not None:
            assert release.version == "4.5.0"
            assert release.asset_url == "https://example.com/celune.zip"

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
            assert updater.check_for_update() is None

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
            assert update is not None
            if update is not None:
                assert update.local_revision == "aaaaaaa"
                assert update.latest_revision == "bbbbbbb"
                assert update.latest_version == "4.4.0"

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
            assert updater.check_for_update() is None

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
            assert update is not None
            if update is not None:
                assert update.local_revision == "aaaaaaa"
                assert update.latest_revision == "bbbbbbb"
                assert update.latest_version == "4.2.0"

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
                assert updater.check_for_update() is None

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
                assert updater.check_for_update() is None

    def test_update_to_latest_rejects_unsafe_states(self) -> None:
        """Verify unsafe repository states reject automatic updates.

        Raises:
            AssertionError: Update safety behavior changes unexpectedly.
        """
        with (
            mock.patch("celune.updater._is_git_checkout", return_value=False),
            pytest.raises(updater.UpdateError, match="did not find"),
        ):
            updater.update_to_latest()

        with (
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._has_local_changes", return_value=True),
            pytest.raises(updater.UpdateError, match="not committed"),
        ):
            updater.update_to_latest()

        with (
            mock.patch("celune.updater._is_git_checkout", return_value=True),
            mock.patch("celune.updater._has_local_changes", return_value=False),
            mock.patch(
                "celune.updater._current_branch",
                side_effect=subprocess.TimeoutExpired("git", 5),
            ),
            pytest.raises(updater.UpdateError, match="timed out"),
        ):
            updater.update_to_latest()
