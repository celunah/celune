# SPDX-License-Identifier: MIT
"""Tests for CEVOICE parsing, writing, loading, and fallback behavior."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from celune import cevoice
from celune.exceptions import CEVoiceError


class CEVoiceTests(unittest.TestCase):
    """Tests for CEVOICE bundle serialization and loader behavior."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for one test.

        Returns:
            None: This fixture prepares per-test paths.

        Raises:
            None: Temporary directory creation errors propagate normally.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup_temp_dir)
        self.path = self.temp_dir / "sample.cevoice"

    def _cleanup_temp_dir(self) -> None:
        """Remove the temporary fixture directory.

        Returns:
            None: This fixture removes temporary files created by the test.

        Raises:
            None: Cleanup ignores missing paths and other removal errors.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def tearDown(self) -> None:
        """Reset shared CEVOICE loader globals after one test.

        Returns:
            None: This fixture clears process-wide CEVOICE state.

        Raises:
            None: Loader cleanup errors are not expected.
        """
        loader = cevoice._DEFAULT_LOADER
        if loader is not None:
            loader.close()
        cevoice._DEFAULT_LOADER = None
        cevoice._DEFAULT_LOADER_INITIALIZED = False
        cevoice._DEFAULT_LOADER_ANNOUNCED = False
        cevoice._DEFAULT_LOADER_FAILED = False
        cevoice._SELECTED_BUNDLE = None

    def _write_bundle(self) -> cevoice.CEVoice:
        """Write and reopen one tiny valid CEVOICE fixture.

        Returns:
            CEVoice: The reopened CEVOICE fixture bundle.

        Raises:
            CEVoiceError: Fixture writing or loading fails unexpectedly.
        """
        cevoice.write_cevoice(
            self.path,
            {
                "balanced": {"wav": b"wav", "pt": b"pt"},
                "bold": {"wav": b"bold"},
            },
            {
                "name": "Fixture",
                "default_voice": "bold",
                "voice_order": ["bold"],
                "theme": {
                    "background": "#101010",
                    "accent": "#abcdef",
                    "glow_color": "#fedcba",
                },
            },
        )
        return cevoice.CEVoice.open(self.path)

    def test_write_open_read_and_materialize_bundle_assets(self) -> None:
        """Verify CEVOICE writes, reads, ordering, and materialization.

        Returns:
            None: Assertions verify CEVOICE success behavior.

        Raises:
            AssertionError: CEVOICE behavior changes unexpectedly.
        """
        bundle = self._write_bundle()
        self.assertEqual(bundle.voice_order, ("bold", "balanced"))
        self.assertEqual(bundle.read_asset("balanced", "wav"), b"wav")
        loader = cevoice.CEVoiceLoader(bundle)
        self.addCleanup(loader.close)
        path = loader.materialize("balanced", "wav")
        self.assertEqual(path.read_bytes(), b"wav")
        self.assertEqual(loader.materialize("balanced", "wav"), path)

    def test_asset_lookup_and_checksum_failures_are_reported(self) -> None:
        """Verify missing assets and checksum corruption are reported.

        Returns:
            None: Assertions verify CEVOICE failure behavior.

        Raises:
            AssertionError: CEVOICE failure handling changes unexpectedly.
        """
        bundle = self._write_bundle()
        with self.assertRaisesRegex(KeyError, "asset 'pt'"):
            bundle.asset("bold", "pt")

        raw = self.path.read_bytes()
        self.path.write_bytes(raw[:-1] + b"x")
        broken = cevoice.CEVoice.open(self.path)
        with self.assertRaisesRegex(CEVoiceError, "checksum mismatch"):
            broken.read_asset("bold", "wav")

    def test_invalid_metadata_is_rejected(self) -> None:
        """Verify malformed CEVOICE metadata is rejected.

        Returns:
            None: Assertions verify metadata validation behavior.

        Raises:
            AssertionError: Metadata validation behavior changes unexpectedly.
        """
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["default_voice"] = "missing"
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "default_voice"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["voice_order"] = ["bold", "bold"]
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "duplicates"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["theme"] = {"background": "#101010", "accent": "blue"}
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "hex color"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["voices"]["balanced"]["assets"]["json"] = {
            "offset": 0,
            "length": 0,
            "sha256": "0" * 64,
        }
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "unsupported asset kind"):
            cevoice.CEVoice.open(self.path)

    def test_default_loader_and_announcement_cover_success_and_failure(self) -> None:
        """Verify loader selection and announcement fallback paths.

        Returns:
            None: Assertions verify loader behavior.

        Raises:
            AssertionError: Loader behavior changes unexpectedly.
        """
        self._write_bundle()
        cevoice.select_voice_bundle(self.path)
        logs: list[tuple[str, str]] = []
        loader = cevoice.default_loader()
        self.assertIsNotNone(loader)

        def log(msg: str, severity: str) -> None:
            logs.append((msg, severity))

        self.assertEqual(cevoice.announce_default_bundle(log), "Fixture")
        self.assertEqual(logs, [("Loading voice bundle: Fixture", "info")])
        self.assertIsNone(cevoice.announce_default_bundle(log))

        cevoice.select_voice_bundle(self.temp_dir / "missing.cevoice")
        self.assertIsNone(cevoice.default_loader())
        self.assertIsNone(cevoice.announce_default_bundle(log))

        invalid_path = self.temp_dir / "invalid.cevoice"
        invalid_path.write_bytes(b"bad")
        cevoice.select_voice_bundle(invalid_path)
        self.assertIsNone(cevoice.default_loader())
        self.assertEqual(cevoice.announce_default_bundle(log), "Celune")
        self.assertEqual(logs[-1][1], "warning")

    def test_materialize_rejects_unsafe_names(self) -> None:
        """Verify unsafe voice and asset names cannot be materialized.

        Returns:
            None: Assertions verify path safety behavior.

        Raises:
            AssertionError: Path safety behavior changes unexpectedly.
        """
        loader = cevoice.CEVoiceLoader(self._write_bundle())
        self.addCleanup(loader.close)
        with self.assertRaisesRegex(CEVoiceError, "invalid voice name"):
            loader.materialize("../bad", "wav")
        with self.assertRaisesRegex(CEVoiceError, "invalid asset kind"):
            loader.materialize("balanced", "../wav")

    def _rewrite_metadata(self, metadata: dict) -> None:
        """Replace fixture metadata while preserving the original payload.

        Args:
            metadata: Metadata object to serialize into the fixture bundle.

        Returns:
            None: This helper rewrites the current fixture file.

        Raises:
            None: Serialization or file errors propagate normally.
        """
        metadata_bytes = json.dumps(
            metadata,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        current = self.path.read_bytes()
        _, _, metadata_length = cevoice.HEADER.unpack(current[: cevoice.HEADER.size])
        payload_offset = cevoice.HEADER.size + metadata_length
        payload = current[payload_offset:]
        self.path.write_bytes(
            cevoice.HEADER.pack(cevoice.MAGIC, cevoice.VERSION, len(metadata_bytes))
            + metadata_bytes
            + payload
        )
