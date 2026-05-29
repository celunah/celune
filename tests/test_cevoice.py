# SPDX-License-Identifier: MIT
"""Tests for CEVOICE parsing, writing, loading, and fallback behavior."""

import copy
import json
import shutil
import tempfile
from typing import cast
from pathlib import Path
from unittest import mock, TestCase

from celune import cevoice
from celune.exceptions import CEVoiceError


class CEVoiceTests(TestCase):
    """Tests for CEVOICE bundle serialization and loader behavior."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for one test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup_temp_dir)
        self.path = self.temp_dir / "sample.cevoice"

    def _cleanup_temp_dir(self) -> None:
        """Remove the temporary fixture directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def tearDown(self) -> None:
        """Reset shared CEVOICE loader globals after one test."""
        loader = cevoice._DEFAULT_LOADER
        if loader is not None:
            loader.close()
        cevoice._DEFAULT_LOADER = None
        cevoice._DEFAULT_LOADER_INITIALIZED = False
        cevoice._DEFAULT_LOADER_ANNOUNCED = False
        cevoice._DEFAULT_LOADER_FAILED = False
        cevoice._SELECTED_BUNDLE = None
        cevoice._SELECTED_BUNDLE_IS_NAMED = False
        cevoice._DEFAULT_LOADER_FELL_BACK_FROM = None

    def _write_bundle(self) -> cevoice.CEVoice:
        """Write and reopen one tiny valid CEVOICE fixture."""
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
                    "faded_accent": "#8866cc",
                },
                "persona": {
                    "identity": {
                        "name": "Fixture",
                        "profile": "A watchful archivist with a dry wit.",
                    },
                    "speaking_style": "Measured, observant, and slightly playful.",
                    "boundaries": [
                        "Do not break character.",
                        "Do not flatten the tone into generic assistant language.",
                    ],
                    "prompt_rules": [
                        "Prefer concrete observations over vague reassurance.",
                    ],
                    "example_dialogue": [
                        "User: i think i fixed it",
                        "Fixture: Sounds like you finally wrestled it into behaving.",
                    ],
                    "style": {
                        "warmth": "high",
                        "directness": "mid",
                        "humor": "mid",
                        "detail": "high",
                        "formality": "mid",
                        "enthusiasm": "low",
                    },
                },
            },
            {
                "balanced": {
                    "cfg_scale": 2.4,
                    "reference_text": "Balanced reference.",
                },
                "bold": {
                    "cfg_scale": 3.0,
                    "reference_text": "Bold reference.",
                },
            },
        )
        return cevoice.CEVoice.open(self.path)

    def test_write_open_read_and_materialize_bundle_assets(self) -> None:
        """Verify CEVOICE writes, reads, ordering, and materialization.

        Raises:
            AssertionError: CEVOICE behavior changes unexpectedly.
        """
        bundle = self._write_bundle()
        self.assertEqual(bundle.metadata["format"], "CECHAR")
        self.assertEqual(bundle.metadata["version"], 2)
        self.assertEqual(bundle.voice_order, ("bold", "balanced"))
        self.assertEqual(bundle.voices["balanced"]["cfg_scale"], 2.4)
        self.assertEqual(
            bundle.voices["balanced"]["reference_text"], "Balanced reference."
        )
        persona = cevoice.persona_metadata_from_manifest(bundle.metadata)
        self.assertIsNotNone(persona)
        assert persona is not None
        self.assertEqual(persona.identity.name, "Fixture")
        self.assertEqual(
            persona.identity.profile, "A watchful archivist with a dry wit."
        )
        self.assertEqual(
            persona.speaking_style, "Measured, observant, and slightly playful."
        )
        self.assertEqual(
            persona.example_dialogue,
            (
                "User: i think i fixed it",
                "Fixture: Sounds like you finally wrestled it into behaving.",
            ),
        )
        self.assertEqual(persona.style.warmth, "high")
        self.assertEqual(
            cevoice.bundle_character_name(bundle),
            "Fixture",
        )
        self.assertEqual(bundle.read_asset("balanced", "wav"), b"wav")
        loader = cevoice.CEVoiceLoader(bundle)
        self.addCleanup(loader.close)
        path = loader.materialize("balanced", "wav")
        self.assertEqual(path.read_bytes(), b"wav")
        self.assertEqual(loader.materialize("balanced", "wav"), path)

        magic, version, _ = cevoice.HEADER.unpack(
            self.path.read_bytes()[: cevoice.HEADER.size]
        )
        self.assertEqual(magic, cevoice.MAGIC)
        self.assertEqual(version, cevoice.VERSION)

    def test_legacy_cevoice_v1_bundle_remains_loadable(self) -> None:
        """Verify legacy CEVOICE v1 bundles still open after the schema rename."""
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["format"] = cevoice.LEGACY_FORMAT_NAME
        metadata["version"] = cevoice.LEGACY_VERSION
        self._rewrite_bundle_header_and_metadata(
            cevoice.LEGACY_MAGIC,
            cevoice.LEGACY_VERSION,
            metadata,
        )

        reopened = cevoice.CEVoice.open(self.path)

        self.assertEqual(reopened.metadata["format"], "CEVOICE")
        self.assertEqual(reopened.metadata["version"], 1)
        self.assertEqual(reopened.voice_order, ("bold", "balanced"))
        self.assertEqual(reopened.read_asset("balanced", "wav"), b"wav")

    def test_asset_lookup_and_checksum_failures_are_reported(self) -> None:
        """Verify missing assets and checksum corruption are reported.

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

        Raises:
            AssertionError: Metadata validation behavior changes unexpectedly.
        """
        bundle = self._write_bundle()
        self.assertEqual(
            cast(dict[str, str], bundle.metadata["theme"])["faded_accent"],
            "#8866cc",
        )
        self.assertEqual(
            cast(dict[str, str], bundle.metadata["persona"])["speaking_style"],
            "Measured, observant, and slightly playful.",
        )

        bundle = self._write_legacy_sleeping_color_bundle()
        self.assertEqual(
            cast(dict[str, str], bundle.metadata["theme"])["faded_accent"],
            "#8866cc",
        )

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
        metadata["persona"] = {"style": {"warmth": 3}}
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "persona style 'warmth'"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        voices["balanced"]["cfg_scale"] = 0
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "cfg_scale"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        voices["balanced"]["reference_text"] = " "
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "reference_text"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        assets = cast(cevoice.Manifest, voices["balanced"]["assets"])
        assets["json"] = {
            "offset": 0,
            "length": 0,
            "sha256": "0" * 64,
        }
        self._rewrite_metadata(metadata)
        with self.assertRaisesRegex(CEVoiceError, "unsupported asset kind"):
            cevoice.CEVoice.open(self.path)

    def _write_legacy_sleeping_color_bundle(self) -> cevoice.CEVoice:
        """Write and reopen one fixture that still uses the legacy sleeping_color key."""
        cevoice.write_cevoice(
            self.path,
            {
                "balanced": {"wav": b"wav", "pt": b"pt"},
            },
            {
                "theme": {
                    "background": "#101010",
                    "accent": "#abcdef",
                    "sleeping_color": "#8866cc",
                },
            },
        )
        return cevoice.CEVoice.open(self.path)

    def test_default_loader_and_announcement_cover_success_and_failure(self) -> None:
        """Verify loader selection and announcement fallback paths.

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
        self.assertEqual(logs, [("Loading voice pack: Fixture", "info")])
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

    def test_missing_named_bundle_falls_back_to_default_bundle(self) -> None:
        """Verify absent named bundles load the built-in default before refs.

        Raises:
            AssertionError: Bundle fallback behavior changes unexpectedly.
        """
        default_path = self.temp_dir / "default.cevoice"
        self._write_bundle()
        shutil.copy(self.path, default_path)
        cevoice.select_voice_bundle("missing")

        with mock.patch(
            "celune.cevoice.default_bundle_path", return_value=default_path
        ):
            loader = cevoice.default_loader()
            logs: list[tuple[str, str]] = []
            self.assertEqual(
                cevoice.announce_default_bundle(
                    lambda msg, severity: logs.append((msg, severity))
                ),
                "Fixture",
            )

        self.assertIsNotNone(loader)
        if loader is not None:
            self.assertEqual(loader.bundle.metadata["name"], "Fixture")
        self.assertEqual(
            logs,
            [
                (
                    "Voice pack missing not found, using default pack instead.",
                    "warning",
                ),
                ("Loading voice pack: Fixture", "info"),
            ],
        )

    def test_missing_selected_and_default_bundles_use_reference_fallback(self) -> None:
        """Verify loose refs are used only after selected and default bundles fail.

        Raises:
            AssertionError: Bundle fallback behavior changes unexpectedly.
        """
        cevoice.select_voice_bundle("missing")
        missing_default = self.temp_dir / "default.cevoice"
        logs: list[tuple[str, str]] = []

        with mock.patch(
            "celune.cevoice.default_bundle_path",
            return_value=missing_default,
        ):
            self.assertIsNone(cevoice.default_loader())
            self.assertEqual(
                cevoice.announce_default_bundle(
                    lambda msg, severity: logs.append((msg, severity))
                ),
                "Celune",
            )

        self.assertEqual(logs[-1][1], "warning")

    def test_materialize_rejects_unsafe_names(self) -> None:
        """Verify unsafe voice and asset names cannot be materialized.

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
        """Replace fixture metadata while preserving the original payload."""
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

    def _rewrite_bundle_header_and_metadata(
        self,
        magic: bytes,
        version: int,
        metadata: dict,
    ) -> None:
        """Replace fixture header and metadata while preserving the original payload."""
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
            cevoice.HEADER.pack(magic, version, len(metadata_bytes))
            + metadata_bytes
            + payload
        )
