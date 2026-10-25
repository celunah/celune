# SPDX-License-Identifier: Apache-2.0
"""Tests for CEVOICE parsing, writing, loading, and fallback behavior."""

import io
import copy
import gzip
import json
import wave
import shutil
import tempfile
from typing import cast
from pathlib import Path
from unittest import mock

import pytest

from celune import cevoice
from celune.cevoice import v4 as cevoice_v4
from celune.cevoice import base as cevoice_base
from celune.exceptions import CEVoiceError

from .support import CeluneTestCase


class TestCEVoice(CeluneTestCase):
    """Tests for CEVOICE bundle serialization and loader behavior."""

    def setUp(self) -> None:
        """Create an isolated temporary directory for one test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup_temp_dir)
        self.path = self.temp_dir / "sample.cevoice"

    def _cleanup_temp_dir(self) -> None:
        """Remove the temporary fixture directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _minimal_wav() -> bytes:
        """Return a small canonical CECHAR reference WAV."""
        buffer = io.BytesIO()
        output = wave.Wave_write(buffer)
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * 240)
        output.close()
        return buffer.getvalue()

    def tearDown(self) -> None:
        """Reset shared CEVOICE loader globals after one test."""
        loader = cevoice_base._DEFAULT_LOADER
        if loader is not None:
            loader.close()
        cevoice_base._DEFAULT_LOADER = None
        cevoice_base._DEFAULT_LOADER_INITIALIZED = False
        cevoice_base._DEFAULT_LOADER_ANNOUNCED = False
        cevoice_base._DEFAULT_LOADER_FAILED = False
        cevoice_base._SELECTED_BUNDLE = None
        cevoice_base._SELECTED_BUNDLE_IS_NAMED = False
        cevoice_base._DEFAULT_LOADER_FELL_BACK_FROM = None

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
                    "persona": {
                        "speaking_style": "More playful and energetic.",
                        "prompt_rules": ["Use a brighter conversational rhythm."],
                        "style": {"enthusiasm": "high"},
                    },
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
        assert bundle.metadata["format"] == "CECHAR"
        assert bundle.metadata["version"] == 3
        assert bundle.voice_order == ("bold", "balanced")
        assert bundle.voices["balanced"]["cfg_scale"] == 2.4
        assert bundle.voices["balanced"]["reference_text"] == "Balanced reference."
        persona = cevoice.persona_metadata_from_manifest(bundle.metadata)
        assert persona is not None
        assert persona.identity.name == "Fixture"
        assert persona.identity.profile == "A watchful archivist with a dry wit."
        assert persona.speaking_style == "Measured, observant, and slightly playful."
        assert persona.example_dialogue == (
            "User: i think i fixed it",
            "Fixture: Sounds like you finally wrestled it into behaving.",
        )
        assert cevoice.persona_files_from_manifest(bundle.metadata) == {}
        assert cevoice.persona_files_from_bundle(bundle) == {}
        assert persona.style.warmth == "high"
        voice_persona = cevoice.persona_metadata_from_voice(bundle, "bold")
        assert voice_persona is not None
        assert voice_persona.speaking_style == "More playful and energetic."
        assert voice_persona.style.enthusiasm == "high"
        assert cevoice.bundle_character_name(bundle) == "Fixture"
        assert bundle.read_asset("balanced", "wav") == b"wav"
        loader = cevoice.CEVoiceLoader(bundle)
        self.addCleanup(loader.close)
        path = loader.materialize("balanced", "wav")
        assert path.read_bytes() == b"wav"
        assert loader.materialize("balanced", "wav") == path
        with mock.patch.object(
            cevoice_base,
            "bundle_matches_default_pack_checksum",
            return_value=True,
        ):
            assert cevoice.bundle_display_name(bundle) == "Fixture (sample)"

        magic, version, _ = cevoice.HEADER.unpack(
            self.path.read_bytes()[: cevoice.HEADER.size]
        )
        assert magic == cevoice.MAGIC
        assert version == cevoice.VERSION

    def test_materialize_recreates_missing_cached_asset_after_cleanup(self) -> None:
        """Verify cached CEVOICE temp assets are rebuilt if cleanup removed them."""
        bundle = self._write_bundle()
        loader = cevoice.CEVoiceLoader(bundle)
        self.addCleanup(loader.close)

        first_path = loader.materialize("balanced", "wav")
        assert first_path.read_bytes() == b"wav"

        shutil.rmtree(first_path.parent)
        rebuilt_path = loader.materialize("balanced", "wav")

        assert rebuilt_path == first_path
        assert rebuilt_path.is_file()
        assert rebuilt_path.read_bytes() == b"wav"

    def test_cechar_v2_bundle_remains_loadable(self) -> None:
        """Verify CECHAR v2 bundles still open after the CECHAR v3 upgrade."""
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["version"] = 2
        self._rewrite_bundle_header_and_metadata(
            cevoice.MAGIC,
            2,
            metadata,
        )

        reopened = cevoice.CEVoice.open(self.path)

        assert reopened.metadata["format"] == "CECHAR"
        assert reopened.metadata["version"] == 2
        assert reopened.voice_order == ("bold", "balanced")
        assert reopened.read_asset("balanced", "wav") == b"wav"

    def test_cechar_header_and_manifest_versions_must_match(self) -> None:
        """Reject a CECHAR v2 header carrying a CECHAR v3 manifest."""
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["version"] = 3
        self._rewrite_bundle_header_and_metadata(cevoice.MAGIC, 2, metadata)

        with pytest.raises(CEVoiceError, match="format/version mismatch"):
            cevoice.CEVoice.open(self.path)

    def test_v4_decompression_rejects_payloads_over_the_logical_limit(self) -> None:
        """Verify compressed CECHAR v4 payloads are bounded before parsing."""
        with (
            mock.patch.object(cevoice_v4, "V4_MAX_DECOMPRESSED_BYTES", 3),
            self.assertRaisesRegex(CEVoiceError, "too large"),
        ):
            cevoice_v4._decompress_v4_payload(
                gzip.compress(b"four"),
                cevoice.V4_COMPRESSION_GZIP,
            )

    def test_cechar_v5_reserves_singing_voice_outside_tts_voice_order(self) -> None:
        """Verify CECHAR v5 keeps its singing voice separate from TTS voices."""
        sample = self._minimal_wav()
        cevoice.write_cechar_v5(
            self.path,
            {"balanced": {"wav": sample}, "sing": {"wav": sample}},
            {"name": "Fixture", "default_voice": "balanced"},
            {
                "balanced": {"reference_text": "Balanced reference."},
                "sing": {
                    "reference_text": "Singing reference.",
                    "cesn": "[lang:en][f#3:4,g3:4][ˈsɪŋ.ɪŋ]Singing.",
                    "range": {"min": 54, "max": 72},
                },
            },
        )

        bundle = cevoice.CEVoice.open(self.path)

        assert bundle.metadata["version"] == cevoice.V5_VERSION
        assert bundle.voice_order == ("balanced",)
        assert bundle.singing_voice == cevoice.SINGING_VOICE_NAME
        assert bundle.voices["balanced"]["cesn"] is None
        assert bundle.voices["sing"]["cfg_scale"] is None
        assert bundle.voices["sing"]["persona"] is None
        assert bundle.voices["sing"]["cesn"] == (
            "[lang:en][f#3:4,g3:4][ˈsɪŋ.ɪŋ]Singing."
        )
        assert bundle.singing_pitch_range == (54, 72)

    def test_default_pack_uses_v5_and_keeps_classic_on_v4(self) -> None:
        """Keep the default pack's singing reference scoped to CECHAR v5."""
        default = cevoice.CEVoice.open(Path("voices/default.cevoice"))
        classic = cevoice.CEVoice.open(Path("voices/classic.cevoice"))

        assert default.metadata["version"] == cevoice.V5_VERSION
        assert default.voice_order == ("balanced", "calm", "bold", "upbeat")
        assert default.singing_voice == cevoice.SINGING_VOICE_NAME
        assert default.singing_pitch_range == (54, 72)
        assert (
            default.voices["sing"]["reference_text"]
            == "Moonlight resting, silver-blue, I'll keep one quiet light for you."
        )
        assert default.voices["sing"]["cesn"] == (
            "[bpm:108,key:g#,scale:natural_minor][lang:en]"
            "[a3:4,b3:4d][ˈmuːn.laɪt] Moonlight "
            "[c#4:4,b3:4d][ˈrɛs.tɪŋ] resting "
            "[g#3:4,a#3:8,a#3:4][ˈsɪl.vɚ.bluː] silver-blue. "
            "[tuplet:3,2][r:1,g#3:4][ˈaɪl] I'll "
            "[a#3:4][ˈkiːp] keep [b3:4][ˈwʌn] one "
            "[tuplet:3,2][c#4:4,c4:4][ˈkwaɪ.ət] quiet "
            "[g#3:4][ˈlaɪt] light [r:2,g#3:4][fɚ] for "
            "[f#3:4][ˈjuː] you."
        )
        assert all(
            default.voices[voice]["cesn"] is None for voice in default.voice_order
        )
        assert cevoice.bundle_matches_default_pack_checksum(default.path)
        assert classic.metadata["version"] == cevoice.V4_VERSION

    def test_voice_display_names_do_not_change_backend_identifiers(self) -> None:
        """Keep lowercase selection keys separate from title-case UI labels."""
        cevoice.write_cechar_v5(
            self.path,
            {
                "balanced": {"wav": self._minimal_wav()},
                "spirited": {"wav": self._minimal_wav()},
            },
            {"name": "Nol", "default_voice": "balanced"},
            {
                "balanced": {
                    "reference_text": "Hello.",
                    "display_name": "Balanced",
                },
                "spirited": {"reference_text": "Ha!"},
            },
        )

        bundle = cevoice.CEVoice.open(self.path)

        assert bundle.voice_order == ("balanced", "spirited")
        assert bundle.voice_display_name("balanced") == "Balanced"
        assert bundle.voice_display_name("spirited") == "Spirited"

    def test_repository_packs_are_registered_as_official_defaults(self) -> None:
        """Keep every repository-bundled character pack in the default registry."""
        assert set(cevoice.KNOWN_DEFAULT_CEVOICE_PACKS) == {
            "default.cevoice",
            "default_sep19.cevoice",
            "classic.cevoice",
            "nol.cevoice",
        }
        for filename, checksums in cevoice.KNOWN_DEFAULT_CEVOICE_PACKS.items():
            bundle = cevoice.CEVoice.open(Path("voices") / filename)
            assert cevoice.bundle_sha256(bundle.path) in checksums
            assert cevoice.bundle_matches_default_pack_checksum(bundle.path)
            if filename == "default_sep19.cevoice":
                assert bundle.metadata["version"] == cevoice.VERSION
            assert cevoice.bundle_display_name(bundle) == (
                f"{cevoice.bundle_character_name(bundle)} ({Path(filename).stem})"
            )

    def test_cechar_v5_requires_singing_cesn_and_null_tts_cesn(self) -> None:
        """Require a score for the singing role and null for speaking roles."""
        sample = self._minimal_wav()
        with pytest.raises(CEVoiceError, match="singing voice cesn is required"):
            cevoice.write_cechar_v5(
                self.path,
                {"balanced": {"wav": sample}, "sing": {"wav": sample}},
                voice_metadata={"sing": {"reference_text": "Singing reference."}},
            )

        with pytest.raises(CEVoiceError, match="voice 'balanced' cesn must be null"):
            cevoice.write_cechar_v5(
                self.path,
                {"balanced": {"wav": sample}, "sing": {"wav": sample}},
                voice_metadata={
                    "balanced": {"cesn": "[lang:en][c3:4,d3:4][ˈspiː.kɪŋ]Speaking."},
                    "sing": {
                        "reference_text": "Singing reference.",
                        "cesn": "[lang:en][c3:4,d3:4][ˈsɪŋ.ɪŋ]Singing.",
                        "range": {"min": 54, "max": 72},
                    },
                },
            )

    def test_cechar_v5_rejects_singing_voice_in_public_voice_order(self) -> None:
        """Verify CECHAR v5 does not let TTS select the singing voice."""
        sample = self._minimal_wav()
        with pytest.raises(CEVoiceError, match="must not include the singing voice"):
            cevoice.write_cechar_v5(
                self.path,
                {"balanced": {"wav": sample}, "sing": {"wav": sample}},
                {"voice_order": ["balanced", "sing"]},
                {
                    "balanced": {"reference_text": "Balanced reference."},
                    "sing": {
                        "reference_text": "Singing reference.",
                        "cesn": "[lang:en][f#3:4,g3:4][ˈsɪŋ.ɪŋ]Singing.",
                        "range": {"min": 54, "max": 72},
                    },
                },
            )

    def test_cechar_v5_requires_a_valid_singing_range(self) -> None:
        """Require every newly written v5 singing entry to declare valid bounds."""
        sample = self._minimal_wav()
        voices = {"balanced": {"wav": sample}, "sing": {"wav": sample}}
        metadata: dict[str, cevoice.Manifest] = {
            "balanced": {"reference_text": "Balanced reference."},
            "sing": {
                "reference_text": "Singing reference.",
                "cesn": "[lang:en][f#3:4,g3:4][ˈsɪŋ.ɪŋ]Singing.",
            },
        }
        with pytest.raises(CEVoiceError, match="singing voice range is required"):
            cevoice.write_cechar_v5(self.path, voices, voice_metadata=metadata)

        metadata["sing"]["range"] = {"min": 73, "max": 72}
        with pytest.raises(CEVoiceError, match="ordered MIDI pitches"):
            cevoice.write_cechar_v5(self.path, voices, voice_metadata=metadata)

    def test_cechar_v4_treats_sing_as_an_ordinary_voice(self) -> None:
        """Keep the reserved singing role scoped to CECHAR v5 and newer."""
        sample = self._minimal_wav()
        cevoice.write_cechar_v4(
            self.path,
            {"balanced": {"wav": sample}, "sing": {"wav": sample}},
            {"voice_order": ["balanced", "sing"]},
            {
                "balanced": {"reference_text": "Balanced reference."},
                "sing": {"reference_text": "Legacy speaking reference."},
            },
        )

        bundle = cevoice.CEVoice.open(self.path)

        assert bundle.voice_order == ("balanced", "sing")
        assert bundle.singing_voice is None
        assert bundle.singing_pitch_range is None

    # any types here are resolved dynamically
    def test_asset_checksums_are_case_insensitive(self) -> None:
        """Accept uppercase hexadecimal asset checksums in a valid manifest."""
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        voice_data = voices["balanced"]
        assets = cast(cevoice.Manifest, voice_data["assets"])
        wav_asset = cast(cevoice.Manifest, assets["wav"])
        wav_asset["sha256"] = cast(str, wav_asset["sha256"]).upper()
        self._rewrite_metadata(metadata)

        reopened = cevoice.CEVoice.open(self.path)

        assert reopened.read_asset("balanced", "wav") == b"wav"

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

        assert reopened.metadata["format"] == "CEVOICE"
        assert reopened.metadata["version"] == 1
        assert reopened.voice_order == ("bold", "balanced")
        assert reopened.read_asset("balanced", "wav") == b"wav"

    def test_asset_lookup_and_checksum_failures_are_reported(self) -> None:
        """Verify missing assets and checksum corruption are reported.

        Raises:
            AssertionError: CEVOICE failure handling changes unexpectedly.
        """
        bundle = self._write_bundle()
        with pytest.raises(KeyError, match="asset 'pt'"):
            bundle.asset("bold", "pt")

        raw = self.path.read_bytes()
        self.path.write_bytes(raw[:-1] + b"x")
        broken = cevoice.CEVoice.open(self.path)
        with pytest.raises(CEVoiceError, match="checksum mismatch"):
            broken.read_asset("bold", "wav")

    def test_invalid_metadata_is_rejected(self) -> None:
        """Verify malformed CEVOICE metadata is rejected.

        Raises:
            AssertionError: Metadata validation behavior changes unexpectedly.
        """
        bundle = self._write_bundle()
        assert (
            cast(dict[str, str], bundle.metadata["theme"])["faded_accent"] == "#8866cc"
        )
        assert (
            cast(dict[str, str], bundle.metadata["persona"])["speaking_style"]
            == "Measured, observant, and slightly playful."
        )

        bundle = self._write_legacy_sleeping_color_bundle()
        assert (
            cast(dict[str, str], bundle.metadata["theme"])["faded_accent"] == "#8866cc"
        )

    def test_persona_files_from_manifest_only_returns_supported_markdown(self) -> None:
        """Verify CECHAR v3 persona files use the supported filename whitelist."""
        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["persona_files"] = {
            "identity.md": "  Identity text.  ",
            "soul.md": "Soul text.",
            "notes.md": "Ignore this.",
        }

        files = cevoice.persona_files_from_manifest(metadata)

        assert files == {
            "identity.md": "Identity text.",
            "soul.md": "Soul text.",
        }

    # any types here are resolved dynamically
    def test_bundle_assets_can_store_supported_persona_markdown(self) -> None:
        """Verify CECHAR v3 persona Markdown is stored as top-level payload assets."""
        cevoice.write_cevoice(
            self.path,
            {"balanced": {"wav": b"wav", "pt": b"pt"}},
            {"name": "Fixture"},
            {"balanced": {"reference_text": "Balanced reference."}},
            bundle_assets={
                "identity.md": b"Name: Fixture\n\nArchivist.",
                "speech_style.md": b"Measured and steady.",
            },
        )

        bundle = cevoice.CEVoice.open(self.path)

        assert bundle.read_bundle_asset("identity.md") == b"Name: Fixture\n\nArchivist."
        assert cevoice.persona_files_from_bundle(bundle) == {
            "identity.md": "Name: Fixture\n\nArchivist.",
            "speech_style.md": "Measured and steady.",
        }

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["default_voice"] = "missing"
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="default_voice"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["voice_order"] = ["bold", "bold"]
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="duplicates"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["theme"] = {"background": "#101010", "accent": "blue"}
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="hex color"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        metadata["persona"] = {"style": {"warmth": 3}}
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="persona style 'warmth'"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        voices["balanced"]["cfg_scale"] = 0
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="cfg_scale"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        voices["balanced"]["reference_text"] = " "
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="reference_text"):
            cevoice.CEVoice.open(self.path)

        bundle = self._write_bundle()
        metadata = copy.deepcopy(bundle.metadata)
        voices = cast(cevoice.VoiceManifest, metadata["voices"])
        del voices["balanced"]["reference_text"]
        self._rewrite_metadata(metadata)
        with pytest.raises(CEVoiceError, match="reference_text is required"):
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
        with pytest.raises(CEVoiceError, match="unsupported asset kind"):
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
            {
                "balanced": {
                    "reference_text": "Balanced reference.",
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
        assert loader is not None

        def log(msg: str, severity: str) -> None:
            logs.append((msg, severity))

        assert cevoice.announce_default_bundle(log) == "Fixture"
        assert logs == [("Loading voice pack: Fixture", "info")]
        assert cevoice.announce_default_bundle(log) is None

        cevoice.select_voice_bundle(self.temp_dir / "missing.cevoice")
        assert cevoice.default_loader() is None
        assert cevoice.announce_default_bundle(log) is None

        invalid_path = self.temp_dir / "invalid.cevoice"
        invalid_path.write_bytes(b"bad")
        cevoice.select_voice_bundle(invalid_path)
        assert cevoice.default_loader() is None
        assert cevoice.announce_default_bundle(log) is None
        assert logs[-1][1] == "warning"

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
            "celune.cevoice.base.default_bundle_path", return_value=default_path
        ):
            loader = cevoice.default_loader()
            logs: list[tuple[str, str]] = []
            assert (
                cevoice.announce_default_bundle(
                    lambda msg, severity: logs.append((msg, severity))
                )
                == "Fixture"
            )

        assert loader is not None
        if loader is not None:
            assert loader.bundle.metadata["name"] == "Fixture"
        assert logs == [
            (
                "Voice pack missing not found, using default pack instead.",
                "warning",
            ),
            ("Loading voice pack: Fixture", "info"),
        ]

    def test_named_bundle_resolution_uses_top_level_voices_directory(self) -> None:
        """Verify bare bundle names resolve from the user-local voices directory.

        Raises:
            AssertionError: Bundle path resolution changes unexpectedly.
        """
        expected = self.temp_dir / "voices" / "fixture.cevoice"
        with mock.patch(
            "celune.cevoice.base.voices_data_dir", return_value=expected.parent
        ):
            assert cevoice.resolve_bundle_path("fixture") == expected
            assert cevoice.resolve_bundle_path("fixture.cevoice") == expected

    def test_bundled_voices_are_copied_to_user_data_on_first_use(self) -> None:
        """Verify repository voice packs are copied to user data once."""
        repository_root = self.temp_dir / "repository-root"
        repository_dir = repository_root / "voices"
        user_dir = self.temp_dir / "user-voices"
        repository_dir.mkdir(parents=True)
        (repository_dir / "default.cevoice").write_bytes(b"default")

        with (
            mock.patch(
                "celune.cevoice.base.project_root",
                return_value=repository_root,
            ),
            mock.patch(
                "celune.cevoice.base.voices_data_dir",
                return_value=user_dir,
            ),
        ):
            assert cevoice.bundled_voices_dir() == user_dir

        assert (user_dir / "default.cevoice").read_bytes() == b"default"

    def test_stale_user_default_pack_is_refreshed_from_repository(self) -> None:
        """Verify the managed default pack follows the bundled repository pack."""
        repository_root = self.temp_dir / "repository-root"
        repository_dir = repository_root / "voices"
        user_dir = self.temp_dir / "user-voices"
        repository_dir.mkdir(parents=True)
        user_dir.mkdir(parents=True)
        (repository_dir / "default.cevoice").write_bytes(b"new-default")
        (user_dir / "default.cevoice").write_bytes(b"old-default")

        with (
            mock.patch(
                "celune.cevoice.base.project_root",
                return_value=repository_root,
            ),
            mock.patch(
                "celune.cevoice.base.voices_data_dir",
                return_value=user_dir,
            ),
        ):
            self.assertEqual(cevoice.bundled_voices_dir(), user_dir)

        self.assertEqual(
            (user_dir / "default.cevoice").read_bytes(),
            b"new-default",
        )

    def test_all_repository_cevoice_packs_are_synced_to_user_data(self) -> None:
        """Verify repository-provided official packs are available to the loader."""
        repository_root = self.temp_dir / "repository-root"
        repository_dir = repository_root / "voices"
        user_dir = self.temp_dir / "user-voices"
        repository_dir.mkdir(parents=True)
        (repository_dir / "default.cevoice").write_bytes(b"default")
        (repository_dir / "classic.cevoice").write_bytes(b"classic")

        with (
            mock.patch(
                "celune.cevoice.base.project_root",
                return_value=repository_root,
            ),
            mock.patch(
                "celune.cevoice.base.voices_data_dir",
                return_value=user_dir,
            ),
        ):
            assert cevoice.bundled_voices_dir() == user_dir

        assert (user_dir / "default.cevoice").read_bytes() == b"default"
        assert (user_dir / "classic.cevoice").read_bytes() == b"classic"

    def test_missing_selected_and_default_bundles_report_no_compatible_pack(
        self,
    ) -> None:
        """Verify missing selected and default bundles report no compatible pack.

        Raises:
            AssertionError: Bundle fallback behavior changes unexpectedly.
        """
        cevoice.select_voice_bundle("missing")
        missing_default = self.temp_dir / "default.cevoice"
        logs: list[tuple[str, str]] = []

        with mock.patch(
            "celune.cevoice.base.default_bundle_path",
            return_value=missing_default,
        ):
            assert cevoice.default_loader() is None
            assert (
                cevoice.announce_default_bundle(
                    lambda msg, severity: logs.append((msg, severity))
                )
                is None
            )

        assert logs[-1][1] == "warning"

    def test_materialize_rejects_unsafe_names(self) -> None:
        """Verify unsafe voice and asset names cannot be materialized.

        Raises:
            AssertionError: Path safety behavior changes unexpectedly.
        """
        loader = cevoice.CEVoiceLoader(self._write_bundle())
        self.addCleanup(loader.close)
        with pytest.raises(CEVoiceError, match="invalid voice name"):
            loader.materialize("../bad", "wav")
        with pytest.raises(CEVoiceError, match="invalid asset kind"):
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
