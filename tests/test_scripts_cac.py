# SPDX-License-Identifier: Apache-2.0
"""Tests for the CEVOICE creation helper script."""

from __future__ import annotations

import io
import sys
import contextlib
import importlib.util
from types import ModuleType
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import soundfile as sf

from celune.cevoice import CEVoice, write_cechar_v4, write_cechar_v5

from .support import CeluneTestCase


def _load_cac_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "cac.py"
    spec = importlib.util.spec_from_file_location("scripts.cac", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/cac.py")
    module = importlib.util.module_from_spec(spec)
    fake_numpy = ModuleType("numpy")
    setattr(fake_numpy, "mean", mock.Mock())  # noqa: B010
    setattr(fake_numpy, "asarray", mock.Mock())  # noqa: B010
    fake_celune = ModuleType("celune")
    fake_cevoice = ModuleType("celune.cevoice")
    fake_soundfile = ModuleType("soundfile")
    setattr(fake_soundfile, "read", mock.Mock())  # noqa: B010
    setattr(fake_soundfile, "info", mock.Mock())  # noqa: B010
    setattr(fake_soundfile, "write", mock.Mock())  # noqa: B010
    fake_scipy = ModuleType("scipy")
    fake_scipy_signal = ModuleType("scipy.signal")
    setattr(fake_scipy_signal, "resample_poly", mock.Mock())  # noqa: B010
    setattr(fake_cevoice, "write_cevoice", mock.Mock())  # noqa: B010
    setattr(fake_cevoice, "write_cechar_v4", mock.Mock())  # noqa: B010
    setattr(fake_cevoice, "write_cechar_v5", mock.Mock())  # noqa: B010
    setattr(fake_cevoice, "SINGING_VOICE_NAME", "sing")  # noqa: B010

    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_numpy,
            "celune": fake_celune,
            "celune.cevoice": fake_cevoice,
            "soundfile": fake_soundfile,
            "scipy": fake_scipy,
            "scipy.signal": fake_scipy_signal,
        },
    ):
        spec.loader.exec_module(module)
    return module


cac = _load_cac_module()


class TestCACScript(CeluneTestCase):
    """Verify the CEVOICE helper script supports simple and wizard modes."""

    def test_simple_mode_prompts_for_reference_text_when_not_provided(self) -> None:
        """Verify simple mode asks for the reference transcript when it was omitted."""
        with TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "voice.wav"
            wav_path.write_bytes(b"RIFFfakeWAVE")

            with (
                mock.patch.object(
                    cac, "write_cevoice", return_value=Path("Nova.cevoice")
                ) as write_cevoice,
                mock.patch.object(
                    cac, "normalize_reference_wav_asset", return_value=b"normalized-wav"
                ) as normalize_reference_wav_asset,
                mock.patch.object(
                    cac, "ask_required_text", return_value="Hello from Nova."
                ) as ask_required_text,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = cac.main(["Nova", str(wav_path)])

        assert exit_code == 0
        ask_required_text.assert_called_once_with(
            "Enter reference transcript for the WAV file",
            "A reference transcript is required.",
        )
        normalize_reference_wav_asset.assert_called_once_with(wav_path)
        write_cevoice.assert_called_once_with(
            Path("Nova.cevoice"),
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"wav": b"normalized-wav"}},
            {
                "name": "Nova",
                "default_voice": cac.DEFAULT_SIMPLE_VOICE_NAME,
                "voice_order": [cac.DEFAULT_SIMPLE_VOICE_NAME],
            },
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"reference_text": "Hello from Nova."}},
        )
        assert "Saved voice pack to Nova.cevoice" in stdout.getvalue()

    def test_simple_mode_accepts_reference_text_argument(self) -> None:
        """Verify simple mode can also take the reference transcript on the command line."""
        with TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "voice.wav"
            wav_path.write_bytes(b"RIFFfakeWAVE")

            with (
                mock.patch.object(
                    cac, "write_cevoice", return_value=Path("Nova.cevoice")
                ) as write_cevoice,
                mock.patch.object(
                    cac, "normalize_reference_wav_asset", return_value=b"normalized-wav"
                ) as normalize_reference_wav_asset,
                mock.patch.object(cac, "ask_required_text") as ask_required_text,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = cac.main(["Nova", str(wav_path), "Hello from Nova."])

        assert exit_code == 0
        ask_required_text.assert_not_called()
        normalize_reference_wav_asset.assert_called_once_with(wav_path)
        write_cevoice.assert_called_once_with(
            Path("Nova.cevoice"),
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"wav": b"normalized-wav"}},
            {
                "name": "Nova",
                "default_voice": cac.DEFAULT_SIMPLE_VOICE_NAME,
                "voice_order": [cac.DEFAULT_SIMPLE_VOICE_NAME],
            },
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"reference_text": "Hello from Nova."}},
        )

    def test_simple_mode_reports_missing_wav_path(self) -> None:
        """Verify simple mode exits cleanly when the WAV file is missing."""
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exit_code = cac.main(["Nova", "missing.wav", "Hello from Nova."])

        assert exit_code == 1
        assert "Error: File not found:" in stdout.getvalue()

    def test_create_cevoice_normalizes_reference_wav_assets(self) -> None:
        """Verify CEVOICE creation normalizes WAV assets before bundling them."""
        wav_path = Path("voice.wav")
        data = {
            "output_path": Path("Nova.cevoice"),
            "voices": {cac.DEFAULT_SIMPLE_VOICE_NAME: {"wav": wav_path}},
            "metadata": {
                "name": "Nova",
                "default_voice": cac.DEFAULT_SIMPLE_VOICE_NAME,
                "voice_order": [cac.DEFAULT_SIMPLE_VOICE_NAME],
            },
            "voice_metadata": {
                cac.DEFAULT_SIMPLE_VOICE_NAME: {"reference_text": "Hello from Nova."}
            },
        }

        with (
            mock.patch.object(
                cac,
                "normalize_reference_wav_asset",
                return_value=b"normalized-wav",
            ) as normalize_reference_wav_asset,
            mock.patch.object(
                cac, "write_cevoice", return_value=Path("Nova.cevoice")
            ) as write_cevoice,
        ):
            output_path = cac.create_cevoice(data)

        assert output_path == Path("Nova.cevoice")
        normalize_reference_wav_asset.assert_called_once_with(wav_path)
        write_cevoice.assert_called_once_with(
            Path("Nova.cevoice"),
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"wav": b"normalized-wav"}},
            {
                "name": "Nova",
                "default_voice": cac.DEFAULT_SIMPLE_VOICE_NAME,
                "voice_order": [cac.DEFAULT_SIMPLE_VOICE_NAME],
            },
            {cac.DEFAULT_SIMPLE_VOICE_NAME: {"reference_text": "Hello from Nova."}},
        )

    def test_no_args_runs_interactive_wizard(self) -> None:
        """Verify no CLI arguments preserve the existing wizard flow."""
        with mock.patch.object(cac, "wizard") as wizard:
            exit_code = cac.main([])

        assert exit_code == 0
        wizard.assert_called_once_with()

    def test_simple_mode_selects_indexed_writer(self) -> None:
        """Pass the requested v4 or v5 generation through the public CLI."""
        with TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "voice.wav"
            wav_path.write_bytes(b"RIFFfakeWAVE")
            for version in (4, 5):
                with (
                    self.subTest(version=version),
                    mock.patch.object(cac, f"write_cechar_v{version}") as writer,
                    mock.patch.object(
                        cac, "normalize_reference_wav_asset", return_value=b"wav"
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    assert (
                        cac.main(
                            [
                                "Nova",
                                str(wav_path),
                                "Hello.",
                                "--format-version",
                                str(version),
                            ]
                        )
                        == 0
                    )
                    assert (
                        writer.call_args.args[3]["Standard"]["reference_text"]
                        == "Hello."
                    )

    def test_v5_wizard_collects_singing_reference(self) -> None:
        """Collect singing metadata without speaking-only overrides or embeddings."""
        with (
            mock.patch.object(
                cac,
                "ask_required_text",
                side_effect=["sing", "Singing.", "[c4:4]Singing."],
            ),
            mock.patch.object(cac, "ask_existing_file", return_value=Path("sing.wav")),
            mock.patch.object(cac, "ask", side_effect=["-1", "60", "59", "72"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            name, assets, metadata = cac.collect_voice_assets(1, 5)
        assert (name, assets) == ("sing", {"wav": Path("sing.wav")})
        assert metadata == {
            "reference_text": "Singing.",
            "cesn": "[c4:4]Singing.",
            "range": {"min": 60, "max": 72},
        }

    def test_collect_voice_assets_accepts_a_separate_display_name(self) -> None:
        """Retain the backend identifier while collecting a readable label."""
        with (
            mock.patch.object(cac, "ask_required_text", return_value="balanced"),
            mock.patch.object(cac, "ask_existing_file", return_value=Path("voice.wav")),
            mock.patch.object(cac, "ask_optional_existing_file", return_value=None),
            mock.patch.object(cac, "ask_optional_float", return_value=None),
            mock.patch.object(
                cac, "ask_optional_text", side_effect=["Hello.", "Balanced"]
            ),
            mock.patch.object(cac, "collect_voice_persona_metadata", return_value=None),
        ):
            name, _assets, metadata = cac.collect_voice_assets(1, 5)

        assert name == "balanced"
        assert metadata == {"reference_text": "Hello.", "display_name": "Balanced"}

    def test_version_option_selects_wizard(self) -> None:
        """Allow format selection when no simple-mode arguments are supplied."""
        with mock.patch.object(cac, "wizard") as wizard:
            assert cac.main(["--format-version", "5"]) == 0
        wizard.assert_called_once_with(5)

    def test_indexed_packs_reencode_non_pcm16_references(self) -> None:
        """Canonicalize an already mono 24 kHz float WAV before indexed writing."""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "reference.wav"
            sf.write(source, np.zeros(240, dtype=np.float32), 24000, subtype="FLOAT")
            for version, writer in ((4, write_cechar_v4), (5, write_cechar_v5)):
                with (
                    self.subTest(version=version),
                    mock.patch.object(cac, "np", np),
                    mock.patch.object(cac, "sf", sf),
                    mock.patch.object(cac, f"write_cechar_v{version}", writer),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    data = cac.build_simple_character_data("Nova", source, "Hello.")
                    data["output_path"] = Path(directory) / f"v{version}.cevoice"
                    bundle = CEVoice.open(cac.create_cevoice(data, version))
                    assert bundle.metadata["version"] == version
                    assert bundle.voice_order == ("Standard",)
                    assert (
                        sf.info(
                            io.BytesIO(bundle.read_asset("Standard", "wav"))
                        ).subtype
                        == "PCM_16"
                    )
