# SPDX-License-Identifier: Apache-2.0
"""Tests for the CEVOICE creation helper script."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest import TestCase, mock


def _load_cac_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "cac.py"
    spec = importlib.util.spec_from_file_location("scripts.cac", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/cac.py")
    module = importlib.util.module_from_spec(spec)
    fake_numpy = ModuleType("numpy")
    setattr(fake_numpy, "mean", mock.Mock())
    setattr(fake_numpy, "asarray", mock.Mock())
    fake_celune = ModuleType("celune")
    fake_cevoice = ModuleType("celune.cevoice")
    fake_soundfile = ModuleType("soundfile")
    setattr(fake_soundfile, "read", mock.Mock())
    setattr(fake_soundfile, "info", mock.Mock())
    setattr(fake_soundfile, "write", mock.Mock())
    fake_scipy = ModuleType("scipy")
    fake_scipy_signal = ModuleType("scipy.signal")
    setattr(fake_scipy_signal, "resample_poly", mock.Mock())
    setattr(fake_cevoice, "write_cevoice", mock.Mock())

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


class CACScriptTests(TestCase):
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

        self.assertEqual(exit_code, 0)
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
        self.assertIn("Saved voice pack to Nova.cevoice", stdout.getvalue())

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

        self.assertEqual(exit_code, 0)
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

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: File not found:", stdout.getvalue())

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

        self.assertEqual(output_path, Path("Nova.cevoice"))
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

        self.assertEqual(exit_code, 0)
        wizard.assert_called_once_with()
