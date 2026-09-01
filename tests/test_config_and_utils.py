# SPDX-License-Identifier: Apache-2.0
"""Tests for configuration and lightweight utility helpers."""

import math
import datetime
from unittest import mock
from collections.abc import Mapping
from typing import Literal, Optional, cast
import pytest
from celune import config, utils
from celune.typing.common import JSON, JSONSerializable

from .support import CeluneTestCase


class TestConfig(CeluneTestCase):
    """Tests for configuration value resolution."""

    def test_env_bool_uses_fallback_and_strict_enabled_values(self) -> None:
        """Verify strict environment booleans and unset fallbacks.

        Raises:
            AssertionError: Environment parsing changes unexpectedly.
        """
        with mock.patch.dict("os.environ", {}, clear=True):
            assert config.env_bool("MISSING", fallback=True)

        with mock.patch.dict("os.environ", {"FLAG": " Enabled "}):
            assert config.env_bool("FLAG")

        with mock.patch.dict("os.environ", {"FLAG": "definitely"}):
            assert not config.env_bool("FLAG", fallback=True)

    def test_config_value_and_config_bool_precedence(self) -> None:
        """Verify configuration lookup and environment precedence.

        Raises:
            AssertionError: Configuration precedence changes unexpectedly.
        """
        values: config.Config = {"enabled": True}
        assert config.config_value(values, "enabled")
        assert config.config_value(None, "missing", 3) == 3

        with mock.patch.dict("os.environ", {"CELUNE_TEST": "false"}):
            assert not config.config_bool(
                values, "CELUNE_TEST", "enabled", default=False
            )

        with mock.patch.dict("os.environ", {}, clear=True):
            assert config.config_bool(values, "CELUNE_TEST", "enabled", default=False)

    def test_config_audio_device_supports_null_name_and_index(self) -> None:
        """Verify audio-device config accepts null, trimmed names, and indices."""
        assert config.config_audio_device({}, "missing") is None
        assert config.config_audio_device({"device": None}, "device") is None
        assert config.config_audio_device({"device": "   "}, "device") is None
        assert (
            config.config_audio_device({"device": "  Stereo Mix  "}, "device")
            == "Stereo Mix"
        )
        assert config.config_audio_device({"device": 3}, "device") == 3
        assert config.config_audio_device({"device": True}, "device") is None

    def test_config_audio_device_appends_windows_hostapi_from_audio_api(self) -> None:
        """Verify Windows audio config auto-appends the selected host API."""
        with mock.patch("celune.config.os.name", "nt"):
            assert (
                config.config_audio_device(
                    {"device": "Razer Kraken V4 - Chat", "audio_api": "wasapi"},
                    "device",
                )
                == "Razer Kraken V4 - Chat, Windows WASAPI"
            )

    def test_config_audio_device_preserves_explicit_windows_hostapi_suffix(
        self,
    ) -> None:
        """Verify an explicit Windows host API suffix is kept stable."""
        with mock.patch("celune.config.os.name", "nt"):
            assert (
                config.config_audio_device(
                    {
                        "device": "Razer Kraken V4 - Chat, Windows DirectSound",
                        "audio_api": "wasapi",
                    },
                    "device",
                )
                == "Razer Kraken V4 - Chat, Windows DirectSound"
            )

    def test_config_audio_api_accepts_supported_windows_hostapis(self) -> None:
        """Verify Windows host API config only accepts supported values."""
        assert config.config_audio_api({}) is None
        assert config.config_audio_api({"audio_api": " WASAPI "}) == "wasapi"
        assert config.config_audio_api({"audio_api": "directsound"}) == "directsound"
        assert config.config_audio_api({"audio_api": "wdm-ks"}) is None

    def test_resolve_audio_device_formats_friendly_multiple_input_matches(self) -> None:
        """Verify ambiguous input device names raise a friendly localized message."""
        devices = [
            {
                "name": "CABLE-A Output (VB-Audio Cable A)",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "hostapi": 0,
            },
            {
                "name": "CABLE-A Output (VB-Audio Cable A)",
                "max_input_channels": 8,
                "max_output_channels": 0,
                "hostapi": 1,
            },
        ]
        hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.sd.query_devices", return_value=devices),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
            pytest.raises(
                ValueError, match="the specified input device name has multiple matches"
            ) as caught,
        ):
            config.resolve_audio_device(
                {"input_device": "CABLE-A Output"},
                "input_device",
                "input",
            )

        message = str(caught.value)
        assert "- [0] CABLE-A Output (VB-Audio Cable A), MME" in message
        assert "- [1] CABLE-A Output (VB-Audio Cable A), Windows WASAPI" in message
        assert "please specify one of the above devices" in message

    def test_resolve_audio_device_prefers_exact_single_output_index(self) -> None:
        """Verify one matching output device is resolved to its exact index."""
        devices = [
            {
                "name": "Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "hostapi": 0,
            },
            {
                "name": "CABLE-B Input (VB-Audio Cable B)",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "hostapi": 1,
            },
        ]
        hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.sd.query_devices", return_value=devices),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {"output_device": "CABLE-B Input"},
                "output_device",
                "output",
            )

        assert resolved == 1

    def test_resolve_audio_device_accepts_single_device_mapping_shape(self) -> None:
        """Verify resolver tolerates a direct device-info mapping from mocks."""
        device_info = {
            "name": "Stereo Mix (Realtek)",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "hostapi": 0,
        }
        hostapis = [{"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.sd.query_devices", return_value=device_info),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {"input_device": "Stereo Mix (Realtek)"},
                "input_device",
                "input",
            )

        assert resolved == 0

    def test_resolve_audio_device_filters_windows_hostapi_matches(self) -> None:
        """Verify Windows host API config resolves ambiguous device names cleanly."""
        devices = [
            {
                "name": "Microphone (Razer Kraken V4 - Chat)",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "hostapi": 0,
            },
            {
                "name": "Microphone (Razer Kraken V4 - Chat)",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "hostapi": 1,
            },
        ]
        hostapis = [{"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.os.name", "nt"),
            mock.patch("celune.config.sd.query_devices", return_value=devices),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {
                    "input_device": "Microphone (Razer Kraken V4 - Chat)",
                    "audio_api": "wasapi",
                },
                "input_device",
                "input",
            )

        assert resolved == 1

    def test_resolve_audio_device_returns_exact_index_after_direct_query_on_windows(
        self,
    ) -> None:
        """Verify Windows host API selection returns an exact index, not an ambiguous string."""
        direct_info = {
            "name": "CABLE-B Input (VB-Audio Cable B)",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "hostapi": 1,
        }
        devices = [
            {
                "name": "CABLE-B Input (VB-Audio Cable B)",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "hostapi": 0,
            },
            direct_info,
        ]
        hostapis = [{"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.os.name", "nt"),
            mock.patch(
                "celune.config.sd.query_devices",
                side_effect=[direct_info, devices],
            ),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {
                    "output_device": "CABLE-B Input (VB-Audio Cable B)",
                    "audio_api": "wasapi",
                },
                "output_device",
                "output",
            )

        assert resolved == 1

    def test_resolve_audio_device_accepts_sequence_results_from_sounddevice(
        self,
    ) -> None:
        """Verify exact-index recovery works when sounddevice returns non-list sequences."""
        direct_info = {
            "name": "CABLE-B Input (VB-Audio Cable B)",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "hostapi": 1,
        }
        devices = (
            {
                "name": "CABLE-B Input (VB-Audio Cable B)",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "hostapi": 0,
            },
            direct_info,
        )
        hostapis = (
            {"name": "Windows DirectSound"},
            {"name": "Windows WASAPI"},
        )

        with (
            mock.patch("celune.config.os.name", "nt"),
            mock.patch(
                "celune.config.sd.query_devices",
                side_effect=[direct_info, devices],
            ),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {
                    "output_device": "CABLE-B Input (VB-Audio Cable B)",
                    "audio_api": "wasapi",
                },
                "output_device",
                "output",
            )

        assert resolved == 1

    def test_resolve_audio_device_accepts_appended_windows_hostapi_selector(
        self,
    ) -> None:
        """Verify copied runtime labels can disambiguate Windows devices directly."""
        devices = [
            {
                "name": "Microphone (Razer Kraken V4 - Chat)",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "hostapi": 0,
            },
            {
                "name": "Microphone (Razer Kraken V4 - Chat)",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "hostapi": 1,
            },
        ]
        hostapis = [{"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]

        with (
            mock.patch("celune.config.os.name", "nt"),
            mock.patch("celune.config.sd.query_devices", return_value=devices),
            mock.patch("celune.config.sd.query_hostapis", return_value=hostapis),
        ):
            resolved = config.resolve_audio_device(
                {
                    "input_device": (
                        "Microphone (Razer Kraken V4 - Chat), Windows WASAPI"
                    ),
                },
                "input_device",
                "input",
            )

        assert resolved == 1

    def test_format_audio_device_name_appends_windows_hostapi(self) -> None:
        """Verify runtime labels show the Windows host API when available."""
        with mock.patch("celune.config.os.name", "nt"):
            label = config.format_audio_device_name(
                {"name": "Microphone", "hostapi": 1},
                [{"name": "MME"}, {"name": "Windows WASAPI"}],
            )

        assert label == "Microphone, Windows WASAPI"

    def test_merge_missing_defaults_preserves_user_values_and_adds_nested_keys(
        self,
    ) -> None:
        """Verify old configs gain new defaults without losing custom values.

        Raises:
            AssertionError: Config merging behavior changes unexpectedly.
        """
        current: Optional[Mapping[str, JSONSerializable]] = {
            "backend": "qwen3",
            "api": {"enabled": False, "port": 9999},
            "theme": "light",
        }
        defaults: JSON = {
            "backend": None,
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 2060,
                "token": None,
            },
            "theme": "dark",
            "voice_bundle": "default",
        }

        merged, changed = config.merge_missing_defaults(current, defaults)

        assert changed
        assert merged["backend"] == "qwen3"
        assert merged["theme"] == "light"
        assert merged["voice_bundle"] == "default"
        assert merged["api"] == {
            "enabled": False,
            "host": "0.0.0.0",
            "port": 9999,
            "token": None,
        }
        assert current == {
            "backend": "qwen3",
            "api": {"enabled": False, "port": 9999},
            "theme": "light",
        }

    def test_merge_missing_defaults_keeps_non_mapping_user_overrides(self) -> None:
        """Verify explicit scalar overrides are not replaced by nested defaults.

        Raises:
            AssertionError: Config merging behavior changes unexpectedly.
        """
        merged, changed = config.merge_missing_defaults(
            {"api": False},
            {"api": {"enabled": True}},
        )

        assert not changed
        assert merged == {"api": False}


class TestUtils(CeluneTestCase):
    """Tests for lightweight common utility functions."""

    def test_special_character_normalization_keeps_default_mode_and_formats_tts(
        self,
    ) -> None:
        """Verify TTS-only formatting handles technical punctuation and Markdown."""
        self.assertEqual(
            utils.normalize_special_characters("“hello”—*world*…"),
            '"hello": world...',
        )
        self.assertEqual(
            utils.normalize_special_characters(r"C:\Users\user", for_tts=True),
            "C drive, Users, user",
        )
        self.assertEqual(
            utils.normalize_special_characters("foo_bar.py", for_tts=True),
            "foo underscore bar dot py",
        )
        self.assertEqual(
            utils.normalize_special_characters('{status: "ok"}', for_tts=True),
            "status, ok",
        )
        self.assertEqual(
            utils.normalize_special_characters(
                "# **status** / ready - now",
                for_tts=True,
            ),
            "status, ready, now",
        )

    def test_format_number_handles_precision_and_non_finite_values(self) -> None:
        """Verify number formatting and invalid precision handling.

        Raises:
            AssertionError: Formatting behavior changes unexpectedly.
        """
        assert utils.format_number(12.3400, 3) == "12.34"
        assert utils.format_number(0.0) == "0"
        assert utils.format_number(math.nan) == "N/A"
        with pytest.raises(ValueError, match="precision must be >= 0"):
            utils.format_number(1.0, -1)

    def test_color_and_text_helpers_validate_inputs(self) -> None:
        """Verify RGB parsing and simple text helpers.

        Raises:
            AssertionError: Helper behavior changes unexpectedly.
        """
        assert utils.to_rgb("#abc") == (170, 187, 204)
        assert utils.to_rgb("0x00ff7f") == (0, 255, 127)
        with pytest.raises(ValueError, match="expected a 3 or 6-character"):
            utils.to_rgb("zzzzzz")

        assert utils.indent("Celune", 2) == "  Celune"
        assert utils.indent("Celune", 2, "right") == "Celune  "
        with pytest.raises(ValueError, match="can't indent"):
            utils.indent(
                "Celune",
                2,
                cast(Literal["left", "right"], "up"),
            )

        assert utils.title_case("celune") == "Celune"

    def test_lunar_cuda_and_interpolation_helpers(self) -> None:
        """Verify lunar, interpolation, and CUDA label helpers.

        Raises:
            AssertionError: Helper output changes unexpectedly.
        """
        phase, illumination, days = utils.lunar_info(
            datetime.datetime(2000, 1, 6, 18, 14, tzinfo=datetime.UTC)
        )
        assert phase == pytest.approx(0.0)
        assert illumination == pytest.approx(0.0)
        assert days > 14.0
        assert utils.lunar_phase(0.5) == "full moon"
        assert utils.range_interpolated(-1.0, 10, 20) == 10.0
        assert utils.range_interpolated(1.0, 10, 20) == 20.0
        assert utils.cuda_architecture((8, 9)) == "Ada Lovelace"
        with pytest.raises(NotImplementedError):
            utils.cuda_architecture((7, 5))
        with pytest.raises(ValueError):
            utils.cuda_architecture((8, 5))

    def test_assertions_language_and_random_replacement(self) -> None:
        """Verify assertions, language fallback, and random replacement.

        Raises:
            AssertionError: Utility behavior changes unexpectedly.
        """
        utils.custom_assert(True, RuntimeError("unused"))

        with pytest.raises(RuntimeError, match="failed"):
            utils.custom_assert(False, RuntimeError("failed"))

        with pytest.raises(AssertionError):
            utils.custom_assert(False, None)

        with pytest.raises(TypeError):
            utils.custom_assert(False, "invalid")  # type: ignore[arg-type]

        result = utils.detect_language("Hello, how are you today?", ["en"])
        assert result["language"] == "en"
        assert result["supported"]

        result = utils.detect_language(
            "Bonjour, comment allez-vous aujourd'hui?", ["en"]
        )
        assert result["language"] == "fr"
        assert not result["supported"]

        result = utils.detect_language("", ["en"])
        assert result["probabilities"] == {"en": 1.0}

        with (
            mock.patch("celune.utils.random.random", return_value=0.0),
            mock.patch("celune.utils.random.choice", return_value="celine"),
        ):
            assert (
                utils.rng_replace("CELUNE Celune celune", ["celune"], ["celine"])
                == "CELINE Celine celine"
            )

    def test_discard_can_clear_attributes(self) -> None:
        """Verify ``discard`` consumes values and clears attributes.

        Raises:
            AssertionError: Discard behavior changes unexpectedly.
        """
        holder = mock.Mock()
        holder.value = "present"
        assert utils.discard("unused") is None
        assert utils.discard(holder, "value") is None
        assert cast(Optional[str], holder.value) is None

    def test_detected_ide_recognizes_supported_markers(self) -> None:
        """Verify the supported IDE environment hints."""
        with mock.patch.dict("os.environ", {"PYCHARM_HOSTED": "1"}, clear=True):
            assert utils.detected_ide() == "PyCharm"
        with mock.patch.dict("os.environ", {"TERM_PROGRAM": "vscode"}, clear=True):
            assert utils.detected_ide() == "VS Code"
        with mock.patch.dict("os.environ", {"TERM_PROGRAM": "wezterm"}, clear=True):
            assert utils.detected_ide() is None
