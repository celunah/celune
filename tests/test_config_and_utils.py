# SPDX-License-Identifier: MIT
"""Tests for configuration and lightweight utility helpers."""

import datetime
import math
import unittest
from unittest import mock

from celune import config, utils


class ConfigTests(unittest.TestCase):
    def test_env_bool_uses_fallback_and_strict_enabled_values(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(config.env_bool("MISSING", fallback=True), True)

        with mock.patch.dict("os.environ", {"FLAG": " Enabled "}):
            self.assertEqual(config.env_bool("FLAG"), True)

        with mock.patch.dict("os.environ", {"FLAG": "definitely"}):
            self.assertEqual(config.env_bool("FLAG", fallback=True), False)

    def test_config_value_and_config_bool_precedence(self) -> None:
        values = {"enabled": True}
        self.assertEqual(config.config_value(values, "enabled"), True)
        self.assertEqual(config.config_value(None, "missing", 3), 3)

        with mock.patch.dict("os.environ", {"CELUNE_TEST": "false"}):
            self.assertEqual(
                config.config_bool(values, "CELUNE_TEST", "enabled", default=False),
                False,
            )

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                config.config_bool(values, "CELUNE_TEST", "enabled", default=False),
                True,
            )


class UtilsTests(unittest.TestCase):
    def test_format_number_handles_precision_and_non_finite_values(self) -> None:
        self.assertEqual(utils.format_number(12.3400, 3), "12.34")
        self.assertEqual(utils.format_number(0.0), "0")
        self.assertEqual(utils.format_number(math.nan), "N/A")
        with self.assertRaisesRegex(ValueError, "precision must be >= 0"):
            utils.format_number(1.0, -1)

    def test_color_and_text_helpers_validate_inputs(self) -> None:
        self.assertEqual(utils.to_rgb("#abc"), (170, 187, 204))
        self.assertEqual(utils.to_rgb("0x00ff7f"), (0, 255, 127))
        with self.assertRaisesRegex(ValueError, "expected a 3 or 6-character"):
            utils.to_rgb("zzzzzz")

        self.assertEqual(utils.indent("Celune", 2), "  Celune")
        self.assertEqual(utils.indent("Celune", 2, "right"), "Celune  ")
        with self.assertRaisesRegex(ValueError, "can't indent"):
            utils.indent("Celune", 2, "up")

        self.assertEqual(utils.title_case("celune"), "Celune")

    def test_lunar_cuda_and_interpolation_helpers(self) -> None:
        phase, illumination, days = utils.lunar_info(
            datetime.datetime(2000, 1, 6, 18, 14, tzinfo=datetime.timezone.utc)
        )
        self.assertAlmostEqual(phase, 0.0, places=6)
        self.assertAlmostEqual(illumination, 0.0, places=6)
        self.assertGreater(days, 14.0)
        self.assertEqual(utils.lunar_phase(0.5), "full moon")
        self.assertEqual(utils.range_interpolated(-1.0, 10, 20), 10.0)
        self.assertEqual(utils.range_interpolated(1.0, 10, 20), 20.0)
        self.assertEqual(utils.cuda_architecture((8, 9)), "Ada Lovelace")
        with self.assertRaises(NotImplementedError):
            utils.cuda_architecture((7, 5))
        with self.assertRaises(ValueError):
            utils.cuda_architecture((8, 5))

    def test_assertions_language_and_random_replacement(self) -> None:
        utils.custom_assert(True, RuntimeError("unused"))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            utils.custom_assert(False, RuntimeError("failed"))
        with self.assertRaises(AssertionError):
            utils.custom_assert(False, None)
        with self.assertRaises(TypeError):
            utils.custom_assert(False, "invalid")  # type: ignore[arg-type]

        with mock.patch("celune.utils.langdetect.detect", return_value="en"):
            with mock.patch(
                "celune.utils.langdetect.detect_langs",
                return_value=[mock.Mock(lang="en", prob=0.9)],
            ):
                result = utils.detect_language("Hello", ["en"])
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["supported"], True)

        with mock.patch(
            "celune.utils.langdetect.detect",
            side_effect=utils.langdetect.LangDetectException(0, "missing"),
        ):
            result = utils.detect_language("", ["en"])
        self.assertEqual(result["probabilities"], {"en": 1.0})

        with mock.patch("celune.utils.random.random", return_value=0.0):
            with mock.patch("celune.utils.random.choice", return_value="celine"):
                self.assertEqual(
                    utils.rng_replace("CELUNE Celune celune", ["celune"], ["celine"]),
                    "CELINE Celine celine",
                )

    def test_discard_can_clear_attributes(self) -> None:
        holder = mock.Mock()
        holder.value = "present"
        self.assertIsNone(utils.discard("unused"))
        self.assertIsNone(utils.discard(holder, "value"))
        self.assertIsNone(holder.value)
