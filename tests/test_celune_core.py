# SPDX-License-Identifier: MIT
"""Tests for Celune core behavior without real models or GPU work."""

import unittest
from pathlib import Path
from unittest import mock

from celune.celune import Celune
from celune.exceptions import BackendError

from tests.support import FakeBackend, FakeGlow


class CeluneCoreTests(unittest.TestCase):
    def _make_celune(self, config: dict) -> Celune:
        with mock.patch("celune.celune.AudioRGBGlow", FakeGlow):
            with mock.patch("celune.celune.default_loader", return_value=None):
                return Celune(config=config, tts_backend=FakeBackend)

    def test_constructor_validates_backend_and_chunk_size(self) -> None:
        with self.assertRaisesRegex(BackendError, "no backend set"):
            Celune(config={}, tts_backend=None)

        celune = self._make_celune({})
        self.assertEqual(celune.chunk_size, 8)
        self.assertEqual(celune.glow.started, True)

        with mock.patch("celune.celune.AudioRGBGlow", FakeGlow):
            with mock.patch("celune.celune.default_loader", return_value=None):
                with self.assertRaisesRegex(BackendError, "invalid chunk length"):
                    Celune(
                        config={},
                        tts_backend=FakeBackend,
                        target_chunk_length=0.65,
                    )

    def test_voice_loading_uses_backend_and_bundle_defaults(self) -> None:
        celune = self._make_celune({})
        self.assertEqual(celune.load_available_voices(), True)
        self.assertEqual(celune.voices, ("balanced", "bold"))
        self.assertEqual(celune.current_voice, "balanced")

        fake_bundle = mock.Mock()
        fake_bundle.voice_order = ("bold", "balanced")
        fake_bundle.metadata = {"default_voice": "bold"}
        fake_loader = mock.Mock(bundle=fake_bundle)
        celune.backend.uses_voice_bundles = True
        with mock.patch("celune.celune.default_loader", return_value=fake_loader):
            self.assertEqual(celune.load_voice_bundle(Path("fixture.cevoice")), True)
        self.assertEqual(celune.current_voice, "bold")

    def test_logging_waiting_and_api_settings_cover_edge_cases(self) -> None:
        logs: list[tuple[str, str]] = []
        celune = self._make_celune({"api": {"port": "bad", "rate_limit_per_minute": "bad"}})
        celune.log_callback = lambda msg, severity="info": logs.append((msg, severity))
        celune.log("hello")
        self.assertEqual(logs[-1], ("hello", "info"))
        celune.log_dev("hidden")
        self.assertEqual(len(logs), 1)
        celune.dev = True
        celune.log_dev("visible")
        self.assertEqual(logs[-1], ("visible", "info"))

        celune.loaded = False
        self.assertEqual(celune._wait_until_idle(timeout=0), False)
        celune.loaded = True
        celune.locked = False
        self.assertEqual(celune._wait_until_idle(timeout=0), True)

        self.assertEqual(
            celune._api_settings(),
            (True, "0.0.0.0", 2060, None, 60),
        )
        self.assertEqual(logs[-2][1], "warning")
        self.assertEqual(logs[-1][1], "warning")

    def test_load_success_and_model_failure_paths_are_stubbed(self) -> None:
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            thread_cls.return_value.start = mock.Mock()
            with mock.patch("celune.celune.validate_runtime", return_value=True):
                with mock.patch("celune.celune.play_readiness_signal", return_value=False):
                    self.assertEqual(celune.load(), True)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(celune.glow.entered, True)

        failing = self._make_celune({})
        failing.setup_extensions = mock.Mock()
        failing.backend.preload_models = mock.Mock()
        failing.backend.load_default_model = mock.Mock(side_effect=RuntimeError("boom"))
        errors: list[str] = []
        failing.error_callback = errors.append
        self.assertEqual(failing.load(), False)
        self.assertEqual(errors, ["Default model failed to load"])

    def test_unload_runtime_state_clears_models_without_cuda(self) -> None:
        celune = self._make_celune({})
        celune.model = object()
        celune.llm = object()
        celune.tokenizer = object()
        celune.backend.model = object()
        with mock.patch("celune.celune.torch.cuda.is_available", return_value=False):
            celune.unload_runtime_state(include_normalizer=True)
        self.assertIsNone(celune.model)
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)
        self.assertIsNone(celune.backend.model)
