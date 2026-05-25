# SPDX-License-Identifier: MIT
"""Tests for Celune core behavior without real models or GPU work."""

from pathlib import Path
from typing import cast
from unittest import mock, TestCase

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from celune.celune import Celune
from celune.config import Config
from celune.exceptions import BackendError

from tests.support import FakeBackend, FakeGlow


class CeluneCoreTests(TestCase):
    """Tests for Celune orchestration without real model work."""

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    def _make_celune(self, config: dict) -> Celune:
        """Build a Celune instance with lightweight fakes.

        Args:
            config: Configuration dictionary supplied to Celune.

        Returns:
            Celune: A Celune instance with fake glow and backend objects.

        Raises:
            BackendError: Celune initialization rejects the supplied config.
        """
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(config=config, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)
            return celune

    def test_constructor_validates_backend_and_chunk_size(self) -> None:
        """Verify constructor validation and derived chunk size behavior.

        Returns:
            None: Assertions verify constructor behavior.

        Raises:
            AssertionError: Constructor behavior changes unexpectedly.
        """
        with self.assertRaisesRegex(BackendError, "no backend set"):
            Celune(config={}, tts_backend=None)

        celune = self._make_celune({})
        self.assertEqual(celune.chunk_size, 8)
        self.assertEqual(getattr(celune.glow, "started"), True)
        celune.close()

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            self.assertRaisesRegex(BackendError, "invalid chunk length"),
        ):
            Celune(
                config={},
                tts_backend=FakeBackend,
                target_chunk_length=0.65,
            )

    def test_voice_loading_uses_backend_and_bundle_defaults(self) -> None:
        """Verify backend voices and bundle metadata determine defaults.

        Returns:
            None: Assertions verify voice selection behavior.

        Raises:
            AssertionError: Voice loading behavior changes unexpectedly.
        """
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

    def test_persona_connection_uses_in_process_runtime(self) -> None:
        """Verify Celune connects to Persona through the local in-process runtime.

        Returns:
            None: Assertions verify Persona client setup.

        Raises:
            AssertionError: Persona connection behavior changes unexpectedly.
        """
        client = mock.Mock()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch(
                "celune.celune.persona_is_available", return_value=True
            ) as available,
            mock.patch(
                "celune.celune.create_persona_client", return_value=client
            ) as create_client,
        ):
            celune = Celune(
                config={"persona": {"enabled": True}}, tts_backend=FakeBackend
            )
            self.addCleanup(self._close_celune, celune)

            self.assertIs(celune.vision, client)
            create_client.assert_called_once()
            self.assertEqual(
                create_client.call_args.args[0],
                {"persona": {"enabled": True}},
            )
            log_dev = create_client.call_args.kwargs["log_dev"]
            self.assertIs(getattr(log_dev, "__self__", None), celune)
            self.assertIs(getattr(log_dev, "__func__", None), Celune.log_dev)
            available.assert_called_once_with()
            celune.close()
            client.close.assert_called_once_with()

    def test_persona_client_is_created_when_runtime_is_available(self) -> None:
        """Verify the Persona helper creates a local client when available.

        Returns:
            None: Assertions verify Persona client creation.

        Raises:
            AssertionError: Persona client creation changes unexpectedly.
        """
        from celune import persona

        with mock.patch("celune.persona.persona_is_available", return_value=True):
            client = persona.create_persona_client({"persona": {"enabled": True}})

        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(persona.persona_model_id(), "Qwen/Qwen2.5-VL-3B-Instruct")
        client.close()

    def test_load_preloads_persona_runtime_when_available(self) -> None:
        """Verify Celune explicitly loads Persona during startup."""
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        persona_client = mock.Mock()
        celune.vision = persona_client
        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_readiness_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            self.assertEqual(celune.load(), True)

        persona_client.load.assert_called_once_with(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "4bit",
        )

    def test_load_disables_persona_when_preload_fails(self) -> None:
        """Verify Persona preload failures fall back to speech-only mode."""
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        persona_client = mock.Mock()
        persona_client.load.side_effect = RuntimeError("persona boom")
        celune.vision = persona_client
        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_readiness_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            self.assertEqual(celune.load(), True)

        persona_client.close.assert_called_once_with()
        self.assertIsNone(celune.vision)

    def test_persona_talkback_config_can_disable_persona_input_mode(self) -> None:
        """Verify persona talkback can be disabled without disabling Persona."""
        from celune.persona.impl import persona_enabled, persona_talkback_enabled

        persona_config: Config = {"enabled": True, "talkback": False}
        config: Config = {"persona": persona_config}
        self.assertEqual(persona_enabled(config), True)
        self.assertEqual(persona_talkback_enabled(config), False)
        self.assertEqual(persona_talkback_enabled({"persona": {}}), True)
        self.assertEqual(persona_talkback_enabled({"pyop": {"talkback": False}}), False)

    def test_think_reconnects_to_persona_before_speech_fallback(self) -> None:
        """Verify stale Celune instances reconnect to Persona on the next think call.

        Returns:
            None: Assertions verify lazy Persona reconnect behavior.

        Raises:
            AssertionError: Persona reconnect behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.vision = None
        celune.locked = False
        celune.cur_state = "idle"
        client = mock.Mock()
        celune._persona_conn = mock.Mock(return_value=client)
        with (
            mock.patch("celune.celune.think_pipeline", return_value=True) as think,
            mock.patch.object(celune, "say", return_value=False) as say,
        ):
            self.assertEqual(celune.think("hello"), True)
            persona_thread = celune._persona_thread
            self.assertIsNotNone(persona_thread)
            assert persona_thread is not None
            persona_thread.join(timeout=2)

        self.assertIs(celune.vision, client)
        think.assert_called_once_with(celune, "hello")
        say.assert_not_called()

    def test_logging_waiting_and_api_settings_cover_edge_cases(self) -> None:
        """Verify logging gates, readiness checks, and API fallbacks.

        Returns:
            None: Assertions verify core utility behavior.

        Raises:
            AssertionError: Core utility behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        celune = self._make_celune(
            {"api": {"port": "bad", "rate_limit_per_minute": "bad"}}
        )
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
            (True, "127.0.0.1", 2060, None, 60),
        )
        self.assertEqual(logs[-2][1], "warning")
        self.assertEqual(logs[-1][1], "warning")

    def test_load_success_and_model_failure_paths_are_stubbed(self) -> None:
        """Verify successful startup and default-model failure handling.

        Returns:
            None: Assertions verify startup behavior.

        Raises:
            AssertionError: Startup behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_readiness_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            self.assertEqual(celune.load(), True)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(getattr(celune.glow, "entered"), True)
        celune.close()

        failing = self._make_celune({})
        failing.setup_extensions = mock.Mock()
        failing.backend.preload_models = mock.Mock()
        failing.backend.load_default_model = mock.Mock(side_effect=RuntimeError("boom"))
        errors: list[str] = []
        failing.error_callback = errors.append
        self.assertEqual(failing.load(), False)
        self.assertEqual(errors, ["Default model failed to load"])

    def test_unload_runtime_state_clears_models_without_cuda(self) -> None:
        """Verify model references are cleared without touching CUDA.

        Returns:
            None: Assertions verify unload behavior.

        Raises:
            AssertionError: Unload behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.model = cast(PreTrainedModel, mock.Mock(spec=PreTrainedModel))
        celune.llm = cast(PreTrainedModel, mock.Mock(spec=PreTrainedModel))
        celune.tokenizer = cast(
            PreTrainedTokenizerBase,
            mock.Mock(spec=PreTrainedTokenizerBase),
        )
        celune.backend.model = mock.Mock()
        with mock.patch("celune.celune.torch.cuda.is_available", return_value=False):
            celune.unload_runtime_state(include_normalizer=True)
        self.assertIsNone(celune.model)
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)
        self.assertIsNone(celune.backend.model)
