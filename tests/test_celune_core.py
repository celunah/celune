# SPDX-License-Identifier: MIT
"""Tests for Celune core behavior without real models or GPU work."""

from pathlib import Path
import threading
from types import SimpleNamespace
from typing import cast
from unittest import mock, TestCase

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from celune.celune import Celune
from celune.config import Config
from celune.backends.qwen3 import Qwen3
from celune.exceptions import BackendError, WarmupError
from celune.persona.impl import persona_quantization
from celune.vram import QWEN3_0_6B_MODEL

from tests.support import FakeBackend, FakeGlow


class CeluneCoreTests(TestCase):
    """Tests for Celune orchestration without real model work."""

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    def _make_celune(self, config: dict) -> Celune:
        """Build a Celune instance with lightweight fakes."""
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

        Raises:
            AssertionError: Constructor behavior changes unexpectedly.
        """
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(log=lambda _msg, _severity="info": None),
            ) as resolve,
        ):
            celune = Celune(config={"vram": "low"}, tts_backend=None)
            self.addCleanup(self._close_celune, celune)

        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.args[0], "qwen3")
        self.assertEqual(
            resolve.call_args.kwargs["clone_model_id"],
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        )
        celune.close()

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

        Raises:
            AssertionError: Voice loading behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        self.assertEqual(celune.load_available_voices(), True)
        self.assertEqual(celune.voices, ("balanced", "bold"))
        self.assertEqual(celune.current_voice, "balanced")

        fake_bundle = mock.Mock()
        fake_bundle.voice_order = ("bold", "balanced")
        fake_bundle.metadata = {
            "name": "Pack Name",
            "default_voice": "bold",
            "persona": {
                "identity": {
                    "name": "Fixture",
                    "profile": "A composed observer.",
                },
                "speaking_style": "Measured and calm.",
                "style": {
                    "warmth": "high",
                    "detail": "high",
                },
            },
        }
        fake_loader = mock.Mock(bundle=fake_bundle)
        celune.backend.uses_voice_bundles = True
        with mock.patch("celune.celune.default_loader", return_value=fake_loader):
            self.assertEqual(celune.load_voice_bundle(Path("fixture.cevoice")), True)
        self.assertEqual(celune.current_voice, "bold")
        self.assertEqual(celune.current_character, "Fixture")
        self.assertIsNotNone(celune.current_character_persona)
        assert celune.current_character_persona is not None
        self.assertEqual(
            celune.current_character_persona.speaking_style,
            "Measured and calm.",
        )

    def test_persona_connection_uses_in_process_runtime(self) -> None:
        """Verify Celune connects to Persona through the local in-process runtime.

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
                config={"vram": "high", "persona": {"enabled": True}},
                tts_backend=FakeBackend,
            )
            self.addCleanup(self._close_celune, celune)

            self.assertIs(celune.vision, client)
            create_client.assert_called_once()
            self.assertEqual(
                create_client.call_args.args[0],
                {"vram": "high", "persona": {"enabled": True}},
            )
            log_dev = create_client.call_args.kwargs["log_dev"]
            self.assertIs(getattr(log_dev, "__self__", None), celune)
            self.assertIs(getattr(log_dev, "__func__", None), Celune.log_dev)
            available.assert_called_once_with()
            celune.close()
            client.close.assert_called_once_with()

    def test_persona_client_is_created_when_runtime_is_available(self) -> None:
        """Verify the Persona helper creates a local client when available.

        Raises:
            AssertionError: Persona client creation changes unexpectedly.
        """
        from celune import persona

        with mock.patch("celune.persona.persona_is_available", return_value=True):
            client = persona.create_persona_client(
                {"vram": "high", "persona": {"enabled": True}}
            )

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
        config: Config = {"vram": "high", "persona": persona_config}
        self.assertEqual(persona_enabled(config), True)
        self.assertEqual(persona_talkback_enabled(config), False)
        self.assertEqual(
            persona_talkback_enabled({"vram": "high", "persona": {}}), True
        )
        self.assertEqual(
            persona_talkback_enabled({"vram": "high", "pyop": {"talkback": False}}),
            False,
        )
        self.assertEqual(
            persona_enabled({"vram": "low", "persona": {"enabled": True}}), False
        )
        self.assertEqual(
            persona_talkback_enabled({"vram": "low", "persona": {}}), False
        )
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            self.assertEqual(persona_quantization({"vram": "high"}), "4bit")
            self.assertEqual(persona_quantization({"vram": "xhigh"}), "8bit")

    def test_low_vram_restricts_backend_to_qwen3(self) -> None:
        """Verify low VRAM falls back to the supported Qwen3 preset."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(log=lambda _msg, _severity="info": None),
            ) as resolve,
        ):
            celune = Celune(
                config={"vram": "low"},
                tts_backend="voxcpm2",
            )
            self.addCleanup(self._close_celune, celune)

        self.assertEqual(resolve.call_args.args[0], "qwen3")

    def test_voice_prompt_support_tracks_qwen3_0_6b_capability(self) -> None:
        """Verify voice prompts are disabled for the low-tier Qwen3 clone model."""
        celune = self._make_celune({})
        with mock.patch.object(Qwen3, "_validate_refs"):
            celune.backend = Qwen3(
                log=lambda _msg, _severity="info": None,
                clone_model_id=QWEN3_0_6B_MODEL,
            )
        celune.voice_prompt = "gentle"

        self.assertEqual(celune.voice_prompt_supported(), False)
        self.assertIsNone(celune.effective_voice_prompt())

    def test_low_vram_rejects_heavy_backend_types(self) -> None:
        """Verify low VRAM rejects explicitly requested heavy backend classes."""

        class HeavyBackend(FakeBackend):
            """A heavy fake backend for usage in tests."""

            name = "voxcpm2"

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            self.assertRaisesRegex(BackendError, "not available for VRAM tier 'low'"),
        ):
            Celune(config={"vram": "low"}, tts_backend=HeavyBackend)

    def test_think_reconnects_to_persona_before_speech_fallback(self) -> None:
        """Verify stale Celune instances reconnect to Persona on the next think call.

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

    def test_setup_extensions_exposes_think_to_extension_context(self) -> None:
        """Verify extension context receives Celune's think entrypoint.

        Raises:
            AssertionError: Extension context wiring changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.think = mock.Mock(return_value=True)
        with mock.patch("celune.celune.CeluneExtensionManager.autoload"):
            celune.setup_extensions()

        self.assertIsNotNone(celune.extension_manager)
        assert celune.extension_manager is not None
        think = celune.extension_manager.context.think
        self.assertEqual(think("hello"), True)
        celune.think.assert_called_once_with("hello")

    def test_logging_waiting_and_api_settings_cover_edge_cases(self) -> None:
        """Verify logging gates, readiness checks, and API fallbacks.

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
        self.assertEqual(getattr(failing.glow, "fatal_called"), True)

    def test_unload_runtime_state_clears_models_without_cuda(self) -> None:
        """Verify model references are cleared without touching CUDA.

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

    def test_unload_runtime_state_runs_model_close_hooks(self) -> None:
        """Verify TTS and normalizer teardown calls object-level release hooks."""
        celune = self._make_celune({})
        tts_model = SimpleNamespace(close=mock.Mock())
        llm = SimpleNamespace(close=mock.Mock())
        tokenizer = SimpleNamespace(close=mock.Mock())
        celune.model = cast(PreTrainedModel, tts_model)
        celune.backend.model = tts_model
        celune.llm = cast(PreTrainedModel, llm)
        celune.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

        with mock.patch("celune.celune.torch.cuda.is_available", return_value=False):
            celune.unload_runtime_state(include_normalizer=True)

        tts_model.close.assert_called_once_with()
        llm.close.assert_called_once_with()
        tokenizer.close.assert_called_once_with()
        self.assertIsNone(celune.model)
        self.assertIsNone(celune.backend.model)
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)

    def test_stale_normalizer_load_does_not_restore_released_references(self) -> None:
        """Verify background normalizer loads cannot repopulate state after unload."""
        celune = self._make_celune({})
        ready = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        fake_tokenizer = cast(
            PreTrainedTokenizerBase,
            mock.Mock(spec=PreTrainedTokenizerBase),
        )
        fake_llm = cast(PreTrainedModel, mock.Mock(spec=PreTrainedModel))

        def fake_load_components(*_args, **_kwargs):
            ready.set()
            self.assertTrue(release.wait(timeout=2))
            finished.set()
            return fake_tokenizer, fake_llm

        with (
            mock.patch(
                "celune.celune.load_normalizer_components",
                side_effect=fake_load_components,
            ),
            mock.patch("celune.celune.torch.cuda.is_available", return_value=False),
        ):
            celune.load_normalizer()
            self.assertTrue(ready.wait(timeout=2))
            celune.unload_normalizer_state()
            release.set()
            self.assertTrue(finished.wait(timeout=2))

        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)

    def test_sleep_mode_unloads_configured_models_and_wakes(self) -> None:
        """Verify sleep mode honors unload settings and reloads on wake."""
        celune = self._make_celune(
            {
                "vram": "high",
                "sleep": {
                    "enabled": True,
                    "timeout": 1,
                    "unload": {"persona": True, "normalizer": True, "tts": True},
                },
                "persona": {"enabled": True},
                "use_normalizer": True,
            }
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.model_name = "fake/balanced"
        celune.llm = cast(PreTrainedModel, mock.Mock(spec=PreTrainedModel))
        celune.tokenizer = cast(
            PreTrainedTokenizerBase,
            mock.Mock(spec=PreTrainedTokenizerBase),
        )
        persona_client = mock.Mock()
        celune.vision = persona_client
        celune._warmup = mock.Mock(return_value=True)
        celune.load_normalizer = mock.Mock()
        celune._persona_conn = mock.Mock(return_value=persona_client)
        old_backend = celune.backend

        self.assertEqual(celune.enter_sleep_mode(), True)

        self.assertEqual(celune.sleeping, True)
        self.assertEqual(celune.loaded, False)
        self.assertEqual(celune.cur_state, "sleeping")
        self.assertEqual(getattr(celune.glow, "sleep_called"), True)
        self.assertIsNone(celune.model)
        self.assertEqual(celune.model_name, "")
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)
        self.assertIsNone(celune.vision)
        persona_client.close.assert_called_once_with()

        self.assertEqual(celune.wake_from_sleep(), True)

        self.assertIsNot(celune.backend, old_backend)
        self.assertEqual(celune.sleeping, False)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(celune.cur_state, "idle")
        self.assertEqual(getattr(celune.glow, "wake_called"), True)
        self.assertEqual(celune.model, {"model_id": "fake/balanced", "kwargs": {}})
        self.assertEqual(celune.model_name, "fake/balanced")
        celune._warmup.assert_called_once_with()
        celune.load_normalizer.assert_called_once_with()
        persona_client.load.assert_called_once_with(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "4bit",
        )

    def test_sleep_mode_closes_persona_even_if_close_raises(self) -> None:
        """Verify sleep still clears Persona references when client shutdown fails."""
        celune = self._make_celune(
            {"sleep": {"enabled": True, "unload": {"persona": True, "tts": False}}}
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.vision = mock.Mock(close=mock.Mock(side_effect=RuntimeError("boom")))

        self.assertEqual(celune.enter_sleep_mode(), True)
        self.assertIsNone(celune.vision)

    def test_wake_failure_switches_glow_to_fatal_color(self) -> None:
        """Verify wake failures trigger the fixed fatal OpenRGB glow state."""
        celune = self._make_celune(
            {
                "vram": "high",
                "sleep": {
                    "enabled": True,
                    "timeout": 1,
                    "unload": {"persona": False, "normalizer": False, "tts": True},
                },
            }
        )
        celune.sleeping = True
        celune.loaded = False
        celune.cur_state = "sleeping"
        celune.current_voice = "balanced"
        celune.voices = ("balanced",)
        failing_backend = FakeBackend(log=lambda _msg, _severity="info": None)
        failing_backend.load_model = mock.Mock(side_effect=RuntimeError("boom"))

        with mock.patch("celune.celune.resolve_backend", return_value=failing_backend):
            self.assertEqual(celune.wake_from_sleep(), False)
        self.assertEqual(getattr(celune.glow, "fatal_called"), True)

    def test_raise_warmup_error_preserves_original_cause(self) -> None:
        """Verify WarmupError keeps the underlying warmup failure as its cause."""
        celune = self._make_celune({})
        cause = RuntimeError("tensor mismatch")
        celune._last_warmup_error = cause

        with self.assertRaises(WarmupError) as exc_info:
            celune._raise_warmup_error("warmup failed after sleep")

        self.assertIs(exc_info.exception.__cause__, cause)
