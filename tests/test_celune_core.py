# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune core behavior without real models or GPU work."""

import json
import queue
import weakref
import tempfile
import threading
import contextlib
from pathlib import Path
from typing import Optional, cast
from types import SimpleNamespace, MappingProxyType
from unittest import mock
from collections.abc import Callable

import numpy as np
import pytest
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from celune import i18n, cevoice
from celune.utils import discard
from celune.celune import Celune
from celune.config import Config
from celune.typing.common import JSONSerializable
from celune.typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from celune.pipeline import close as close_pipeline
from celune.persona.impl import persona_quantization
from celune.exceptions import WarmupError, BackendError, CEDTSTimeoutError
from celune.persona.emotion import PersonaEmotionAnalyzer
from celune.agent import (
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentCancellationReason,
)
from celune.pipeline import (
    play_signal,
    release_pipeline,
    handle_audio_input,
    convert_audio_input,
)
from celune.dataclasses.celune import (
    CeluneAudioState,
    CeluneModelState,
    CeluneVoiceState,
    CeluneBackendState,
    CeluneRuntimeState,
    CeluneCallbackState,
    CelunePipelineState,
)

from .support import (
    CeluneAsyncTestCase,
    CeluneTestCase,
    FakeBackend,
    FakeGlow,
    FakeVCBackend,
)


class TestCeluneCore(CeluneTestCase):
    """Tests for Celune orchestration without real model work."""

    _cached_celune: Optional[Celune] = None
    _cached_instance_keys: frozenset[str] = frozenset()

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    @classmethod
    def _reset_cached_celune(cls, celune: Celune) -> None:
        """Restore the reusable default test instance to constructor state."""
        if celune._playback_thread is not None:
            close_pipeline(celune)
        if celune.vision is not None or celune._persona_load_thread is not None:
            Celune._unload_persona_state(celune)

        persona_thread = getattr(celune, "_persona_thread", None)
        if (
            persona_thread is not None
            and persona_thread is not threading.current_thread()
        ):
            persona_thread.join(timeout=2)
        persona_queue = getattr(celune, "_persona_queue", None)
        if persona_queue is not None:
            while True:
                try:
                    persona_queue.get_nowait()
                except queue.Empty:
                    break

        for name in set(celune.__dict__) - cls._cached_instance_keys:
            delattr(celune, name)

        backend = FakeBackend(log=celune._noop_message, fatal=celune.fatal)
        celune._callbacks = CeluneCallbackState(
            log_callback=celune._noop_message,
            status_callback=celune._noop_message,
            error_callback=lambda _error: None,
            idle_callback=lambda: None,
            queue_avail_callback=lambda: None,
            voice_changed_callback=lambda _name: None,
            change_input_state_callback=celune._noop_input_state,
            change_voice_lock_state_callback=celune._noop_voice_lock_state,
            progress_callback=celune._noop_progress,
            caption_progress_callback=celune._noop_progress,
            caption_callback=celune._noop_caption,
            caption_timing_callback=celune._noop_caption_timing,
        )
        celune._backend_state = CeluneBackendState(
            config={},
            backend_spec=FakeBackend,
            backend_kwargs={},
            backend=backend,
            tts_backend=backend.name,
            input_mode="text_to_speech",
            chunk_size=8,
        )
        celune._model_state = CeluneModelState()
        celune._voice_state = CeluneVoiceState()
        celune._pipeline_state = CelunePipelineState(audio_queue=queue.Queue(maxsize=8))
        celune._audio_state = CeluneAudioState()
        celune._runtime_state = CeluneRuntimeState()
        celune._async_runtime_lock = threading.Lock()
        celune._voice_reload_guard = threading.Lock()
        celune._voice_reload_active = False
        celune._event_dispatcher = type(celune._event_dispatcher)(
            log_warning=celune.log,
            log_level="info",
        )
        glow = FakeGlow("#cebaff", celune=celune)
        setattr(celune, "glow", glow)  # noqa: B010
        celune._wrap_fatal_glow()
        glow.start()
        celune._model_ready.set()
        celune._playback_done.set()
        Celune._instance = None

    @classmethod
    def _cache_celune(cls, celune: Celune) -> None:
        """Install close tracking on one reusable default test instance."""
        original_close = celune.close

        def close_cached() -> None:
            """Close and invalidate the reusable default test instance."""
            if cls._cached_celune is celune:
                cls._cached_celune = None
            original_close()

        celune.close = close_cached
        cls._cached_celune = celune

    @classmethod
    def tearDownClass(cls) -> None:
        """Close the cached default test instance after the core suite."""
        if cls._cached_celune is not None:
            cached_celune = cls._cached_celune
            cls._cached_celune = None
            cached_celune.close()

    def _make_celune(
        self,
        config: dict,
        startup_callback: Optional[Callable[[str], None]] = None,
    ) -> Celune:
        """Build a Celune instance with lightweight fakes."""
        if not config and startup_callback is None:
            cached_celune = type(self)._cached_celune
            if cached_celune is not None:
                type(self)._reset_cached_celune(cached_celune)
                type(self)._cache_celune(cached_celune)
                return cached_celune

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config=config,
                tts_backend=FakeBackend,
                startup_callback=startup_callback,
            )
            if not config and startup_callback is None:
                owner = type(self)
                owner._cached_instance_keys = frozenset(celune.__dict__)
                owner._cache_celune(celune)
                # Keep the cached harness available while constructor tests use the
                # production singleton slot for their own temporary instances.
                Celune._instance = None
            else:
                self.addCleanup(self._close_celune, celune)
            return celune

    @staticmethod
    def _immediate_thread(*args, **kwargs):
        """Return a thread stub whose ``start()`` runs the target immediately."""
        target = kwargs.get("target")
        if target is None and args:
            target = args[0]
        target_args = kwargs.get("args", ())

        class _ImmediateThread:
            """Immediate thread harness used by hot-reload tests."""

            @staticmethod
            def start() -> None:
                """Run the target synchronously."""
                if target is not None:
                    target(*target_args)

        return _ImmediateThread()

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
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(config={"vram": "low"}, tts_backend=None)
            self.addCleanup(self._close_celune, celune)

        resolve.assert_called_once()
        assert resolve.call_args.args[0] == "mini"
        assert "clone_model_id" not in resolve.call_args.kwargs
        celune.close()

        celune = self._make_celune({})
        self.assertEqual(celune.chunk_size, 8)
        self.assertEqual(getattr(celune.glow, "started"), True)  # noqa: B009
        celune.close()

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            pytest.raises(BackendError, match="invalid chunk length"),
        ):
            Celune(
                config={},
                tts_backend=FakeBackend,
                target_chunk_length=0.65,
            )

    def test_callback_registration_rejects_the_same_callback_twice(self) -> None:
        """Verify Celune rejects duplicate callback registration attempts."""
        celune = self._make_celune({})

        def callback(
            msg: str,
            severity: str = "info",
            *,
            loglevel: str = "info",
        ) -> None:
            del msg, severity, loglevel

        celune.log_callback = callback
        with pytest.raises(ValueError, match="already registered"):
            celune.log_callback = callback

        celune.close()

    def test_constructor_accepts_backend_alias_for_tts_runtime(self) -> None:
        """Verify ``backend=`` can configure the TTS runtime directly."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(config={}, backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

        assert celune.backend.name == "fake"
        assert celune.tts_backend == "fake"

    def test_constructor_accepts_backend_alias_for_vc_runtime(self) -> None:
        """Verify ``backend=`` can configure the VC runtime in VC mode."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(),
            ),
        ):
            celune = Celune(config={"mode": "voice_conversion"}, backend=FakeVCBackend)
            self.addCleanup(self._close_celune, celune)

        assert celune.input_mode == "voice_conversion"
        assert celune.vc_backend is not None
        assert celune.vc_backend is not None
        assert celune.vc_backend.name == "fake-vc"

    def test_constructor_accepts_backend_alias_string_for_vc_runtime(self) -> None:
        """Verify string backend aliases resolve to VC backends when selected."""
        fake_vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        fake_vc_backend.name = "seed-vc"
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(),
            ),
            mock.patch(
                "celune.celune.resolve_vc_backend",
                return_value=fake_vc_backend,
            ) as resolve_vc_backend,
        ):
            celune = Celune(config={"mode": "voice_conversion"}, backend="seed-vc")
            self.addCleanup(self._close_celune, celune)

        resolve_vc_backend.assert_called_once_with("seed-vc", log=celune.log_callback)
        assert celune.vc_backend is not None
        assert celune.vc_backend is not None
        assert celune.vc_backend.name == "seed-vc"

    def test_constructor_uses_explicit_locale_override_from_config(self) -> None:
        """Verify an explicit config locale wins over system auto-detection."""
        previous_locale = i18n.get_locale()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch("celune.celune.get_system_locale", return_value="pl"),
        ):
            celune = Celune(config={"locale": "en-US"}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

        self.addCleanup(i18n.set_locale, previous_locale)
        assert i18n.get_locale() == "en-US"

    def test_constructor_uses_system_locale_when_no_override_is_configured(
        self,
    ) -> None:
        """Verify locale auto-selection still uses the detected system locale by default."""
        previous_locale = i18n.get_locale()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch("celune.celune.get_system_locale", return_value="pl"),
        ):
            celune = Celune(config={"locale": None}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

        self.addCleanup(i18n.set_locale, previous_locale)
        assert i18n.get_locale() == "pl"

    def test_constructor_rejects_duplicate_backend_alias_for_tts(self) -> None:
        """Verify ``backend=`` cannot be combined with ``tts_backend=``."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            pytest.raises(
                BackendError, match="cannot specify both 'backend' and 'tts_backend'"
            ),
        ):
            Celune(config={}, backend=FakeBackend, tts_backend=FakeBackend)

    def test_constructor_rejects_duplicate_backend_alias_for_vc(self) -> None:
        """Verify ``backend=`` cannot be combined with ``vc_backend=``."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            pytest.raises(
                BackendError, match="cannot specify both 'backend' and 'vc_backend'"
            ),
        ):
            Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                backend=FakeVCBackend,
                vc_backend=FakeVCBackend,
            )

    def test_load_seeds_historical_generated_speech_seconds_from_outputs(self) -> None:
        """Verify Celune seeds total savings history from existing outputs."""
        celune = self._make_celune({})

        with (
            mock.patch(
                "celune.celune.saved_output_speech_seconds",
                return_value=42.5,
            ),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            assert celune.load(skip_runtime_check=True)

        assert celune.historical_generated_speech_seconds == 42.5

    def test_load_reports_core_initialization_startup_checkpoint(self) -> None:
        """Verify ``load`` reports its dedicated startup checkpoint once."""
        startup_callback = mock.Mock()
        celune = self._make_celune({}, startup_callback=startup_callback)

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune.load(skip_runtime_check=True)

        startup_callback.assert_called_once_with(
            i18n.string("ui.startup_initializing_core")
        )

    def test_voice_loading_uses_backend_and_bundle_defaults(self) -> None:
        """Verify backend voices and bundle metadata determine defaults.

        Raises:
            AssertionError: Voice loading behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        assert celune.load_available_voices()
        assert celune.voices == ("balanced", "bold")
        assert celune.current_voice == "balanced"

        fake_bundle = mock.Mock()
        fake_bundle.path = Path("fixture.cevoice")
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
        with (
            mock.patch("celune.celune.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.celune.bundle_matches_default_pack_checksum",
                return_value=False,
            ),
        ):
            assert celune.load_voice_bundle(Path("fixture.cevoice"))
        assert celune.current_voice == "bold"
        assert celune.current_character == "Fixture"
        assert celune.current_character_persona is not None
        assert celune.current_character_persona is not None
        assert celune.current_character_persona.speaking_style == "Measured and calm."

    def test_cleanup_residual_temp_data_removes_unprotected_temp_entries(self) -> None:
        """Verify shutdown temp cleanup removes every unprotected temp entry."""
        celune = self._make_celune({})
        celune.log_callback = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            extracted_dir = temp_root / "celune-cevoice-fixture"
            extracted_dir.mkdir()
            (extracted_dir / "balanced.wav").write_bytes(b"wav")
            pocket_dir = temp_root / "celune-pocket-tts-fixture"
            pocket_dir.mkdir()
            (pocket_dir / "english-fixture.yaml").write_text(
                "weights_path: demo\n",
                encoding="utf-8",
            )
            rag_prompt = temp_root / "rag_prompt.txt"
            rag_prompt.write_text("prompt", encoding="utf-8")
            temporary_audio = temp_root / "temporary_audio.wav"
            temporary_audio.write_bytes(b"RIFFdemoWAVE")
            bundle_file = temp_root / "default.cevoice"
            bundle_file.write_bytes(b"core")
            memory_note = temp_root / "keep.txt"
            memory_note.write_text("keep", encoding="utf-8")

            celune._cleanup_residual_temp_data(temp_root)

            assert not extracted_dir.exists()
            assert not pocket_dir.exists()
            assert not rag_prompt.exists()
            assert not temporary_audio.exists()
            assert not bundle_file.exists()
            assert not memory_note.exists()

        celune.log_callback.assert_any_call(
            "Celune found 6 residual temporary items.",
            "warning",
        )
        celune.log_callback.assert_any_call("Deleting...", "warning")

    def test_cleanup_residual_temp_data_preserves_protected_temp_paths(self) -> None:
        """Verify live protected temp paths survive cleanup even when names match disposable prefixes."""
        celune = self._make_celune({})
        celune.log_callback = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            protected_dir = temp_root / "celune-cevoice-live"
            protected_dir.mkdir()
            (protected_dir / "balanced.wav").write_bytes(b"wav")
            stale_dir = temp_root / "celune-cevoice-stale"
            stale_dir.mkdir()
            stale_file = temp_root / "temporary_audio.wav"
            stale_file.write_bytes(b"RIFFdemoWAVE")

            cevoice.register_protected_temp_path(protected_dir)
            try:
                celune._cleanup_residual_temp_data(temp_root)
            finally:
                cevoice.unregister_protected_temp_path(protected_dir)

            assert protected_dir.exists()
            assert not stale_dir.exists()
            assert not stale_file.exists()

        celune.log_callback.assert_any_call(
            "Celune found 2 residual temporary items.",
            "warning",
        )

    def test_load_voice_bundle_marks_default_pack_from_checksum(self) -> None:
        """Verify default-pack detection follows the CEVOICE checksum, not the character name."""
        celune = self._make_celune({})
        fake_bundle = mock.Mock()
        fake_bundle.path = Path("renamed-default.cevoice")
        fake_bundle.voice_order = ("balanced", "bold")
        fake_bundle.metadata = {"name": "Pack Name", "default_voice": "balanced"}
        fake_loader = mock.Mock(bundle=fake_bundle)
        celune.backend.uses_voice_bundles = True

        with (
            mock.patch("celune.celune.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.celune.bundle_matches_default_pack_checksum",
                return_value=True,
            ),
        ):
            assert celune.load_voice_bundle(Path("fixture.cevoice"))

        assert celune.voice_bundle_is_default

    def test_load_voice_bundle_rejects_named_celune_without_default_checksum(
        self,
    ) -> None:
        """Verify non-default packs named Celune do not inherit default-pack behavior."""
        celune = self._make_celune({})
        fake_bundle = mock.Mock()
        fake_bundle.path = Path("custom-celune.cevoice")
        fake_bundle.voice_order = ("balanced", "bold")
        fake_bundle.metadata = {"name": "Celune", "default_voice": "balanced"}
        fake_loader = mock.Mock(bundle=fake_bundle)
        celune.backend.uses_voice_bundles = True

        with (
            mock.patch("celune.celune.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.celune.bundle_matches_default_pack_checksum",
                return_value=False,
            ),
        ):
            assert celune.load_voice_bundle(Path("fixture.cevoice"))

        assert celune.current_character == "Celune"
        assert not celune.voice_bundle_is_default

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

            assert celune.vision is client
            create_client.assert_called_once()
            assert create_client.call_args.args[0] == {
                "vram": "high",
                "persona": {"enabled": True},
            }
            log = create_client.call_args.kwargs["log"]
            assert getattr(log, "__self__", None) is celune
            assert getattr(log, "__func__", None) is Celune.log
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

        assert client is not None
        assert client is not None
        assert persona.persona_model_id() == "Qwen/Qwen3-VL-4B-Instruct"
        client.close()

    def test_load_starts_persona_after_tts_is_ready(self) -> None:
        """Verify TTS becomes ready before Persona starts downloading."""
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        persona_client = mock.Mock()
        celune.vision = persona_client
        progress_events: list[tuple[Optional[float], Optional[float]]] = []
        celune.progress_callback = lambda progress, total: progress_events.append(
            (progress, total)
        )
        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        persona_thread = thread_cls.call_args_list[-1]
        assert not celune.persona_ready
        assert celune.persona_loading
        persona_thread.kwargs["target"](*persona_thread.kwargs["args"])
        persona_client.load.assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            "4bit",
        )
        assert celune.persona_ready
        assert not celune.persona_loading
        assert progress_events[-1] == (1, 1)

    def test_load_defers_temp_cleanup_until_shutdown(self) -> None:
        """Verify temp cleanup waits until runtime shutdown after initialization."""
        celune = self._make_celune({})
        celune.backend.uses_voice_bundles = True
        call_order: list[str] = []

        def cleanup(_temp_dir: Path) -> None:
            call_order.append("cleanup")

        def close_loader() -> None:
            call_order.append("loader")

        def load_voices() -> bool:
            call_order.append("voices")
            return False

        celune._try_play_signal = mock.Mock(return_value=False)
        celune.error_callback = mock.Mock()

        with (
            mock.patch.object(
                celune, "_cleanup_residual_temp_data", side_effect=cleanup
            ),
            mock.patch("celune.celune.close_default_loader", side_effect=close_loader),
            mock.patch.object(celune, "load_available_voices", side_effect=load_voices),
            mock.patch("celune.celune.log_runtime_banner"),
        ):
            assert not celune.load()
            assert call_order == ["voices"]
            celune.close()

        assert call_order == ["voices", "loader", "cleanup"]

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
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        celune._load_persona_background(persona_client)
        persona_client.close.assert_called_once_with()
        assert celune.vision is None
        assert not celune.persona_ready

    def test_think_falls_back_to_speech_while_persona_loads(self) -> None:
        """Verify queued text uses TTS while Persona is still downloading."""
        celune = self._make_celune({})
        celune.vision = mock.Mock()
        celune.persona_loading = True
        celune.locked = False
        celune.cur_state = "idle"
        celune._wait_for_persona_playback = mock.Mock(return_value=True)
        celune.say = mock.Mock(return_value=True)
        celune._persona_queue.put("hello while downloading")

        celune._think_worker()

        celune.say.assert_called_once_with("hello while downloading")
        celune.vision.post.assert_not_called()

    def test_load_voice_conversion_mode_skips_tts_model_load_and_warmup(self) -> None:
        """Verify VC mode does not boot the TTS runtime during startup."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend=FakeVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "unused"})
        assert celune.vc_backend is not None
        celune.vc_backend.preload_models = mock.Mock()

        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        celune.backend.preload_models.assert_not_called()
        celune.backend.load_default_model.assert_not_called()
        celune._warmup.assert_not_called()
        celune.vc_backend.preload_models.assert_called_once_with()
        assert celune.model is None
        assert celune.model_name == ""
        assert celune._generation_thread is None
        assert thread_cls.call_count == 1

    def test_change_voice_returns_runtime_state_to_idle(self) -> None:
        """Verify successful voice reload leaves Celune in the idle state."""
        celune = self._make_celune({})
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.model_name = "shared-model"
        celune.loaded = True
        celune.cur_state = "idle"
        celune.backend.model_id_for_voice = mock.Mock(return_value="shared-model")
        statuses: list[tuple[str, str]] = []
        celune.status_callback = lambda msg, severity="info", *, loglevel="info": (
            statuses.append((msg, severity))
        )
        celune.voice_changed_callback = mock.Mock()

        with mock.patch("celune.celune.play_signal", return_value=False):
            celune.change_voice("bold")

        assert celune.current_voice == "bold"
        assert celune.loaded
        assert celune.cur_state == "idle"
        assert statuses[-1] == ("Idle", "info")
        celune.voice_changed_callback.assert_called_once_with("bold")

    def test_change_voice_in_voice_conversion_mode_skips_tts_reload(self) -> None:
        """Verify VC mode updates the target voice without loading TTS models."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.loaded = True
        celune.cur_state = "idle"
        celune.backend.model_id_for_voice = mock.Mock(return_value="shared-model")
        celune.backend.load_model = mock.Mock(return_value={"model": "unused"})
        celune._warmup = mock.Mock(return_value=True)
        statuses: list[tuple[str, str]] = []
        celune.status_callback = lambda msg, severity="info", *, loglevel="info": (
            statuses.append((msg, severity))
        )
        celune.voice_changed_callback = mock.Mock()

        with mock.patch("celune.celune.play_signal", return_value=False):
            celune.change_voice("bold")

        assert celune.current_voice == "bold"
        assert celune.loaded
        assert celune.cur_state == "idle"
        assert statuses[-1] == ("Idle", "info")
        celune.backend.model_id_for_voice.assert_not_called()
        celune.backend.load_model.assert_not_called()
        celune._warmup.assert_not_called()
        celune.voice_changed_callback.assert_called_once_with("bold")

    def test_voice_change_waits_for_playback_before_resetting_pipeline(self) -> None:
        """Verify voice changes drain pending playback instead of force-stopping it."""
        celune = self._make_celune({})
        celune.voices = ("balanced", "bold")
        celune.loaded = True
        celune.locked = True
        celune.model_ready.set()
        celune.playback_done.clear()
        prepare_started = threading.Event()

        def input_state(locked: bool) -> None:
            if locked:
                prepare_started.set()

        celune.change_input_state_callback = mock.Mock(
            side_effect=input_state,
        )
        celune.force_stop_speech = mock.Mock()
        result: list[bool] = []

        worker = threading.Thread(
            target=lambda: result.append(celune._prepare_voice_change("bold"))
        )
        worker.start()
        assert prepare_started.wait(timeout=1)
        assert worker.is_alive()

        celune.locked = False
        celune.playback_done.set()
        worker.join(timeout=1)

        assert not worker.is_alive()
        assert result == [True]
        celune.force_stop_speech.assert_not_called()
        assert not celune.loaded
        assert not celune.model_ready.is_set()

    def test_voice_change_does_not_wait_for_non_speech_playback(self) -> None:
        """Verify voice changes ignore active non-verbal playback."""
        celune = self._make_celune({})
        celune.voices = ("balanced", "bold")
        celune.loaded = True
        celune.locked = False
        celune.cur_state = "speaking"
        celune.model_ready.set()
        celune.playback_done.clear()
        celune._playback_source_meta[1] = {
            "kind": "sfx",
            "base_gain": 1.0,
            "current_gain": 1.0,
            "total_frames": 48000.0,
            "played_frames": 0.0,
        }

        assert celune._prepare_voice_change("bold")
        assert not celune.loaded
        assert not celune.model_ready.is_set()

    def test_fatal_glow_marks_runtime_error_state(self) -> None:
        """Verify fatal glow always stamps Celune into the error state."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.locked = False
        celune.cur_state = "idle"
        celune._ready_announced = True

        celune.glow.fatal()

        assert celune.cur_state == "error"
        assert not celune.loaded
        assert celune.locked
        assert not celune._ready_announced

    def test_error_signal_does_not_leave_error_state(self) -> None:
        """Verify fatal error signals do not overwrite Celune's error state."""
        celune = self._make_celune({})
        celune.cur_state = "error"
        celune.locked = False
        celune._playback_thread = mock.Mock(is_alive=mock.Mock(return_value=True))

        with mock.patch("celune.celune.play_signal", wraps=play_signal):
            result = celune.try_play_signal("error")

        assert result
        assert celune.cur_state == "error"

    def test_release_pipeline_keeps_error_state_sticky(self) -> None:
        """Verify cleanup does not revive Celune from a fatal error."""
        celune = self._make_celune({})
        celune.cur_state = "error"
        celune.locked = True

        release_pipeline(celune)

        assert celune.cur_state == "error"
        assert not celune.locked

    def test_persona_mode_controls_persona_input_mode(self) -> None:
        """Verify the operation mode controls Persona without legacy switches."""
        from celune.persona.impl import persona_enabled, persona_talkback_enabled

        config: Config = {"mode": "converse", "vram": "high", "persona": {}}
        assert persona_enabled(config)
        assert persona_talkback_enabled(config)
        assert not persona_enabled({"mode": "speak", "vram": "high", "persona": {}})
        assert not persona_talkback_enabled({"vram": "low", "persona": {}})
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert persona_quantization({"vram": "high"}) == "4bit"
            assert persona_quantization({"vram": "xhigh"}) == "8bit"

    def test_low_vram_restricts_heavy_backends_to_mini(self) -> None:
        """Verify low VRAM falls back to the supported mini preset."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(
                config={"vram": "low"},
                tts_backend="voxcpm2",
            )
            self.addCleanup(self._close_celune, celune)

        assert resolve.call_args.args[0] == "mini"

    def test_low_vram_restricts_dotstts_to_mini(self) -> None:
        """Verify low VRAM falls back to mini when dots.tts is requested."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_backend",
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(
                config={"vram": "low"},
                tts_backend="dotstts",
            )
            self.addCleanup(self._close_celune, celune)

        assert resolve.call_args.args[0] == "mini"

    def test_low_vram_rejects_heavy_backend_types(self) -> None:
        """Verify low VRAM rejects explicitly requested heavy backend classes."""

        class HeavyBackend(FakeBackend):
            """A heavy fake backend for usage in tests."""

            name = "voxcpm2"

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            pytest.raises(BackendError, match="not available for VRAM tier 'low'"),
        ):
            Celune(config={"vram": "low"}, tts_backend=HeavyBackend)

    def test_think_reconnects_and_starts_persona_loading_before_speech_fallback(
        self,
    ) -> None:
        """Verify stale Celune instances reconnect and start loading Persona on the next think call.

        Raises:
            AssertionError: Persona reconnect behavior changes unexpectedly.
        """
        celune = self._make_celune({"mode": "converse", "vram": "high"})
        celune.vision = None
        celune.locked = False
        celune.cur_state = "idle"
        client = mock.Mock()
        celune._persona_conn = mock.Mock(return_value=client)
        call_order: list[str] = []

        def record_say(text: str) -> bool:
            call_order.append("say")
            discard(text)
            return False

        def record_start_load() -> None:
            call_order.append("start_load")

        with (
            mock.patch("celune.celune.think_pipeline") as think,
            mock.patch.object(celune, "say", side_effect=record_say) as say,
            mock.patch.object(
                celune,
                "_start_persona_background_load",
                side_effect=record_start_load,
            ) as start_load,
        ):
            assert celune.think("hello")

            persona_thread = celune._persona_thread
            assert persona_thread is not None
            assert persona_thread is not None
            persona_thread.join(timeout=2)
            assert not persona_thread.is_alive()

        assert celune.vision is client
        celune._persona_conn.assert_called_once_with()
        start_load.assert_called_once_with()
        say.assert_called_once_with("hello")
        assert call_order == ["start_load", "say"]
        think.assert_not_called()

    def test_think_queues_requests_while_persona_is_speaking(self) -> None:
        """Verify Persona requests submitted during playback run after the active reply."""
        celune = self._make_celune({"mode": "converse", "vram": "high"})
        celune.error_callback = mock.Mock()
        celune.vision = mock.Mock()
        celune.persona_ready = True
        celune.locked = True
        celune.cur_state = "speaking"
        celune.playback_done.clear()
        calls: list[str] = []

        def process_request(engine: Celune, text: str) -> bool:
            discard(engine)  # LOL, discarding Celune, why?
            calls.append(text)
            return True

        with mock.patch("celune.celune.think_pipeline", side_effect=process_request):
            assert celune.think("first queued")
            assert celune.think("second queued")
            assert celune.error_callback.call_count == 0

            celune.locked = False
            celune.cur_state = "idle"
            celune.playback_done.set()
            persona_thread = celune._persona_thread
            assert persona_thread is not None
            assert persona_thread is not None
            persona_thread.join(timeout=2)

        assert calls == ["first queued", "second queued"]
        assert celune._persona_queue.empty()

    def test_think_worker_keeps_task_like_input_in_converse_mode(self) -> None:
        """Keep a valid task classification on the Persona path in converse mode."""
        celune = self._make_celune({"mode": "converse", "vram": "high"})
        celune.persona_ready = True
        celune.locked = False
        celune.cur_state = "idle"
        celune.playback_done.set()
        response = mock.Mock()
        response.raise_for_status = mock.Mock()
        response.json.return_value = {
            "text": json.dumps(
                {"classification": "task", "route": "task", "confidence": 0.99}
            )
        }
        celune.vision = mock.Mock(post=mock.Mock(return_value=response))
        with (
            mock.patch.object(celune, "_wait_for_persona_playback", return_value=True),
            mock.patch(
                "celune.celune.think_pipeline", return_value=True
            ) as think_pipeline,
            mock.patch(
                "celune.agent.routing.build_agent_classification_request",
                return_value={"format": "test"},
            ),
        ):
            self.assertTrue(celune.think("Delete the fixture."))
            persona_thread = celune._persona_thread
            self.assertIsNotNone(persona_thread)
            assert persona_thread is not None
            persona_thread.join(timeout=2)

        think_pipeline.assert_called_once_with(celune, "Delete the fixture.")
        self.assertIsNone(celune.agent_runtime.get_active_task("default"))

    def test_think_falls_back_to_speech_when_persona_vram_is_incompatible(
        self,
    ) -> None:
        """Use direct speech when the selected preset cannot load Persona."""
        celune = self._make_celune({"mode": "converse", "vram": "medium"})

        with mock.patch.object(celune, "say", return_value=True) as say:
            assert celune.think("Read this aloud.")

        say.assert_called_once_with("Read this aloud.")

    def test_think_in_speak_mode_does_not_invoke_persona_or_agent_routing(self) -> None:
        """Send speak-mode input directly to speech without touching Persona."""
        celune = self._make_celune({"mode": "speak"})
        celune.vision = mock.Mock()
        with mock.patch.object(celune, "say", return_value=True) as say:
            self.assertTrue(celune.think("Speak this exactly."))

        say.assert_called_once_with("Speak this exactly.")
        celune.vision.post.assert_not_called()
        self.assertIsNone(celune.agent_runtime.get_active_task("default"))

    def test_classifier_failure_does_not_fall_through_to_persona_generation(
        self,
    ) -> None:
        """Keep a Persona classifier transport error out of normal generation."""
        celune = self._make_celune({"mode": "agent", "vram": "xhigh"})
        celune.persona_ready = True
        celune.locked = False
        celune.cur_state = "idle"
        celune.playback_done.set()
        celune.vision = mock.Mock()
        celune.vision.post.side_effect = RuntimeError("classifier transport failed")
        with (
            mock.patch("celune.vram.torch.cuda.is_available", return_value=False),
            mock.patch("celune.celune.think_pipeline") as think_pipeline,
            mock.patch.object(celune, "say", return_value=False) as say,
        ):
            self.assertTrue(celune.think("Delete the fixture."))
            persona_thread = celune._persona_thread
            self.assertIsNotNone(persona_thread)
            assert persona_thread is not None
            persona_thread.join(timeout=2)

        think_pipeline.assert_not_called()
        say.assert_called_once_with(i18n.string("agent.classifier_unavailable"))
        self.assertIsNone(celune.agent_runtime.get_active_task("default"))

    def test_think_interrupts_active_agent_and_speech_before_queueing_input(
        self,
    ) -> None:
        """Invalidate active speech and agent work before accepting new input."""
        celune = self._make_celune({"mode": "converse", "vram": "high"})
        celune.locked = True
        celune.cur_state = "speaking"
        celune._persona_thread = mock.Mock()
        celune._persona_thread.is_alive.return_value = True
        task = celune.agent_runtime.create_task(
            AgentRequest(
                request="Check whether this process is running.",
                session=AgentSession(session_id="default"),
            ),
            task_id="active-input-task",
        )
        celune.agent_runtime.start_task(task.task_id)
        celune.agent_runtime.classify_task(task.task_id)
        self.addCleanup(celune.agent_runtime.cancel_task, task.task_id)
        celune.force_stop_speech = mock.Mock()

        self.assertTrue(celune.think("Use the safer status check."))

        celune.force_stop_speech.assert_called_once_with()
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)
        self.assertEqual(
            celune._persona_queue.get_nowait(), "Use the safer status check."
        )

    def test_close_cancels_active_agent_before_runtime_teardown(self) -> None:
        """Verify shutdown records runtime cancellation before closing resources."""
        celune = self._make_celune({})
        task = celune.agent_runtime.create_task(
            AgentRequest(
                request="Check the current status.",
                session=AgentSession(session_id="default"),
            ),
            task_id="shutdown-task",
        )
        celune.agent_runtime.start_task(task.task_id)
        celune.agent_runtime.classify_task(task.task_id)
        backend_abort = mock.Mock()
        celune._playback_thread = mock.Mock(is_alive=mock.Mock(return_value=True))

        with (
            mock.patch("celune.celune.close_pipeline"),
            mock.patch.object(celune, "_unload_persona_state"),
            mock.patch.object(celune, "unload_runtime_state"),
            mock.patch.object(celune.backend, "abort", backend_abort, create=True),
        ):
            celune.close()

        self.assertEqual(task.state, AgentTaskState.CANCELLED)
        self.assertEqual(
            task.cancellation_reason,
            AgentCancellationReason.RUNTIME_SHUTDOWN,
        )
        backend_abort.assert_called_once_with()

    def test_close_aborts_backends_even_when_pipeline_workers_are_cooperative(
        self,
    ) -> None:
        """Verify shutdown aborts backend workers before pipeline cleanup."""
        celune = self._make_celune({})
        backend_abort = mock.Mock()
        celune._playback_thread = mock.Mock(is_alive=mock.Mock(return_value=False))

        with (
            mock.patch("celune.celune.close_pipeline"),
            mock.patch.object(celune, "_unload_persona_state"),
            mock.patch.object(celune, "unload_runtime_state"),
            mock.patch.object(celune.backend, "abort", backend_abort, create=True),
        ):
            celune.close()

        backend_abort.assert_called_once_with()

    def test_close_aborts_a_backend_candidate_stuck_during_reload(self) -> None:
        """Verify shutdown aborts a candidate before it is published as active."""
        celune = self._make_celune({})
        candidate_abort = mock.Mock()
        candidate = mock.Mock(abort=candidate_abort)
        celune._reload_backend = candidate

        with (
            mock.patch("celune.celune.close_pipeline"),
            mock.patch.object(celune, "_unload_persona_state"),
            mock.patch.object(celune, "unload_runtime_state"),
        ):
            celune.close()

        candidate_abort.assert_called_once_with()

    def test_close_aborts_non_cooperative_pipeline_before_teardown(self) -> None:
        """Verify bounded shutdown escalates work that remains active."""
        celune = self._make_celune({})
        backend_abort = mock.Mock()
        celune._playback_thread = mock.Mock(is_alive=mock.Mock(return_value=True))
        events: list[str] = []
        backend_abort.side_effect = lambda: events.append("abort")

        def record_pipeline_close(_engine: Celune) -> None:
            """Record the graceful shutdown boundary for ordering assertions."""
            events.append("graceful")

        with (
            mock.patch(
                "celune.celune.close_pipeline", side_effect=record_pipeline_close
            ),
            mock.patch.object(celune, "_unload_persona_state"),
            mock.patch.object(
                celune,
                "unload_runtime_state",
                side_effect=lambda **_kwargs: events.append("teardown"),
            ),
            mock.patch.object(celune.backend, "abort", backend_abort, create=True),
        ):
            celune.close()

        self.assertEqual(events, ["abort", "graceful", "teardown"])

    def test_close_does_not_wait_for_reload_lock(self) -> None:
        """Verify shutdown continues while a reload still owns its lock."""
        celune = self._make_celune({})
        backend_abort = mock.Mock()
        close_done = threading.Event()
        close_errors: list[BaseException] = []
        celune._async_runtime_lock.acquire()

        def close_engine() -> None:
            """Close the engine from the simulated reload worker boundary."""
            try:
                celune.close()
            except BaseException as error:  # pragma: no cover - assertion aid
                close_errors.append(error)
            finally:
                close_done.set()

        with (
            mock.patch("celune.celune.close_pipeline"),
            mock.patch.object(celune, "_unload_persona_state"),
            mock.patch.object(celune, "unload_runtime_state"),
            mock.patch.object(celune.backend, "abort", backend_abort, create=True),
        ):
            closer = threading.Thread(target=close_engine)
            closer.start()
            try:
                self.assertTrue(close_done.wait(timeout=1))
            finally:
                celune._async_runtime_lock.release()
            closer.join(timeout=1)

        self.assertFalse(close_errors)
        backend_abort.assert_called_once_with()

    def test_unload_persona_state_clears_the_bound_emotion_analyzer(self) -> None:
        """Verify Persona teardown does not retain VLM references through emotion analysis."""
        celune = self._make_celune({})
        celune.vision = mock.Mock()
        analyzer = PersonaEmotionAnalyzer()
        clear_vlm = mock.patch.object(analyzer, "clear_vlm", wraps=analyzer.clear_vlm)
        clear_vlm_mock = clear_vlm.start()
        self.addCleanup(clear_vlm.stop)
        celune.persona_emotion_analyzer = analyzer

        celune._unload_persona_state()

        clear_vlm_mock.assert_called_once_with()
        self.assertIsNone(getattr(celune, "persona_emotion_analyzer"))  # noqa: B009

    def test_reset_persona_conversation_clears_history_summary_and_attachments(
        self,
    ) -> None:
        """Verify a character transition cannot reuse old Persona context."""
        celune = self._make_celune({})
        celune.persona_history = [{"role": "user", "content": "old context"}]
        celune.persona_session_summary = "old summary"
        celune.persona_attachments = [{"path": "old.png", "kind": "image"}]

        celune._reset_persona_conversation()

        assert celune.persona_history == []
        assert celune.persona_session_summary == ""
        assert celune.persona_attachments == []

    def test_setup_extensions_exposes_think_to_extension_context(self) -> None:
        """Verify extension context receives Celune's think entrypoint.

        Raises:
            AssertionError: Extension context wiring changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.think = mock.Mock(return_value=True)
        with mock.patch("celune.celune.CeluneExtensionManager.autoload"):
            celune.setup_extensions()

        assert celune.extension_manager is not None
        assert celune.extension_manager is not None
        think = celune.extension_manager.context.think
        assert think("hello")
        celune.think.assert_called_once_with("hello")

    def test_logging_levels_waiting_and_api_settings_cover_edge_cases(self) -> None:
        """Verify log-level gates, readiness checks, and API fallbacks.

        Raises:
            AssertionError: Core utility behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        celune = self._make_celune(
            {"api": {"port": "bad", "rate_limit_per_minute": "bad"}}
        )
        celune.log_callback = lambda msg, severity="info", *, loglevel="info": (
            logs.append((msg, severity))
        )
        celune.log("hello")
        assert logs[-1] == ("hello", "info")
        celune.log("hidden", loglevel="verbose")
        assert len(logs) == 1
        celune.log_level = "verbose"
        celune.log("visible", loglevel="verbose")
        assert logs[-1] == ("visible", "info")
        celune.log("hidden debug", loglevel="debug")
        assert logs[-1] == ("visible", "info")
        celune.log_level = "debug"
        celune.log("visible debug", loglevel="debug")
        assert logs[-1] == ("visible debug", "info")

        celune.loaded = False
        assert not celune.wait_until_idle(timeout=0)
        celune.loaded = True
        celune.locked = False
        assert celune.wait_until_idle(timeout=0)

        assert celune.api_settings() == (True, "127.0.0.1", 2060, None, 60)
        assert logs[-2][1] == "warning"
        assert logs[-1][1] == "warning"

    def test_submit_audio_is_accepted_and_does_not_use_tts(self) -> None:
        """Verify audio input is accepted without disturbing the text/TTS path."""
        celune = self._make_celune({})
        audio = np.ones((32, 2), dtype=np.float32)

        with (
            mock.patch("celune.celune.handle_audio_input", return_value=True) as handle,
            mock.patch("celune.celune.say_pipeline", return_value=True) as say_pipeline,
        ):
            assert celune.submit_audio(audio, 48000, label="fixture")
            assert celune.say("hello")

        handle.assert_called_once()
        submitted_request = handle.call_args.args[1]
        assert submitted_request.sample_rate == 48000
        assert submitted_request.label == "fixture"
        assert submitted_request.audio.shape == (32, 2)
        assert submitted_request.reset_ready_announcement
        say_pipeline.assert_called_once_with(
            celune,
            "hello",
            save=True,
            display_text=None,
        )

    def test_submit_audio_routes_to_vc_backend_in_voice_conversion_mode(self) -> None:
        """Verify VC mode routes audio input through the configured VC backend."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        audio = np.ones((24, 2), dtype=np.float32)

        with (
            mock.patch(
                "celune.celune.handle_audio_input", wraps=handle_audio_input
            ) as handle,
            mock.patch(
                "celune.pipeline.queue_sfx_audio", return_value=True
            ) as queue_sfx,
            mock.patch("celune.celune.say_pipeline", return_value=True) as say_pipeline,
        ):
            assert celune.submit_audio(audio, 44100, label="fixture")

        handle.assert_called_once()
        queue_sfx.assert_called_once()
        say_pipeline.assert_not_called()

    def test_convert_audio_returns_vc_output_without_queueing_playback(self) -> None:
        """Verify direct conversion returns audio output without touching playback."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        audio = np.ones((16, 2), dtype=np.float32)

        with (
            mock.patch(
                "celune.celune.convert_audio_input", wraps=convert_audio_input
            ) as convert_input,
            mock.patch(
                "celune.pipeline.queue_sfx_audio", return_value=True
            ) as queue_sfx,
        ):
            output = celune.convert_audio(audio, 32000, label="fixture")

        assert output is not None
        assert output.sample_rate == 32000
        assert output.label == "fixture"
        assert output.audio.shape == (16, 2)
        convert_input.assert_called_once()
        submitted_request = convert_input.call_args.args[1]
        assert submitted_request.pitch_shift is None
        queue_sfx.assert_not_called()

    def test_convert_audio_accepts_pitch_shift_override(self) -> None:
        """Verify direct conversion forwards one pitch-shift override."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        audio = np.ones((16, 2), dtype=np.float32)

        with mock.patch(
            "celune.celune.convert_audio_input", return_value=None
        ) as convert_input:
            celune.convert_audio(audio, 32000, label="fixture", pitch_shift=7)

        submitted_request = convert_input.call_args.args[1]
        assert submitted_request.pitch_shift == 7

    def test_convert_audio_accepts_f0_condition_override(self) -> None:
        """Verify direct conversion forwards one f0 conditioning override."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        audio = np.ones((16, 2), dtype=np.float32)

        with mock.patch(
            "celune.celune.convert_audio_input", return_value=None
        ) as convert_input:
            celune.convert_audio(audio, 32000, label="fixture", f0_condition=True)

        submitted_request = convert_input.call_args.args[1]
        assert submitted_request.f0_condition

    def test_constructor_uses_runtime_vc_pitch_shift_default(self) -> None:
        """Verify VC pitch shift starts at its runtime default."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend=FakeVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        assert celune.vc_pitch_shift == 0

    def test_constructor_uses_runtime_vc_f0_default(self) -> None:
        """Verify F0 conditioning starts at its runtime default."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend=FakeVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        assert not celune.vc_f0_condition

    def test_convert_audio_rejects_text_to_speech_mode(self) -> None:
        """Verify direct conversion stays unavailable outside VC mode."""
        celune = self._make_celune({})
        audio = np.ones((16, 2), dtype=np.float32)

        with mock.patch("celune.celune.convert_audio_input") as convert_input:
            output = celune.convert_audio(audio, 48000, label="fixture")

        assert output is None
        convert_input.assert_not_called()

    def test_hot_backend_reload_failure_restores_previous_runtime(self) -> None:
        """Verify failed backend reloads roll Celune back to the previous backend."""

        class FailingBackend(FakeBackend):
            """Backend fixture whose model load always fails."""

            name = "failing"

            def load_model(self, model_id: str, **kwargs: JSONSerializable):
                raise RuntimeError("boom")

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert not celune._hot_reload_backend(FailingBackend, "balanced")

        assert celune.tts_backend == "fake"
        assert celune.current_voice == "balanced"
        assert celune.model_name == "fake/balanced"
        assert celune.loaded
        assert celune.cur_state == "idle"

    def test_failed_reload_closes_aborted_candidate_without_unloading_it(self) -> None:
        """Verify a timed-out remote candidate cannot mask the reload failure during cleanup."""

        class AbortedBackend(FakeBackend):
            """Backend fixture that models a worker already terminated by a timeout."""

            name = "aborted"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def load_model(self, model_id: str, **kwargs: JSONSerializable):
                """Simulate a timeout that has already terminated the worker."""
                raise CEDTSTimeoutError("load_model", 180.0)

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                """Simulate cleanup through a worker already terminated by abort."""
                raise RuntimeError("backend worker is not running")

            def close(self) -> None:
                """Record direct candidate shutdown without touching the worker."""
                self.close_calls += 1

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert not celune._hot_reload_backend(AbortedBackend, "balanced")

        assert celune.tts_backend == "fake"
        assert celune.loaded
        assert celune.model is not None

    def test_busy_backend_reload_restores_readiness_after_preparation(self) -> None:
        """Verify a busy model-loading owner cannot strand a prepared reload."""
        celune = self._make_celune({})
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        celune.voices = ("balanced", "bold")
        celune._last_component_busy = mock.sentinel.stale_busy
        owner = ComponentLockOwner(operation_id="runtime-unload")
        acquisition, lease = celune.component_locks.try_acquire_lease(
            (ComponentLockRequirement(ComponentLockName.MODEL_LOADING),),
            owner,
        )
        self.assertTrue(acquisition.acquired)
        self.assertIsNotNone(lease)
        self.addCleanup(celune.component_locks.release_all)

        with (
            mock.patch.object(celune, "force_stop_speech"),
            mock.patch.object(celune, "_try_play_signal", return_value=False),
        ):
            self.assertTrue(celune._prepare_backend_reload(FakeBackend))

        self.assertFalse(celune._model_ready.is_set())
        self.assertTrue(celune._reload_pending)
        self.assertFalse(celune._hot_reload_backend(FakeBackend))
        self.assertFalse(celune._reload_pending)
        self.assertTrue(celune._model_ready.is_set())
        self.assertIsNone(celune.last_component_busy)
        celune.change_input_state_callback.assert_any_call(locked=False)
        celune.change_voice_lock_state_callback.assert_any_call(locked=False)

    def test_persona_loading_does_not_block_backend_reload(self) -> None:
        """Verify Persona loading does not occupy the TTS model-loading resource."""
        celune = self._make_celune({})
        persona_client = mock.Mock()
        celune.vision = persona_client
        celune._hot_reload_backend_impl = mock.Mock(return_value=True)

        def load_persona(*_args: JSONSerializable) -> None:
            self.assertIsNotNone(
                celune.component_locks.owner_for(ComponentLockName.VLM)
            )
            self.assertIsNone(
                celune.component_locks.owner_for(ComponentLockName.MODEL_LOADING)
            )
            self.assertTrue(celune._hot_reload_backend(FakeBackend, "balanced"))

        persona_client.load.side_effect = load_persona

        celune._load_persona_background(persona_client)

        persona_client.load.assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            "4bit",
        )
        celune._hot_reload_backend_impl.assert_called_once_with(
            FakeBackend,
            "balanced",
        )
        self.assertIsNone(celune.component_locks.owner_for(ComponentLockName.VLM))
        self.assertIsNone(
            celune.component_locks.owner_for(ComponentLockName.MODEL_LOADING)
        )

    def test_hot_backend_reload_failure_reports_restore_status(self) -> None:
        """Verify failed backend reloads announce the rollback phase."""

        class FailingWarmupBackend(FakeBackend):
            """Backend fixture whose warmup generation always fails."""

            name = "failingwarmup"
            voice_models = MappingProxyType({"storm": "warmup/storm"})
            default_voice = "storm"

            def generate_stream(self, model, **kwargs: JSONSerializable):
                discard(model)
                discard(kwargs)
                raise RuntimeError("warmup blew up")

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.status_callback = mock.Mock()

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert not celune._hot_reload_backend(FailingWarmupBackend, "storm")

        status_calls = [call.args[0] for call in celune.status_callback.call_args_list]
        assert "Warming up" in status_calls
        assert "Restoring backend" in status_calls
        assert status_calls[-1] == "Idle"

    def test_hot_backend_reload_unloads_previous_runtime_before_loading_new_one(
        self,
    ) -> None:
        """Verify backend swaps release the old runtime before loading the next one."""

        events: list[str] = []

        class AltFakeBackend(FakeBackend):
            """Alternative backend fixture used by unload-order tests."""

            name = "altfake"
            voice_models = MappingProxyType({"storm": "alt/storm"})
            default_voice = "storm"

            def load_model(self, model_id: str, **kwargs: JSONSerializable):
                events.append(f"load:{model_id}")
                return super().load_model(model_id, **kwargs)

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune._warmup = mock.Mock(return_value=True)
        original_unload = celune.backend.unload_model

        def record_unload() -> None:
            events.append("unload:fake")
            original_unload()

        celune.backend.unload_model = mock.Mock(side_effect=record_unload)

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune._hot_reload_backend(AltFakeBackend, "storm")

        assert events[:2] == ["unload:fake", "load:alt/storm"]

    def test_set_backend_marks_reload_pending_before_playing_working_signal(
        self,
    ) -> None:
        """Verify backend switching marks reload pending before the transition signal."""
        celune = self._make_celune({})
        celune.cur_state = "idle"
        celune.loaded = True
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        signal_states: list[tuple[str, str, bool, bool]] = []

        def record_signal(signal_type: str) -> bool:
            signal_states.append(
                (signal_type, celune.cur_state, celune.loaded, celune._reload_pending)
            )
            return True

        celune._try_play_signal = mock.Mock(side_effect=record_signal)

        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            assert celune.set_backend("mini")

        assert signal_states == [("working", "idle", True, True)]
        celune.change_input_state_callback.assert_called_once_with(locked=True)
        celune.change_voice_lock_state_callback.assert_called_once_with(locked=True)
        assert celune.cur_state == "idle"
        assert celune.loaded
        assert celune._reload_pending
        assert not celune._model_ready.is_set()
        thread_cls.return_value.start.assert_called_once_with()

    def test_set_backend_and_wait_failure_keeps_previous_runtime_loaded(self) -> None:
        """Verify failed backend switches preserve the previously loaded backend runtime."""

        class FailingBackend(FakeBackend):
            """Backend fixture whose model load always fails."""

            name = "failing"

            def load_model(self, model_id: str, **kwargs: JSONSerializable):
                raise RuntimeError("boom")

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")

        with (
            mock.patch(
                "celune.celune.threading.Thread", side_effect=self._immediate_thread
            ),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            assert not celune.set_backend_and_wait(FailingBackend)

        assert celune.tts_backend == "fake"
        assert celune.current_voice == "balanced"
        assert celune.model_name == "fake/balanced"
        assert celune.loaded
        assert celune.model is not None
        assert celune.backend.model is celune.model

    def test_set_backend_and_wait_recovers_when_reload_setup_crashes(self) -> None:
        """Verify backend reload setup failures still release the waiting caller."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")

        with (
            mock.patch(
                "celune.celune.threading.Thread", side_effect=self._immediate_thread
            ),
            mock.patch.object(
                celune,
                "_backend_reload_kwargs",
                side_effect=RuntimeError("setup blew up"),
            ),
        ):
            assert not celune.set_backend_and_wait("mini", timeout=0.1)

        assert celune.cur_state == "idle"
        assert celune.loaded
        assert not celune._reload_pending
        assert celune._model_ready.is_set()
        assert celune.tts_backend == "fake"
        assert celune.current_voice == "balanced"

    def test_set_backend_rejects_reentrant_reload_requests(self) -> None:
        """Verify a second backend switch is refused while reloading is already active."""
        celune = self._make_celune({})
        celune.cur_state = "reloading"
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        celune.log_callback = mock.Mock()

        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            assert not celune.set_backend("mini")

        thread_cls.assert_not_called()
        celune.change_input_state_callback.assert_not_called()
        celune.change_voice_lock_state_callback.assert_not_called()
        celune.log_callback.assert_called_once_with(
            "A backend or character reload is already in progress.",
            "warning",
        )

    def test_set_backend_and_wait_uses_unbounded_wait_by_default(self) -> None:
        """Verify direct backend switches wait indefinitely unless a timeout is supplied."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.set_backend = mock.Mock(return_value=True)
        celune._model_ready.wait = mock.Mock(return_value=True)
        celune._active_runtime_backend_name = mock.Mock(return_value="mini")

        assert celune.set_backend_and_wait("mini")

        celune.set_backend.assert_called_once_with("mini")
        celune._model_ready.wait.assert_called_once_with(timeout=None)
        celune._active_runtime_backend_name.assert_called_once_with()

    def test_set_backend_and_wait_can_switch_between_tts_and_vc_backends(self) -> None:
        """Verify backend hot reloads can move across the TTS and VC backend families."""

        class CountingBackend(FakeBackend):
            """Fake TTS backend that records unload requests."""

            name = "mini"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.unload_calls = 0

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                self.unload_calls += 1
                super().unload_model(release_cuda_cache=release_cuda_cache)

        class CountingVCBackend(FakeVCBackend):
            """Fake VC backend that records preload and unload requests."""

            name = "counting-vc"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.preload_calls = 0
                self.unload_calls = 0

            def preload_models(self) -> None:
                self.preload_calls += 1

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                self.unload_calls += 1

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.threading.Thread", side_effect=self._immediate_thread
            ),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            celune = Celune(config={}, tts_backend=CountingBackend)
            self.addCleanup(self._close_celune, celune)

            initial_backend = cast(CountingBackend, celune.backend)
            celune.loaded = True
            celune.model = {"model_id": "counting/balanced", "kwargs": {}}
            celune.backend.model = celune.model
            celune.model_name = "counting/balanced"
            celune.current_voice = "balanced"
            celune.voices = ("balanced", "bold")

            assert celune.set_backend_and_wait(CountingVCBackend)

            switched_vc_backend = cast(CountingVCBackend, celune.vc_backend)
            assert initial_backend.unload_calls == 1
            assert switched_vc_backend.preload_calls == 1
            assert celune.input_mode == "voice_conversion"
            assert celune._active_runtime_backend_name() == "counting-vc"
            assert celune.model is None
            assert celune.model_name == ""

            assert celune.set_backend_and_wait(FakeBackend)

            assert switched_vc_backend.unload_calls == 1
            assert celune.input_mode == "text_to_speech"
            assert celune.vc_backend is None
            assert celune._active_runtime_backend_name() == "fake"
            assert celune.tts_backend == "fake"
            assert celune.model is not None

    def test_set_backend_and_wait_restores_vc_runtime_after_failed_tts_switch(
        self,
    ) -> None:
        """Verify failed VC-to-TTS switches rebuild the previous VC runtime."""

        class CountingVCBackend(FakeVCBackend):
            """Fake VC backend that records lifecycle operations."""

            name = "counting-vc"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.preload_calls = 0
                self.unload_calls = 0

            def preload_models(self) -> None:
                self.preload_calls += 1

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                self.unload_calls += 1

        class FailingBackend(FakeBackend):
            """Backend fixture whose model load always fails."""

            name = "failing"

            def load_model(self, model_id: str, **kwargs: JSONSerializable):
                raise RuntimeError("boom")

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.threading.Thread", side_effect=self._immediate_thread
            ),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            celune = Celune(config={}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

            assert celune.set_backend_and_wait(CountingVCBackend)
            previous_vc_backend = cast(CountingVCBackend, celune.vc_backend)

            assert not celune.set_backend_and_wait(FailingBackend)

            assert previous_vc_backend.unload_calls == 1
            assert celune.input_mode == "voice_conversion"
            assert celune.vc_backend is not None
            assert isinstance(celune.vc_backend, CountingVCBackend)
            assert celune.vc_backend is not previous_vc_backend
            assert celune._active_runtime_backend_name() == "counting-vc"
            assert celune.voice_conversion_backend == "counting-vc"

    def test_set_backend_rejects_unknown_backend_before_reload_side_effects(
        self,
    ) -> None:
        """Verify unknown backend names do not trigger reload state or working signals."""
        celune = self._make_celune({})
        celune.cur_state = "idle"
        celune.loaded = True
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        celune.log_callback = mock.Mock()
        celune._try_play_signal = mock.Mock()

        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            assert not celune.set_backend("qwen")

        thread_cls.assert_not_called()
        celune.change_input_state_callback.assert_not_called()
        celune.change_voice_lock_state_callback.assert_not_called()
        celune._try_play_signal.assert_not_called()
        celune.log_callback.assert_called_once_with(
            "unknown backend: qwen (available: mini, qwen3, dotstts, voxcpm2, gpt-sovits, seed-vc)",
            "warning",
        )
        assert celune.cur_state == "idle"
        assert celune.loaded
        assert not celune._reload_pending

    def test_set_backend_unknown_name_keeps_previous_runtime_live(self) -> None:
        """Verify invalid backend names leave the previously loaded backend fully usable."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"

        assert not celune.set_backend_and_wait("qwen")

        assert celune.backend.name == "fake"
        assert celune.tts_backend == "fake"
        assert celune.model_name == "fake/balanced"
        assert celune.current_voice == "balanced"
        assert celune.loaded
        assert celune.backend.model is celune.model
        assert not celune._reload_pending

    def test_set_cevoice_rejects_reentrant_reload_requests(self) -> None:
        """Verify a second character switch is refused while reloading is already active."""
        celune = self._make_celune({})
        celune.cur_state = "reloading"
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        celune.log_callback = mock.Mock()

        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            assert not celune.set_cevoice("nova")

        thread_cls.assert_not_called()
        celune.change_input_state_callback.assert_not_called()
        celune.change_voice_lock_state_callback.assert_not_called()
        celune.log_callback.assert_called_once_with(
            "A backend or character reload is already in progress.",
            "warning",
        )

    def test_set_cevoice_and_wait_uses_unbounded_wait_by_default(self) -> None:
        """Verify direct character switches wait indefinitely unless a timeout is supplied."""
        celune = self._make_celune({})
        target_bundle = Path("celune.cevoice")
        celune.loaded = True
        celune.set_cevoice = mock.Mock(return_value=True)
        celune._model_ready.wait = mock.Mock(return_value=True)

        with (
            mock.patch("celune.celune.active_bundle_path", return_value=target_bundle),
            mock.patch("celune.celune.resolve_bundle_path", return_value=target_bundle),
        ):
            assert celune.set_cevoice_and_wait(target_bundle)

        celune.set_cevoice.assert_called_once_with(target_bundle)
        celune._model_ready.wait.assert_called_once_with(timeout=None)

    def test_set_cevoice_rejects_missing_bundle_before_reload_side_effects(
        self,
    ) -> None:
        """Verify missing CEVOICE bundles do not start a reload or lock the UI."""
        celune = self._make_celune({})
        celune.change_input_state_callback = mock.Mock()
        celune.change_voice_lock_state_callback = mock.Mock()
        celune.log_callback = mock.Mock()

        with mock.patch("celune.celune.threading.Thread") as thread_cls:
            assert not celune.set_cevoice("invalid_character")

        thread_cls.assert_not_called()
        celune.change_input_state_callback.assert_not_called()
        celune.change_voice_lock_state_callback.assert_not_called()
        celune.log_callback.assert_called_once_with(
            "Voice pack not found: invalid_character",
            "warning",
        )
        assert not celune._reload_pending

    def test_hot_backend_reload_warmup_does_not_publish_candidate_backend_early(
        self,
    ) -> None:
        """Verify candidate backend warmup runs before the live backend pointer is swapped."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        observed_backend_names: list[str] = []

        class FailingWarmupBackend(FakeBackend):
            """Backend fixture whose warmup generation fails after observing live state."""

            name = "failingwarmup"
            voice_models = MappingProxyType({"storm": "alt/storm"})
            default_voice = "storm"

            def generate_stream(self, model, **kwargs: JSONSerializable):
                observed_backend_names.append(celune.backend.name)
                discard(model)
                discard(kwargs)
                raise RuntimeError("warmup blew up")

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert not celune._hot_reload_backend(FailingWarmupBackend, "storm")

        assert observed_backend_names == ["fake"]
        assert celune.backend.name == "fake"
        assert celune.loaded

    def test_with_backend_temporarily_switches_and_restores_backend(self) -> None:
        """Verify the backend context manager restores the original backend after use."""

        class AltFakeBackend(FakeBackend):
            """Alternative backend fixture used by context-manager tests."""

            name = "altfake"
            voice_models = MappingProxyType({"storm": "alt/storm", "calm": "alt/calm"})
            default_voice = "storm"

        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.locked = False
        celune.model_ready.set()
        celune.playback_done.set()
        celune._warmup = mock.Mock(return_value=True)

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch("celune.celune.play_signal", return_value=False)
            )
            with celune.with_backend(AltFakeBackend):
                assert celune.tts_backend == "altfake"
                assert celune.current_voice == "storm"
                assert celune.model_name == "alt/storm"

        assert celune.tts_backend == "fake"
        assert celune.current_voice == "balanced"
        assert celune.model_name == "fake/balanced"

    def test_with_backend_does_not_pin_old_backend_instance_during_override(
        self,
    ) -> None:
        """Verify temporary backend overrides do not keep the old backend instance alive."""

        class AltFakeBackend(FakeBackend):
            """Alternative backend fixture used by backend lifetime tests."""

            name = "altfake"
            voice_models = MappingProxyType({"storm": "alt/storm"})
            default_voice = "storm"

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            original_backend = FakeBackend()
            backend_ref = weakref.ref(original_backend)
            celune = Celune(config={}, tts_backend=original_backend)
            self.addCleanup(self._close_celune, celune)

        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.locked = False
        celune.model_ready.set()
        celune.playback_done.set()
        celune._warmup = mock.Mock(return_value=True)
        del original_backend

        with (
            mock.patch("celune.celune.play_signal", return_value=False),
            celune.with_backend(AltFakeBackend),
        ):
            import gc as _gc

            _gc.collect()
            assert backend_ref() is None

    def test_hot_cevoice_reload_failure_restores_previous_bundle_state(self) -> None:
        """Verify failed CEVOICE reloads restore the previous bundle and voice state."""
        celune = self._make_celune({})
        celune.log_level = "verbose"
        celune.log_callback = mock.Mock()
        celune.backend.uses_voice_bundles = True
        celune.backend.validate_refs = mock.Mock()
        celune.backend.model_id_for_voice = mock.Mock(
            side_effect=lambda voice: f"fake/{voice}"
        )
        celune.backend.load_model = mock.Mock(
            side_effect=lambda model_id: {"model_id": model_id}
        )
        celune._warmup = mock.Mock(return_value=True)
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced"}
        celune.backend.model = celune.model
        celune.model_name = "fake/balanced"
        celune.current_voice = "balanced"
        celune.current_character = "Celune"
        celune.voices = ("balanced", "bold")

        selected = {"path": Path("celune.cevoice")}
        first_bundle = SimpleNamespace(
            path=Path("celune.cevoice"),
            voice_order=("balanced", "bold"),
            metadata={"name": "Celune", "default_voice": "balanced"},
        )

        def fake_select(bundle=None):
            selected["path"] = (
                Path(bundle) if bundle is not None else Path("celune.cevoice")
            )
            return selected["path"]

        def fake_loader():
            if selected["path"].name == "celune.cevoice":
                return SimpleNamespace(bundle=first_bundle)
            if selected["path"].name == "broken.cevoice":
                return SimpleNamespace(
                    bundle=SimpleNamespace(
                        path=Path("broken.cevoice"),
                        voice_order=(),
                        metadata={"name": "Broken"},
                    )
                )
            return None

        with (
            mock.patch("celune.celune.select_voice_bundle", side_effect=fake_select),
            mock.patch("celune.celune.default_loader", side_effect=fake_loader),
            mock.patch(
                "celune.celune.bundle_matches_default_pack_checksum",
                side_effect=lambda path: Path(path).name == "celune.cevoice",
            ),
            mock.patch(
                "celune.celune.active_bundle_path", side_effect=lambda: selected["path"]
            ),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            assert not celune._hot_reload_cevoice(Path("broken.cevoice"))

        assert selected["path"] == Path("celune.cevoice")
        assert celune.current_character == "Celune"
        assert celune.current_voice == "balanced"
        assert celune.model_name == "fake/balanced"
        assert celune.loaded
        assert any(
            "no voices found" in str(call.args[0])
            for call in celune.log_callback.call_args_list
            if call.args
        )

    def test_set_cevoice_and_wait_recovers_when_reload_setup_crashes(self) -> None:
        """Verify CEVOICE reload setup failures still release the waiting caller."""
        celune = self._make_celune({})
        celune.loaded = True
        celune.cur_state = "idle"
        celune.current_voice = "balanced"

        with (
            mock.patch(
                "celune.celune.threading.Thread", side_effect=self._immediate_thread
            ),
            mock.patch(
                "celune.celune.default_loader", side_effect=RuntimeError("boom")
            ),
        ):
            assert not celune.set_cevoice_and_wait(Path("broken.cevoice"), timeout=0.1)

        assert celune.cur_state == "idle"
        assert celune.loaded
        assert not celune._reload_pending
        assert celune._model_ready.is_set()
        assert celune.current_voice == "balanced"

    def test_voice_conversion_mode_rejects_text_input(self) -> None:
        """Verify VC mode rejects text input instead of using the TTS backend."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"

        with mock.patch(
            "celune.celune.say_pipeline", return_value=True
        ) as say_pipeline:
            assert not celune.say("hello")

        say_pipeline.assert_not_called()

    def test_constructor_defaults_to_seedvc_vc_backend_in_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC mode resolves the default Seed-VC backend cleanly."""
        fake_vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        fake_vc_backend.name = "seed-vc"
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_vc_backend",
                return_value=fake_vc_backend,
            ) as resolve_vc_backend,
        ):
            celune = Celune(
                config={"mode": "voice_conversion"}, tts_backend=FakeBackend
            )
            self.addCleanup(self._close_celune, celune)

        resolve_vc_backend.assert_called_once_with("seed-vc", log=celune.log_callback)
        assert celune.input_mode == "voice_conversion"
        assert celune.voice_conversion_backend == "seed-vc"

    def test_constructor_rejects_unknown_vc_backend_cleanly(self) -> None:
        """Verify unsupported VC backends surface a readable backend error."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            pytest.raises(BackendError, match="unknown voice-conversion backend"),
        ):
            Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend="missing",
            )

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
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            self.assertEqual(celune.load(), True)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(getattr(celune.glow, "entered"), True)  # noqa: B009
        celune.close()

        failing = self._make_celune({})
        failing.setup_extensions = mock.Mock()
        failing.backend.preload_models = mock.Mock()
        failing.backend.load_default_model = mock.Mock(side_effect=RuntimeError("boom"))
        errors: list[str] = []
        failing.error_callback = errors.append
        with mock.patch("celune.celune.play_signal", return_value=False):
            self.assertEqual(failing.load(), False)
        self.assertEqual(errors, ["Default model failed to load"])
        self.assertEqual(getattr(failing.glow, "fatal_called"), True)  # noqa: B009

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
        assert celune.model is None
        assert celune.llm is None
        assert celune.tokenizer is None
        assert celune.backend.model is None

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
        assert celune.model is None
        assert celune.backend.model is None
        assert celune.llm is None
        assert celune.tokenizer is None

    def test_sleep_transition_defers_cuda_cache_release(self) -> None:
        """Verify automatic sleep unloads references without flushing the CUDA allocator."""
        celune = self._make_celune(
            {
                "sleep": {
                    "enabled": True,
                    "unload": {"persona": False, "normalizer": True, "tts": True},
                }
            }
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.current_voice = "balanced"
        celune.voices = ("balanced",)
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.llm = cast(PreTrainedModel, mock.Mock(spec=PreTrainedModel))
        celune.tokenizer = cast(
            PreTrainedTokenizerBase,
            mock.Mock(spec=PreTrainedTokenizerBase),
        )

        with (
            mock.patch("celune.celune.play_signal", return_value=False),
            mock.patch("celune.celune.torch.cuda.is_available", return_value=True),
            mock.patch("celune.celune.torch.cuda.synchronize") as synchronize,
            mock.patch("celune.celune.torch.cuda.empty_cache") as empty_cache,
        ):
            self.assertTrue(celune.enter_sleep_mode())

        synchronize.assert_not_called()
        empty_cache.assert_not_called()
        self.assertTrue(celune.sleeping)
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
            assert release.wait(timeout=2)
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
            assert ready.wait(timeout=2)
            celune.unload_normalizer_state()
            release.set()
            assert finished.wait(timeout=2)

        assert celune.llm is None
        assert celune.tokenizer is None

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
        persona_load_started = threading.Event()
        release_persona_load = threading.Event()

        def load_persona(*_args, **_kwargs) -> None:
            persona_load_started.set()
            release_persona_load.wait(timeout=2)

        persona_client.load.side_effect = load_persona

        with mock.patch("celune.celune.play_signal", return_value=False):
            self.assertEqual(celune.enter_sleep_mode(), True)

        self.assertEqual(celune.sleeping, True)
        self.assertEqual(celune.loaded, False)
        self.assertEqual(celune.cur_state, "sleeping")
        self.assertEqual(getattr(celune.glow, "sleep_called"), True)  # noqa: B009
        self.assertIsNone(celune.model)
        self.assertEqual(celune.model_name, "")
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)
        self.assertIsNone(celune.vision)
        persona_client.close.assert_called_once_with()

        wake_result: list[bool] = []
        wake_thread = threading.Thread(
            target=lambda: wake_result.append(celune.wake_from_sleep())
        )
        wake_thread.start()
        wake_thread.join(timeout=1)
        assert not wake_thread.is_alive()
        assert wake_result == [True]
        assert persona_load_started.wait(timeout=2)
        release_persona_load.set()
        background_thread = celune._wake_background_thread
        if background_thread is not None:
            background_thread.join(timeout=2)

        self.assertIsNot(celune.backend, old_backend)
        self.assertEqual(celune.sleeping, False)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(celune.cur_state, "idle")
        self.assertEqual(getattr(celune.glow, "wake_called"), True)  # noqa: B009
        self.assertEqual(celune.model, {"model_id": "fake/balanced", "kwargs": {}})
        self.assertEqual(celune.model_name, "fake/balanced")
        celune._warmup.assert_called_once_with()
        celune.load_normalizer.assert_called_once_with()
        persona_client.load.assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            "4bit",
        )

    def test_sleep_mode_can_unload_vc_without_unloading_tts(self) -> None:
        """Verify sleep mode can explicitly unload and reload the active VC backend."""

        class CountingVCBackend(FakeVCBackend):
            """Fake VC backend that records sleep lifecycle calls."""

            name = "counting-vc"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.preload_calls = 0
                self.unload_calls = 0

            def preload_models(self) -> None:
                self.preload_calls += 1

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                self.unload_calls += 1

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config={
                    "mode": "voice_conversion",
                    "sleep": {
                        "enabled": True,
                        "unload": {
                            "persona": False,
                            "normalizer": False,
                            "tts": False,
                            "vc": True,
                        },
                    },
                },
                tts_backend=FakeBackend,
                vc_backend=CountingVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.backend.unload_model = mock.Mock()
        original_vc_backend = cast(CountingVCBackend, celune.vc_backend)

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune.enter_sleep_mode()

        assert original_vc_backend.unload_calls == 1
        celune.backend.unload_model.assert_not_called()

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune.wake_from_sleep()

        restored_vc_backend = cast(CountingVCBackend, celune.vc_backend)
        assert restored_vc_backend is not original_vc_backend
        assert restored_vc_backend.preload_calls == 1

    def test_sleep_tts_unload_can_keep_vc_loaded_when_explicitly_disabled(
        self,
    ) -> None:
        """Verify ``sleep.unload.vc`` can opt out of the legacy TTS-coupled VC unload."""

        class CountingVCBackend(FakeVCBackend):
            """Fake VC backend that records unload requests."""

            name = "counting-vc"

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.unload_calls = 0

            def unload_model(self, release_cuda_cache: bool = True) -> None:
                self.unload_calls += 1

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(
                config={
                    "sleep": {
                        "enabled": True,
                        "unload": {
                            "persona": False,
                            "normalizer": False,
                            "tts": True,
                            "vc": False,
                        },
                    },
                },
                tts_backend=FakeBackend,
                vc_backend=CountingVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        vc_backend = cast(CountingVCBackend, celune.vc_backend)

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune.enter_sleep_mode()

        assert celune.vc_backend is vc_backend
        assert vc_backend.unload_calls == 0

    def test_sleep_mode_closes_persona_even_if_close_raises(self) -> None:
        """Verify sleep still clears Persona references when client shutdown fails."""
        celune = self._make_celune(
            {"sleep": {"enabled": True, "unload": {"persona": True, "tts": False}}}
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune.vision = mock.Mock(close=mock.Mock(side_effect=RuntimeError("boom")))

        with mock.patch("celune.celune.play_signal", return_value=False):
            assert celune.enter_sleep_mode()
        assert celune.vision is None

    def test_sleep_mode_plays_signal_after_releasing_pipeline_lock(self) -> None:
        """Verify the sleeping signal is not invoked while ``say_lock`` is still held."""
        celune = self._make_celune(
            {"sleep": {"enabled": True, "unload": {"persona": False, "tts": False}}}
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"

        def play_sleep_signal(engine: Celune, signal_type: str) -> bool:
            assert signal_type == "sleeping"
            assert engine.say_lock.acquire(blocking=False)
            engine.say_lock.release()
            return False

        with mock.patch("celune.celune.play_signal", side_effect=play_sleep_signal):
            assert celune.enter_sleep_mode()

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
        failing_backend = FakeBackend()
        failing_backend.load_model = mock.Mock(side_effect=RuntimeError("boom"))

        with (
            mock.patch("celune.celune.resolve_backend", return_value=failing_backend),
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            self.assertEqual(celune.wake_from_sleep(), False)
        self.assertEqual(getattr(celune.glow, "fatal_called"), True)  # noqa: B009

    def test_concurrent_wake_requests_only_recreate_backend_once(self) -> None:
        """Verify repeated wake requests cannot duplicate backend recreation."""
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
        celune.model = None
        celune._warmup = mock.Mock(return_value=True)

        load_started = threading.Event()
        release_load = threading.Event()
        recreated_backend = FakeBackend()

        def blocking_load_model(model_id: str) -> dict[str, JSONSerializable]:
            load_started.set()
            assert release_load.wait(timeout=1)
            return {"model_id": model_id, "kwargs": {}}

        recreated_backend.load_model = mock.Mock(side_effect=blocking_load_model)

        with (
            mock.patch(
                "celune.celune.resolve_backend", return_value=recreated_backend
            ) as resolve_backend,
            mock.patch("celune.celune.play_signal", return_value=False),
        ):
            results: list[bool] = []

            def wake() -> None:
                results.append(celune.wake_from_sleep())

            first = threading.Thread(target=wake)
            second = threading.Thread(target=wake)
            first.start()
            assert load_started.wait(timeout=1)
            second.start()
            release_load.set()
            first.join(timeout=1)
            second.join(timeout=1)

        assert results == [True, True]
        assert resolve_backend.call_count == 1
        recreated_backend.load_model.assert_called_once_with("fake/balanced")
        assert not celune.sleeping
        assert celune.cur_state == "idle"

    def test_raise_warmup_error_preserves_original_cause(self) -> None:
        """Verify WarmupError keeps the underlying warmup failure as its cause."""
        celune = self._make_celune({})
        cause = RuntimeError("tensor mismatch")
        celune._last_warmup_error = cause

        with pytest.raises(WarmupError) as exc_info:
            celune.raise_warmup_error("warmup failed after sleep")

        assert exc_info.value.__cause__ is cause

    def test_nonfatal_warmup_failure_does_not_enter_fatal_error_state(self) -> None:
        """Verify rollback-scoped warmup failures stay non-fatal."""
        celune = self._make_celune({})
        celune.cur_state = "reloading"
        celune.loaded = True
        celune.model = {"model_id": "fake/balanced", "kwargs": {}}
        celune.backend.model = celune.model
        celune.error_callback = mock.Mock()
        celune.backend.generate_stream = mock.Mock(
            side_effect=RuntimeError("warmup blew up")
        )

        result = celune._warmup(fatal_on_failure=False)

        self.assertEqual(result, False)
        self.assertEqual(celune.cur_state, "reloading")
        self.assertEqual(celune.loaded, True)
        self.assertEqual(getattr(celune.glow, "fatal_called"), False)  # noqa: B009
        celune.error_callback.assert_not_called()


@pytest.mark.anyio
class TestCeluneAsyncRuntime(CeluneAsyncTestCase):
    """Tests for async Celune runtime entry points."""

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

    async def test_set_backend_async_waits_for_model_ready_via_to_thread(self) -> None:
        """Verify async backend switching runs preparation and reload through to_thread."""
        celune = self._make_celune({})
        celune.loaded = True
        celune._prepare_backend_reload = mock.Mock(return_value=True)
        celune.current_voice = "nova"
        celune._hot_reload_backend = mock.Mock(return_value=True)
        celune._active_runtime_backend_name = mock.Mock(return_value="mini")
        to_thread = mock.AsyncMock(side_effect=lambda func, *args: func(*args))

        with mock.patch("celune.celune.asyncio.to_thread", to_thread):
            switched = await celune.set_backend_async("mini", timeout=12.0)

        assert switched
        celune._prepare_backend_reload.assert_called_once_with("mini")
        celune._hot_reload_backend.assert_called_once_with("mini", "nova")
        assert to_thread.await_count == 3

    def test_set_backend_stops_active_speech_before_starting_reload(self) -> None:
        """Verify backend reload requests invalidate active speech before reloading."""
        celune = self._make_celune({})
        order: list[str] = []
        celune.force_stop_speech = mock.Mock(side_effect=lambda: order.append("stop"))
        celune._hot_reload_backend = mock.Mock(
            side_effect=lambda *_args: order.append("reload")
        )

        with mock.patch(
            "celune.celune.threading.Thread",
            side_effect=TestCeluneCore._immediate_thread,
        ):
            started = celune.set_backend("mini")

        assert started
        assert order == ["stop", "reload"]

    async def test_wake_from_sleep_async_uses_to_thread(self) -> None:
        """Verify waking from sleep moves the blocking reload path off the event loop."""
        celune = self._make_celune({})
        celune.wake_from_sleep = mock.Mock(return_value=True)
        to_thread = mock.AsyncMock(side_effect=lambda func, *args: func(*args))

        with mock.patch("celune.celune.asyncio.to_thread", to_thread):
            woke = await celune.wake_from_sleep_async()

        assert woke
        celune.wake_from_sleep.assert_called_once_with()
        assert to_thread.await_count == 2
