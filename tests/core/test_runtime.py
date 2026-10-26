# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune core behavior without real models or GPU work."""

import json
import tempfile
import threading
from types import MappingProxyType
from typing import Optional, cast
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from celune import i18n, cevoice
from celune.api import CloudflaredQuickTunnel
from celune.core import voice as core_voice
from celune.core import reload as core_reload
from celune.core import loading as core_loading
from celune.core import lifecycle as core_lifecycle
from celune.core import conversation as core_conversation
from celune.agent import (
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentCancellationReason,
)
from celune.utils import discard
from celune.celune import Celune
from celune.config import Config
from celune.exceptions import BackendError, CEDTSTimeoutError
from celune.persona.impl import persona_quantization
from celune.typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from celune.typing.modes import OperationState
from celune.typing.common import JSONSerializable
from celune.persona.emotion import PersonaEmotionAnalyzer
from celune.pipeline.playback import play_signal
from celune.dataclasses.events import StateChangedEvent
from celune.pipeline.buffering import release_pipeline
from celune.pipeline.submission import handle_audio_input, convert_audio_input

from ..support import FakeGlow, FakeBackend, FakeVCBackend
from .fixtures import CeluneCoreTestSupport


class TestCeluneRuntime(CeluneCoreTestSupport):
    """Tests for Celune orchestration without real model work."""

    _cached_celune: Optional[Celune] = None
    _cached_instance_keys: frozenset[str] = frozenset()

    def test_constructor_validates_backend_and_chunk_size(self) -> None:
        """Verify constructor validation and derived chunk size behavior.

        Raises:
            AssertionError: Constructor behavior changes unexpectedly.
        """
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(config={"vram": "low"}, tts_backend=None)
            self.addCleanup(self._close_celune, celune)

        resolve.assert_called_once()
        assert resolve.call_args.args[0] == "mini"
        assert "clone_model_id" not in resolve.call_args.kwargs
        assert celune.operation_state is OperationState.NORMAL
        celune.close()
        self.assertEqual(celune.operation_state, OperationState.STOPPED)

        celune = self._make_celune({})
        self.assertEqual(celune.chunk_size, 8)
        self.assertEqual(getattr(celune.glow, "started"), True)  # noqa: B009
        celune.close()

        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            pytest.raises(BackendError, match="invalid chunk length"),
        ):
            Celune(
                config={},
                tts_backend=FakeBackend,
                target_chunk_length=0.65,
            )

    def test_fatal_operation_state_distinguishes_configuration_failures(self) -> None:
        """Retain settings access only for failures attributed to configuration."""
        celune = self._make_celune({})

        with mock.patch("celune.core.voice.try_play_signal", return_value=False):
            celune.fatal(semi_fatal=True)
            self.assertEqual(celune.operation_state, OperationState.SEMI_FATAL)

            celune.fatal()
            self.assertEqual(celune.operation_state, OperationState.FATAL)

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
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
        ):
            celune = Celune(config={}, backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

        assert celune.backend.name == "fake"
        assert celune.tts_backend == "fake"

    def test_tts_cache_quantization_setting_reaches_initial_and_reloaded_backends(
        self,
    ) -> None:
        """Pass the shared cache policy into cache-backed TTS workers."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(
                config={"vram": "medium", "quantize_kv_cache": False},
                tts_backend="qwen3",
            )
            self.addCleanup(self._close_celune, celune)

        assert resolve.call_args.kwargs["quantize_kv_cache"] is False
        assert (
            core_reload._backend_reload_kwargs(celune, "voxcpm2")["quantize_kv_cache"]
            is False
        )

    def test_soulxsinger_backend_options_reach_initial_and_reloaded_backends(
        self,
    ) -> None:
        """Forward namespaced SoulX options during initial load and reload."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.lifecycle.resolve_backend_name",
                return_value="soulxsinger",
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
                return_value=FakeBackend(),
            ) as resolve,
        ):
            celune = Celune(
                config={
                    "backend_options": {
                        "soulxsinger": {"high_precision": True},
                    },
                },
                tts_backend="soulxsinger",
            )
            self.addCleanup(self._close_celune, celune)

        assert resolve.call_args.kwargs["high_precision"] is True
        assert (
            core_reload._backend_reload_kwargs(celune, "soulxsinger")["high_precision"]
            is True
        )
        assert "high_precision" not in core_reload._backend_reload_kwargs(
            celune, "mini"
        )

    def test_constructor_accepts_backend_alias_for_vc_runtime(self) -> None:
        """Verify ``backend=`` can configure the VC runtime in VC mode."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
                return_value=FakeBackend(),
            ),
        ):
            celune = Celune(config={"mode": "voice_conversion"}, backend=FakeVCBackend)
            self.addCleanup(self._close_celune, celune)

        assert celune.input_mode == "voice_conversion"
        assert celune.vc_backend is not None
        assert celune.vc_backend.name == "fake-vc"

    def test_constructor_accepts_backend_alias_string_for_vc_runtime(self) -> None:
        """Verify string backend aliases resolve to VC backends when selected."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
                return_value=FakeBackend(),
            ),
            mock.patch(
                "celune.core.reload.resolve_vc_backend",
                return_value=FakeVCBackend(
                    log=lambda _msg, _severity="info": None,
                ),
            ) as resolve_vc,
        ):
            celune = Celune(config={"mode": "voice_conversion"}, backend="seed-vc")
            self.addCleanup(self._close_celune, celune)

        resolve_vc.assert_called_once()
        assert resolve_vc.call_args.args[1] == "seed-vc"
        assert celune.vc_backend is not None
        assert celune.vc_backend.name == "fake-vc"

    def test_constructor_uses_explicit_locale_override_from_config(self) -> None:
        """Verify an explicit config locale wins over system auto-detection."""
        previous_locale = i18n.get_locale()
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch("celune.core.lifecycle.get_system_locale", return_value="pl"),
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
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch("celune.core.lifecycle.get_system_locale", return_value="pl"),
        ):
            celune = Celune(config={"locale": None}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

        self.addCleanup(i18n.set_locale, previous_locale)
        assert i18n.get_locale() == "pl"

    def test_constructor_rejects_duplicate_backend_alias_for_tts(self) -> None:
        """Verify ``backend=`` cannot be combined with ``tts_backend=``."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            pytest.raises(
                BackendError, match="cannot specify both 'backend' and 'tts_backend'"
            ),
        ):
            Celune(config={}, backend=FakeBackend, tts_backend=FakeBackend)

    def test_constructor_rejects_duplicate_backend_alias_for_vc(self) -> None:
        """Verify ``backend=`` cannot be combined with ``vc_backend=``."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
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
                "celune.core.loading.saved_output_speech_seconds",
                return_value=42.5,
            ),
            mock.patch("celune.core.voice.play_signal", return_value=False),
        ):
            assert celune.load(skip_runtime_check=True)

        assert celune.historical_generated_speech_seconds == 42.5

    def test_load_reports_core_initialization_startup_checkpoint(self) -> None:
        """Verify ``load`` reports its dedicated startup checkpoint once."""
        startup_callback = mock.Mock()
        celune = self._make_celune({}, startup_callback=startup_callback)

        with mock.patch("celune.core.voice.play_signal", return_value=False):
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
            mock.patch("celune.core.voice.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.core.voice.bundle_matches_default_pack_checksum",
                return_value=False,
            ),
        ):
            assert celune.load_voice_bundle(Path("fixture.cevoice"))
        assert celune.current_voice == "bold"
        assert celune.current_character == "Fixture"
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

            core_lifecycle._cleanup_residual_temp_data(celune, temp_root)

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
                core_lifecycle._cleanup_residual_temp_data(celune, temp_root)
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
            mock.patch("celune.core.voice.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.core.voice.bundle_matches_default_pack_checksum",
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
            mock.patch("celune.core.voice.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.core.voice.bundle_matches_default_pack_checksum",
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
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=True
            ) as available,
            mock.patch(
                "celune.core.lifecycle.create_persona_client", return_value=client
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
        assert persona.persona_model_id() == "Qwen/Qwen3-VL-4B-Instruct"
        client.close()

    def test_load_starts_persona_after_tts_is_ready(self) -> None:
        """Verify TTS becomes ready before Persona starts downloading."""
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        self.patch_owner(core_loading, "warmup", return_value=True)
        self.patch_owner(core_loading, "_start_configured_api")
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
            mock.patch("celune.core.loading.threading.Thread") as thread_cls,
            mock.patch("celune.core.loading.validate_runtime", return_value=True),
            mock.patch("celune.core.voice.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        progress_events_before_persona = progress_events.copy()
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
        assert progress_events == progress_events_before_persona

    def test_load_defers_temp_cleanup_until_shutdown(self) -> None:
        """Verify temp cleanup waits until runtime shutdown after initialization."""
        celune = self._make_celune({})
        celune.backend.uses_voice_bundles = True
        call_order: list[str] = []

        def cleanup(_engine: Celune, _temp_dir: Path) -> None:
            call_order.append("cleanup")

        def close_loader() -> None:
            call_order.append("loader")

        def load_voices() -> bool:
            call_order.append("voices")
            return False

        self.patch_owner(core_voice, "try_play_signal", return_value=False)
        celune.error_callback = mock.Mock()

        with (
            mock.patch.object(
                core_lifecycle, "_cleanup_residual_temp_data", side_effect=cleanup
            ),
            mock.patch(
                "celune.core.lifecycle.close_default_loader", side_effect=close_loader
            ),
            mock.patch.object(celune, "load_available_voices", side_effect=load_voices),
            mock.patch("celune.core.loading.log_runtime_banner"),
        ):
            assert not celune.load()
            assert call_order == ["voices"]
            celune.close()

        assert call_order == ["voices", "loader", "cleanup"]

    def test_load_disables_persona_when_preload_fails(self) -> None:
        """Verify Persona preload failures fall back to speech-only mode."""
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        self.patch_owner(core_loading, "warmup", return_value=True)
        self.patch_owner(core_loading, "_start_configured_api")
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        persona_client = mock.Mock()
        persona_client.load.side_effect = RuntimeError("persona boom")
        celune.vision = persona_client
        with (
            mock.patch("celune.core.loading.threading.Thread") as thread_cls,
            mock.patch("celune.core.loading.validate_runtime", return_value=True),
            mock.patch("celune.core.voice.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        core_lifecycle.load_persona_background(celune, persona_client)
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
        self.patch_owner(
            core_conversation, "wait_for_persona_playback", return_value=True
        )
        say = self.patch_owner(core_conversation, "say", return_value=True)
        celune._persona_queue.put("hello while downloading")

        core_conversation._think_worker(celune)

        say.assert_called_once_with(celune, "hello while downloading")
        celune.vision.post.assert_not_called()

    def test_load_voice_conversion_mode_skips_tts_model_load_and_warmup(self) -> None:
        """Verify VC mode does not boot the TTS runtime during startup."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
        ):
            celune = Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend=FakeVCBackend,
            )
            self.addCleanup(self._close_celune, celune)

        celune.setup_extensions = mock.Mock()
        warmup = self.patch_owner(core_loading, "warmup", return_value=True)
        self.patch_owner(core_loading, "_start_configured_api")
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "unused"})
        assert celune.vc_backend is not None
        celune.vc_backend.preload_models = mock.Mock()

        with (
            mock.patch("celune.core.loading.threading.Thread") as thread_cls,
            mock.patch("celune.core.loading.validate_runtime", return_value=True),
            mock.patch("celune.core.voice.play_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            assert celune.load()

        celune.backend.preload_models.assert_not_called()
        celune.backend.load_default_model.assert_not_called()
        warmup.assert_not_called()
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
        readiness_states: list[tuple[bool, str]] = []
        celune.voice_changed_callback = lambda _name: readiness_states.append(
            (celune.model_ready.is_set(), celune.cur_state)
        )

        with mock.patch("celune.core.voice.play_signal", return_value=False):
            celune.change_voice("bold")

        assert celune.current_voice == "bold"
        assert celune.loaded
        assert celune.cur_state == "idle"
        assert statuses[-1] == ("Idle", "info")
        assert readiness_states == [(True, "idle")]

    def test_status_and_sleep_delegate_to_runtime_callbacks(self) -> None:
        """Verify extension helpers forward status and sleep to Celune's runtime."""
        celune = self._make_celune({})

        with (
            mock.patch.object(celune._callbacks, "status_callback") as status,
            mock.patch("celune.core.voice.time.sleep") as sleep,
        ):
            celune.status("Status test", "warning")
            celune.sleep(1.25)

        status.assert_called_once_with("Status test", "warning")
        sleep.assert_called_once_with(1.25)

    def test_closing_core_rejects_late_extension_requests(self) -> None:
        """Verify extension calls cannot enqueue new work after shutdown starts."""
        celune = self._make_celune({})
        celune._exit_requested = True

        with (
            mock.patch("celune.pipeline.submission.queue_speech") as say,
            mock.patch("celune.pipeline.submission.play") as play,
            mock.patch("celune.core.voice.play_signal") as signal,
        ):
            assert not celune.say("late speech")
            assert not celune.play("late.wav")
            assert not celune.try_play_signal("readiness")

        say.assert_not_called()
        play.assert_not_called()
        signal.assert_not_called()

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
        warmup = self.patch_owner(core_loading, "warmup", return_value=True)
        statuses: list[tuple[str, str]] = []
        celune.status_callback = lambda msg, severity="info", *, loglevel="info": (
            statuses.append((msg, severity))
        )
        celune.voice_changed_callback = mock.Mock()

        with mock.patch("celune.core.voice.play_signal", return_value=False):
            celune.change_voice("bold")

        assert celune.current_voice == "bold"
        assert celune.loaded
        assert celune.cur_state == "idle"
        assert statuses[-1] == ("Idle", "info")
        celune.backend.model_id_for_voice.assert_not_called()
        celune.backend.load_model.assert_not_called()
        warmup.assert_not_called()
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
            target=lambda: result.append(
                core_voice._prepare_voice_change(celune, "bold")
            )
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
        celune.playback_source_meta[1] = {
            "kind": "sfx",
            "base_gain": 1.0,
            "current_gain": 1.0,
            "total_frames": 48000.0,
            "played_frames": 0.0,
        }

        assert core_voice._prepare_voice_change(celune, "bold")
        assert not celune.loaded
        assert not celune.model_ready.is_set()

    def test_singing_backend_rejects_speaking_voice_changes(self) -> None:
        """Keep ordinary voice changes away from singing backends."""
        celune = self._make_celune({})
        celune.backend.singing = True
        celune.loaded = True
        celune.model_ready.set()
        celune.change_input_state_callback = mock.Mock()

        assert not core_voice._prepare_voice_change(celune, "bold")
        assert celune.loaded
        assert celune.model_ready.is_set()
        celune.change_input_state_callback.assert_not_called()

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

        with mock.patch("celune.core.voice.play_signal", wraps=play_signal):
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

    def test_playback_state_events_dispatch_after_releasing_say_lock(self) -> None:
        """Verify completion and overlay callbacks run outside the pipeline lock."""
        celune = self._make_celune({})
        lock_availability: list[tuple[str, bool]] = []
        queued_markers: list[bool] = []

        def check_lock(event: StateChangedEvent) -> None:
            acquired = celune.say_lock.acquire(blocking=False)
            lock_availability.append((event.new_state, acquired))
            if acquired:
                celune.say_lock.release()

        celune.cur_state = "generating"
        celune.locked = True
        celune._event_dispatcher.subscribe("state_changed", check_lock)
        release_pipeline(celune)

        celune.locked = False

        def queue_completion(*_args, **_kwargs) -> bool:
            queued_markers.append(True)
            return True

        with (
            mock.patch(
                "celune.pipeline.playback.readiness_signal",
                return_value=np.zeros((8, 2), dtype=np.float32),
            ),
            mock.patch("celune.pipeline.captions.register_playback_source"),
            mock.patch("celune.pipeline.buffering.queue_playback_chunk"),
            mock.patch(
                "celune.pipeline.playback.queue_playback_done",
                side_effect=queue_completion,
            ),
        ):
            assert play_signal(celune, "readiness")

        assert lock_availability == [("idle", True), ("speaking", True)]
        assert queued_markers == [True]
        celune._event_dispatcher.unsubscribe("state_changed", check_lock)
        celune.playback_done.set()
        celune.cur_state = "idle"

    def test_persona_mode_controls_persona_input_mode(self) -> None:
        """Verify the operation mode controls Persona without legacy switches."""
        from celune.persona.impl import persona_enabled, persona_talkback_enabled

        config: Config = {"mode": "converse", "vram": "high", "persona": {}}
        assert persona_enabled(config)
        assert persona_talkback_enabled(config)
        assert not persona_enabled(
            {"mode": "performance", "vram": "high", "persona": {}}
        )
        assert not persona_talkback_enabled({"vram": "low", "persona": {}})
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert persona_quantization({"vram": "high"}) == "4bit"
            assert persona_quantization({"vram": "xhigh"}) == "8bit"

    def test_low_vram_restricts_heavy_backends_to_mini(self) -> None:
        """Verify low VRAM falls back to the supported mini preset."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
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
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_backend",
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
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
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
        persona_conn = self.patch_owner(
            core_lifecycle, "persona_conn", return_value=client
        )
        call_order: list[str] = []

        def record_say(_engine: Celune, text: str) -> bool:
            call_order.append("say")
            discard(text)
            return False

        def record_start_load(_engine: Celune) -> None:
            call_order.append("start_load")

        with (
            mock.patch("celune.core.conversation.think_persona") as think,
            mock.patch.object(core_conversation, "say", side_effect=record_say) as say,
            mock.patch(
                "celune.core.lifecycle.start_persona_background_load",
                side_effect=record_start_load,
            ) as start_load,
        ):
            assert celune.think("hello")

            persona_thread = celune._persona_thread
            assert persona_thread is not None
            persona_thread.join(timeout=2)
            assert not persona_thread.is_alive()

        assert celune.vision is client
        persona_conn.assert_called_once_with(celune)
        start_load.assert_called_once_with(celune)
        say.assert_called_once_with(celune, "hello")
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

        with mock.patch(
            "celune.core.conversation.think_persona", side_effect=process_request
        ):
            assert celune.think("first queued")
            assert celune.think("second queued")
            assert celune.error_callback.call_count == 0

            celune.locked = False
            celune.cur_state = "idle"
            celune.playback_done.set()
            persona_thread = celune._persona_thread
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
            mock.patch(
                "celune.core.conversation.wait_for_persona_playback",
                return_value=True,
            ),
            mock.patch(
                "celune.core.conversation.think_persona", return_value=True
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

        with mock.patch.object(core_conversation, "say", return_value=True) as say:
            assert celune.think("Read this aloud.")

        say.assert_called_once_with(celune, "Read this aloud.")

    def test_think_in_performance_mode_does_not_invoke_persona_or_agent_routing(
        self,
    ) -> None:
        """Send performance-mode input directly to speech without touching Persona."""
        celune = self._make_celune({"mode": "performance"})
        celune.vision = mock.Mock()
        with mock.patch.object(core_conversation, "say", return_value=True) as say:
            self.assertTrue(celune.think("Speak this exactly."))

        say.assert_called_once_with(celune, "Speak this exactly.")
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
            mock.patch("celune.core.conversation.think_persona") as think_pipeline,
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

    def test_think_interrupts_active_agent_without_stopping_speech(
        self,
    ) -> None:
        """Preserve active speech while interrupting agent work for new input."""
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

        celune.force_stop_speech.assert_not_called()
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)
        self.assertEqual(
            celune._persona_queue.get_nowait(), "Use the safer status check."
        )

    def test_say_does_not_stop_active_speech(self) -> None:
        """Verify a new synchronous speech request does not cancel active playback."""
        celune = self._make_celune({})
        self.patch_owner(core_voice, "speech_playback_active", return_value=True)
        celune.force_stop_speech = mock.Mock()

        with mock.patch(
            "celune.pipeline.submission.queue_speech", return_value=True
        ) as queue_speech:
            self.assertTrue(celune.say("next utterance"))

        celune.force_stop_speech.assert_not_called()
        queue_speech.assert_called_once_with(
            celune,
            "next utterance",
            save=True,
            stream_queue=None,
            display_text=None,
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
        tunnel = mock.Mock()
        celune._api_tunnel = cast(CloudflaredQuickTunnel, tunnel)

        with (
            mock.patch("celune.core.lifecycle._close_speech_pipeline"),
            mock.patch("celune.core.lifecycle.unload_persona_state"),
            mock.patch("celune.core.lifecycle.unload_runtime_state"),
            mock.patch.object(celune.backend, "abort", backend_abort, create=True),
        ):
            celune.close()

        self.assertEqual(task.state, AgentTaskState.CANCELLED)
        self.assertEqual(
            task.cancellation_reason,
            AgentCancellationReason.RUNTIME_SHUTDOWN,
        )
        backend_abort.assert_called_once_with()
        tunnel.stop.assert_called_once_with()

    def test_close_aborts_backends_even_when_pipeline_workers_are_cooperative(
        self,
    ) -> None:
        """Verify shutdown aborts backend workers before pipeline cleanup."""
        celune = self._make_celune({})
        backend_abort = mock.Mock()
        celune._playback_thread = mock.Mock(is_alive=mock.Mock(return_value=False))

        with (
            mock.patch("celune.core.lifecycle._close_speech_pipeline"),
            mock.patch("celune.core.lifecycle.unload_persona_state"),
            mock.patch("celune.core.lifecycle.unload_runtime_state"),
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
            mock.patch("celune.core.lifecycle._close_speech_pipeline"),
            mock.patch("celune.core.lifecycle.unload_persona_state"),
            mock.patch("celune.core.lifecycle.unload_runtime_state"),
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
                "celune.core.lifecycle._close_speech_pipeline",
                side_effect=record_pipeline_close,
            ),
            mock.patch("celune.core.lifecycle.unload_persona_state"),
            mock.patch(
                "celune.core.lifecycle.unload_runtime_state",
                side_effect=lambda _engine, **_kwargs: events.append("teardown"),
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
            mock.patch("celune.core.lifecycle._close_speech_pipeline"),
            mock.patch("celune.core.lifecycle.unload_persona_state"),
            mock.patch("celune.core.lifecycle.unload_runtime_state"),
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

        core_lifecycle.unload_persona_state(celune)

        clear_vlm_mock.assert_called_once_with()
        self.assertIsNone(celune.persona_emotion_analyzer)

    def test_reset_persona_conversation_clears_history_summary_and_attachments(
        self,
    ) -> None:
        """Verify a character transition cannot reuse old Persona context."""
        celune = self._make_celune({})
        celune.persona_history = [{"role": "user", "content": "old context"}]
        celune.persona_session_summary = "old summary"
        celune.persona_attachments = [{"path": "old.png", "kind": "image"}]

        celune.reset_persona_conversation()

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
        with mock.patch("celune.core.voice.CeluneExtensionManager.autoload"):
            celune.setup_extensions()

        assert celune.extension_manager is not None
        think = celune.extension_manager.context.think
        assert think("hello")
        celune.think.assert_called_once_with("hello")

    def test_logging_levels_waiting_and_api_settings_cover_edge_cases(self) -> None:
        """Verify log-level gates, readiness checks, and API fallbacks.

        Raises:
            AssertionError: Core utility behavior changes unexpectedly.
        """
        logs: list[tuple[str, str, str]] = []
        celune = self._make_celune(
            {"api": {"port": "bad", "rate_limit_per_minute": "bad"}}
        )
        celune.log_callback = lambda msg, severity="info", *, loglevel="info": (
            logs.append((msg, severity, loglevel))
        )
        celune.log("hello")
        assert logs[-1] == ("hello", "info", "info")
        celune.log("hidden", loglevel="verbose")
        assert len(logs) == 1
        celune.log_level = "verbose"
        celune.log("visible", loglevel="verbose")
        assert logs[-1] == ("visible", "info", "verbose")
        celune.log("hidden debug", loglevel="debug")
        assert logs[-1] == ("visible", "info", "verbose")
        celune.log_level = "debug"
        celune.log("visible debug", loglevel="debug")
        assert logs[-1] == ("visible debug", "info", "debug")

        celune.loaded = False
        assert not celune.wait_until_idle(timeout=0)
        celune.loaded = True
        celune.locked = False
        assert celune.wait_until_idle(timeout=0)

        with tempfile.TemporaryDirectory() as temp_directory:
            key_path = Path(temp_directory) / "api.key"
            with (
                mock.patch("celune.core.lifecycle.api_key_path", return_value=key_path),
                mock.patch.dict("os.environ", {"CELUNE_API_TOKEN": ""}),
            ):
                settings = celune.api_settings()
                assert settings[:3] == (True, "127.0.0.1", 2060)
                assert settings[4:] == (60, "same_machine")
                assert settings[3] is not None and settings[3].startswith("C-")
                assert len(settings[3]) == 27
                assert key_path.read_text(encoding="utf-8").strip() == settings[3]

        assert logs[-3][1] == "info"
        assert str(key_path) in logs[-3][0]
        assert logs[-2][1] == "warning"
        assert logs[-1][1] == "warning"

    def test_api_access_modes_map_to_local_and_network_listeners(self) -> None:
        """Verify each configured access level chooses its intended local bind."""
        expected_hosts = {
            "same_machine": "127.0.0.1",
            "same_network": "0.0.0.0",
            "everyone": "127.0.0.1",
        }
        for access_level, host in expected_hosts.items():
            celune = self._make_celune(
                {"api": {"access": access_level, "token": "configured"}}
            )
            try:
                settings = celune.api_settings()
                assert settings[1] == host
                assert settings[5] == access_level
            finally:
                celune.close()

    def test_submit_audio_is_accepted_and_does_not_use_tts(self) -> None:
        """Verify audio input is accepted without disturbing the text/TTS path."""
        celune = self._make_celune({})
        audio = np.ones((32, 2), dtype=np.float32)

        with (
            mock.patch(
                "celune.pipeline.submission.handle_audio_input", return_value=True
            ) as handle,
            mock.patch(
                "celune.pipeline.submission.queue_speech", return_value=True
            ) as say_pipeline,
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
            stream_queue=None,
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
                "celune.pipeline.submission.handle_audio_input",
                wraps=handle_audio_input,
            ) as handle,
            mock.patch(
                "celune.pipeline.submission.queue_sfx_audio", return_value=True
            ) as queue_sfx,
            mock.patch(
                "celune.pipeline.submission.queue_speech", return_value=True
            ) as say_pipeline,
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
                "celune.pipeline.submission.convert_audio_input",
                wraps=convert_audio_input,
            ) as convert_input,
            mock.patch(
                "celune.pipeline.submission.queue_sfx_audio", return_value=True
            ) as queue_sfx,
        ):
            output = celune.convert_audio(audio, 32000, label="fixture")

        assert output is not None
        assert output.sample_rate == 32000
        assert output.label == "fixture"
        assert output.audio.shape == (16, 2)
        convert_input.assert_called_once()
        queue_sfx.assert_not_called()

    def test_convert_audio_accepts_f0_condition_override(self) -> None:
        """Verify direct conversion forwards one f0 conditioning override."""
        celune = self._make_celune({})
        celune.input_mode = "voice_conversion"
        celune.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        audio = np.ones((16, 2), dtype=np.float32)

        with mock.patch(
            "celune.pipeline.submission.convert_audio_input", return_value=None
        ) as convert_input:
            celune.convert_audio(audio, 32000, label="fixture", f0_condition=True)

        submitted_request = convert_input.call_args.args[1]
        assert submitted_request.f0_condition

    def test_constructor_uses_runtime_vc_f0_default(self) -> None:
        """Verify F0 conditioning starts at its runtime default."""
        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.reload.resolve_vc_backend",
                return_value=FakeVCBackend(
                    log=lambda _msg, _severity="info": None,
                ),
            ) as resolve_vc,
        ):
            celune = Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
            )
            self.addCleanup(self._close_celune, celune)

        resolve_vc.assert_called_once()
        assert resolve_vc.call_args.args[1] == "seed-vc"
        assert not celune.vc_f0_condition

    def test_convert_audio_rejects_text_to_speech_mode(self) -> None:
        """Verify direct conversion stays unavailable outside VC mode."""
        celune = self._make_celune({})
        audio = np.ones((16, 2), dtype=np.float32)

        with mock.patch(
            "celune.pipeline.submission.convert_audio_input"
        ) as convert_input:
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

        with mock.patch("celune.core.voice.play_signal", return_value=False):
            assert not core_reload.reload_backend(celune, FailingBackend, "balanced")

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

        with mock.patch("celune.core.voice.play_signal", return_value=False):
            assert not core_reload.reload_backend(celune, AbortedBackend, "balanced")

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
            mock.patch.object(core_voice, "try_play_signal", return_value=False),
        ):
            self.assertTrue(core_voice._prepare_backend_reload(celune, FakeBackend))

        self.assertFalse(celune._model_ready.is_set())
        self.assertTrue(celune._reload_pending)
        self.assertFalse(core_reload.reload_backend(celune, FakeBackend))
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
        hot_reload_impl = self.patch_owner(
            core_reload, "_hot_reload_backend_impl", return_value=True
        )

        def load_persona(*_args: JSONSerializable) -> None:
            self.assertIsNotNone(
                celune.component_locks.owner_for(ComponentLockName.VLM)
            )
            self.assertIsNone(
                celune.component_locks.owner_for(ComponentLockName.MODEL_LOADING)
            )
            self.assertTrue(core_reload.reload_backend(celune, FakeBackend, "balanced"))

        persona_client.load.side_effect = load_persona

        core_lifecycle.load_persona_background(celune, persona_client)

        persona_client.load.assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            "4bit",
        )
        hot_reload_impl.assert_called_once_with(
            celune,
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
        celune.error_callback = mock.Mock()

        with mock.patch("celune.core.voice.play_signal", return_value=False):
            assert not core_reload.reload_backend(celune, FailingWarmupBackend, "storm")

        status_calls = [call.args[0] for call in celune.status_callback.call_args_list]
        assert "Warming up" in status_calls
        assert "Restoring backend" in status_calls
        assert status_calls[-1] == "Idle"
        celune.error_callback.assert_called_once_with(
            i18n.string("status.could_not_reload", app_name="Celune")
        )
