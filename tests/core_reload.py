# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune core behavior without real models or GPU work."""

import weakref
import threading
import contextlib
from types import SimpleNamespace, MappingProxyType
from typing import Optional, cast
from pathlib import Path
from unittest import mock

import pytest
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from celune.utils import discard
from celune.celune import Celune
from celune.exceptions import WarmupError, BackendError
from celune.typing.common import JSONSerializable

from .support import (
    FakeGlow,
    FakeBackend,
    FakeVCBackend,
)
from .core_runtime import TestCeluneCore as _TestCeluneCore


class TestCeluneCore(_TestCeluneCore):
    """Exercise backend reload behavior of the Celune core."""

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

        with mock.patch("celune.voice.play_signal", return_value=False):
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            "unknown backend: qwen (available: mini, qwen3, fireredtts3, dotstts, voxcpm2, gpt-sovits, seed-vc)",
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
            mock.patch("celune.voice.active_bundle_path", return_value=target_bundle),
            mock.patch("celune.voice.resolve_bundle_path", return_value=target_bundle),
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

        with mock.patch("celune.voice.play_signal", return_value=False):
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
                mock.patch("celune.voice.play_signal", return_value=False)
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            "celune.speech.queue_speech", return_value=True
        ) as say_pipeline:
            assert not celune.say("hello")

        say_pipeline.assert_not_called()

    def test_constructor_defaults_to_seedvc_vc_backend_in_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC mode resolves the default Seed-VC backend cleanly."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_vc_backend",
                return_value=FakeVCBackend(
                    log=lambda _msg, _severity="info": None,
                ),
            ) as resolve_vc,
        ):
            celune = Celune(
                config={"mode": "voice_conversion"}, tts_backend=FakeBackend
            )
            self.addCleanup(self._close_celune, celune)

        resolve_vc.assert_called_once()
        assert resolve_vc.call_args.args[0] == "seed-vc"
        assert celune.input_mode == "voice_conversion"
        assert celune.vc_backend is not None
        assert celune.vc_backend.name == "fake-vc"

    def test_constructor_rejects_unknown_vc_backend_cleanly(self) -> None:
        """Verify unsupported VC backends surface a readable backend error."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch(
                "celune.celune.resolve_vc_backend",
                side_effect=ValueError("unknown voice-conversion backend: 'missing'"),
            ) as resolve_vc,
            pytest.raises(BackendError, match="unknown voice-conversion backend"),
        ):
            Celune(
                config={"mode": "voice_conversion"},
                tts_backend=FakeBackend,
                vc_backend="missing",
            )

        resolve_vc.assert_called_once()

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
            mock.patch("celune.loader.validate_runtime", return_value=True),
            mock.patch("celune.voice.play_signal", return_value=False),
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
        with mock.patch("celune.voice.play_signal", return_value=False):
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
            mock.patch("celune.voice.play_signal", return_value=False),
            mock.patch("celune.sleep.torch.cuda.is_available", return_value=True),
            mock.patch("celune.sleep.torch.cuda.synchronize") as synchronize,
            mock.patch("celune.sleep.torch.cuda.empty_cache") as empty_cache,
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
        progress_events: list[tuple[Optional[float], Optional[float]]] = []
        celune.progress_callback = lambda progress, total: progress_events.append(
            (progress, total)
        )
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
                "celune.loader.load_normalizer_components",
                side_effect=fake_load_components,
            ) as load_components,
            mock.patch("celune.loader.torch.cuda.is_available", return_value=False),
        ):
            celune.load_normalizer()
            assert ready.wait(timeout=2)
            celune.unload_normalizer_state()
            release.set()
            assert finished.wait(timeout=2)

        assert celune.llm is None
        assert celune.tokenizer is None
        load_components.assert_called_once()
        assert load_components.call_args.kwargs["progress_callback"] is None
        assert not progress_events

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

        with mock.patch("celune.voice.play_signal", return_value=False):
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

        with mock.patch("celune.voice.play_signal", return_value=False):
            assert celune.enter_sleep_mode()

        assert original_vc_backend.unload_calls == 1
        celune.backend.unload_model.assert_not_called()

        with mock.patch("celune.voice.play_signal", return_value=False):
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

        with mock.patch("celune.voice.play_signal", return_value=False):
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

        with mock.patch("celune.voice.play_signal", return_value=False):
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

        with mock.patch("celune.voice.play_signal", side_effect=play_sleep_signal):
            assert celune.enter_sleep_mode()

    def test_sleep_mode_waits_for_active_sfx_playback(self) -> None:
        """Verify sleep does not unload runtime state while an SFX source drains."""
        celune = self._make_celune(
            {"sleep": {"enabled": True, "unload": {"persona": False, "tts": False}}}
        )
        celune.locked = False
        celune.loaded = True
        celune.cur_state = "idle"
        celune._playback_source_meta[1] = {"kind": "sfx"}

        with mock.patch("celune.voice.play_signal") as play_signal_mock:
            assert not celune.enter_sleep_mode()

        play_signal_mock.assert_not_called()
        assert not celune.sleeping

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
            mock.patch("celune.voice.play_signal", return_value=False),
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
            mock.patch("celune.voice.play_signal", return_value=False),
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
