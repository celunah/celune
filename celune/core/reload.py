# SPDX-License-Identifier: Apache-2.0

"""Celune model and character reload transactions."""

from __future__ import annotations
import time
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Optional, cast

from transformers.modeling_utils import PreTrainedModel

from . import lifecycle
from .lifecycle import (
    TTS_KV_CACHE_BACKENDS,
    _core_backend_target,
    tts_quantization_enabled,
)

from ..constants import APP_NAME
from ..i18n import string, tagged_string
from ..typing.backends import BackendModel
from ..typing.common import JSONSerializable
from ..backends.vc import (
    CeluneVCBackend,
)
from ..utils import available, format_error_message
from ..backends.tts import CeluneBackend, resolve_backend
from ..exceptions import WarmupError, BackendError, NotAvailableError
from ..config import config_bool, config_value, config_backend_options
from ..backends.vc import (
    resolve_vc_backend as create_vc_backend,
)
from ..vram import (
    VramPreset,
    resolve_vram_preset,
    is_cuda_out_of_memory,
    release_cuda_after_oom,
)
from ..typing.celune import (
    VCBackendSpec,
    TTSBackendSpec,
    CoreBackendSpec,
    MessageCallback,
    ReleasableObject,
)
from ..dataclasses.events import (
    VoiceChangedEvent,
    CharacterLoadedEvent,
    CharacterChangedEvent,
    CharacterUnloadedEvent,
)
from ..cevoice import (
    CEVoicePersona,
    default_loader,
    active_bundle_path,
    select_voice_bundle,
    bundle_character_name,
    persona_metadata_from_manifest,
    bundle_matches_default_pack_checksum,
)


@dataclass
class ReloadSnapshot:
    """Rollback state captured before a backend or CEVOICE hot reload."""

    backend: CeluneBackend
    restorable_backend_spec: Union[str, type[CeluneBackend]]
    backend_spec: Optional[Union[str, type[CeluneBackend]]]
    backend_kwargs: dict[str, JSONSerializable]
    vc_backend: Optional[CeluneVCBackend]
    restorable_vc_backend_spec: Optional[Union[str, type[CeluneVCBackend]]]
    vc_backend_spec: Optional[Union[str, type[CeluneVCBackend]]]
    voice_conversion_backend: str
    tts_backend: str
    input_mode: str
    model: Optional[BackendModel]
    model_name: str
    voices: tuple[str, ...]
    current_voice: Optional[str]
    current_character: Optional[str]
    current_character_persona: Optional[CEVoicePersona]
    voice_bundle_is_default: bool
    loaded: bool
    cur_state: str


def release_loaded_object(value: ReleasableObject) -> None:
    """Best-effort release hook for one loaded runtime object."""
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()
        return

    unload = getattr(value, "unload", None)
    if callable(unload):
        with contextlib.suppress(Exception):
            unload()


def unload_backend_model(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool,
) -> None:
    """Unload one backend while retaining compatibility with older backend plugins."""
    try:
        backend.unload_model(release_cuda_cache=release_cuda_cache)
    except TypeError as error:
        if "release_cuda_cache" not in str(error):
            raise
        backend.unload_model()


def dispose_backend(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool = True,
) -> None:
    """Unload one backend and close its worker process when supported."""
    try:
        unload_backend_model(backend, release_cuda_cache)
    finally:
        close_backend(backend)


def close_backend(backend: Union[CeluneBackend, CeluneVCBackend]) -> None:
    """Close one backend process without repeating model unloading."""
    close = getattr(backend, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def shutdown_backend(
    backend: Union[CeluneBackend, CeluneVCBackend],
    release_cuda_cache: bool,
) -> None:
    """Release one backend without waiting on an already-aborted operation."""
    if callable(getattr(backend, "close", None)):
        close_backend(backend)
        return
    unload_backend_model(backend, release_cuda_cache)


def validate_backend_against_preset(
    backend: CeluneBackend,
    preset: VramPreset,
) -> None:
    """Reject backend instances that bypass preset-specific runtime limits."""
    if (
        backend.name == "qwen3"
        and getattr(backend, "clone_model_id", None) != preset.qwen3_clone_model_id
    ):
        raise BackendError(
            f"backend '{backend.name}' is not available with model "
            f"'{getattr(backend, 'clone_model_id', None)}' for VRAM tier '{preset.tier}'"
        )


def resolve_tts_backend(
    self,
    backend_spec: TTSBackendSpec,
    **backend_kwargs,
) -> CeluneBackend:
    """Resolve a TTS backend and bind runtime callbacks."""
    backend = resolve_backend(backend_spec, **backend_kwargs)
    backend.bind_progress(self.progress_callback)
    backend.log_level = getattr(self, "log_level", "info")
    return backend


def resolve_vc_backend(
    self,
    backend_spec: VCBackendSpec,
    log: Optional[MessageCallback] = None,
) -> CeluneVCBackend:
    """Resolve a voice-conversion backend and bind runtime callbacks."""
    backend = create_vc_backend(backend_spec, log=log)
    backend.bind_progress(self.progress_callback)
    backend.log_level = getattr(self, "log_level", "info")
    return backend


def recreate_tts_backend(self) -> bool:
    """Rebuild the TTS backend from its original constructor recipe."""
    if self._backend_spec is None:
        return False
    candidate_backend = resolve_tts_backend(
        self,
        self._backend_spec,
        log=self.log_callback,
        fatal=self.fatal,
        **self._backend_kwargs,
    )
    candidate_backend.bind_fatal(self.fatal)
    close_backend(self.backend)
    self.backend = candidate_backend
    self.tts_backend = candidate_backend.name
    return True


def recreate_vc_backend(self) -> bool:
    """Rebuild the VC backend from its original constructor recipe."""
    if self._vc_backend_spec is None:
        return False
    candidate_backend = resolve_vc_backend(
        self,
        self._vc_backend_spec,
        log=self.log_callback,
    )
    if available("f0_condition", obj=candidate_backend):
        candidate_backend.f0_condition = self.vc_f0_condition
    previous_backend = self.vc_backend
    if previous_backend is not None:
        close_backend(previous_backend)
    self.vc_backend = candidate_backend
    self.voice_conversion_backend = candidate_backend.name
    return True


def restorable_vc_backend_spec(
    self,
) -> Optional[Union[str, type[CeluneVCBackend]]]:
    """Return a VC backend recipe that does not pin the active instance."""
    if self.vc_backend is None:
        return None
    if self._vc_backend_spec is not None:
        return self._vc_backend_spec
    return type(self.vc_backend)


def restorable_active_backend_spec(
    self,
) -> Union[type[Union[CeluneBackend, CeluneVCBackend]], str]:
    """Return a restorable recipe for the active TTS or VC backend."""
    if lifecycle.is_voice_conversion_mode(self) and self.vc_backend is not None:
        restorable_vc_backend = restorable_vc_backend_spec(self)
        if restorable_vc_backend is not None:
            return restorable_vc_backend
    return _restorable_backend_spec(self)


def emit_character_event_transition(
    self,
    old_character: Optional[str],
    old_bundle_path: Optional[str],
    new_character: Optional[str],
    new_bundle_path: Optional[str],
    new_is_default: bool,
) -> None:
    """Emit the correct character event for one CEVOICE transition."""
    if old_character and new_character:
        if old_character == new_character and old_bundle_path == new_bundle_path:
            return
        self.emit_event(
            "character_changed",
            CharacterChangedEvent(
                celune=self,
                old_character=old_character,
                new_character=new_character,
                old_bundle_path=old_bundle_path,
                new_bundle_path=new_bundle_path,
                new_is_default=new_is_default,
            ),
        )
        return
    if new_character:
        self.emit_event(
            "character_loaded",
            CharacterLoadedEvent(
                celune=self,
                character_name=new_character,
                bundle_path=new_bundle_path,
                is_default=new_is_default,
            ),
        )
        return
    if old_character:
        self.emit_event(
            "character_unloaded",
            CharacterUnloadedEvent(
                celune=self,
                character_name=old_character,
                bundle_path=old_bundle_path,
            ),
        )


def _backend_reload_kwargs(
    self,
    backend_spec: Union[str, CeluneBackend, type[CeluneBackend]],
) -> dict[str, JSONSerializable]:
    """Return constructor kwargs needed to instantiate one backend specification."""
    backend_kwargs: dict[str, JSONSerializable] = {
        "quantize": tts_quantization_enabled(self.config, backend_spec),
    }
    raw_name = getattr(backend_spec, "name", None)
    backend_name = (
        backend_spec.strip().lower()
        if isinstance(backend_spec, str)
        else raw_name
        if isinstance(raw_name, str)
        else None
    )
    quantize_kv_cache = (
        config_value(self.config, "quantize_kv_cache", True) is not False
    )
    if isinstance(backend_spec, CeluneBackend):
        backend_spec.quantization_requested = bool(backend_kwargs["quantize"])
        backend_spec.quantize_kv_cache = quantize_kv_cache
        return {}

    if backend_name in TTS_KV_CACHE_BACKENDS:
        backend_kwargs["quantize_kv_cache"] = quantize_kv_cache
    if backend_name == "qwen3":
        preset = resolve_vram_preset(self.config)
        backend_kwargs["x_vector_only"] = config_bool(
            self.config,
            "CELUNE_QWEN3_X_VECTOR_ONLY",
            "qwen3_x_vector_only",
        )
        backend_kwargs["clone_model_id"] = preset.qwen3_clone_model_id
    backend_kwargs.update(config_backend_options(self.config, backend_name))
    return backend_kwargs


def _capture_reload_snapshot(self) -> ReloadSnapshot:
    """Capture the current backend runtime state for rollback."""
    return ReloadSnapshot(
        backend=self.backend,
        restorable_backend_spec=_restorable_backend_spec(self),
        backend_spec=self._backend_spec,
        backend_kwargs=dict(self._backend_kwargs),
        vc_backend=self.vc_backend,
        restorable_vc_backend_spec=restorable_vc_backend_spec(self),
        vc_backend_spec=self._vc_backend_spec,
        voice_conversion_backend=self.voice_conversion_backend,
        tts_backend=self.tts_backend,
        input_mode=self.input_mode,
        model=self.model,
        model_name=self.model_name,
        voices=self.voices,
        current_voice=self.current_voice,
        current_character=self.current_character,
        current_character_persona=self.current_character_persona,
        voice_bundle_is_default=self.voice_bundle_is_default,
        loaded=self.loaded,
        cur_state=self.cur_state,
    )


def _release_reload_snapshot(self, snapshot: ReloadSnapshot) -> None:
    """Drop snapshot references after a successful hot reload."""
    snapshot.model = None
    snapshot.backend = self.backend
    snapshot.vc_backend = self.vc_backend
    snapshot.current_character_persona = None


def _restorable_backend_spec(self) -> Union[str, type[CeluneBackend]]:
    """Return a backend specification that does not pin the current instance."""
    if self._backend_spec is not None:
        return self._backend_spec
    return type(self.backend)


def _restore_reload_snapshot(self, snapshot: ReloadSnapshot) -> None:
    """Restore backend runtime state after a failed hot reload."""
    self.backend = snapshot.backend
    self._backend_spec = snapshot.backend_spec
    self._backend_kwargs = dict(snapshot.backend_kwargs)
    self.vc_backend = snapshot.vc_backend
    self._vc_backend_spec = snapshot.vc_backend_spec
    self.voice_conversion_backend = snapshot.voice_conversion_backend
    self.tts_backend = snapshot.tts_backend
    self.input_mode = snapshot.input_mode
    self.model = cast(Optional[PreTrainedModel], snapshot.model)
    self.backend.model = snapshot.model
    self.model_name = snapshot.model_name
    self.voices = snapshot.voices
    self.current_voice = snapshot.current_voice
    self.current_character = snapshot.current_character
    self.current_character_persona = snapshot.current_character_persona
    self.voice_bundle_is_default = snapshot.voice_bundle_is_default
    self.loaded = snapshot.loaded
    self.cur_state = snapshot.cur_state


def _rebuild_reload_snapshot_runtime(self, snapshot: ReloadSnapshot) -> None:
    """Recreate the previous backend runtime from a rollback snapshot."""
    close_backend(snapshot.backend)
    restored_backend = resolve_tts_backend(
        self,
        snapshot.restorable_backend_spec,
        log=self.log_callback,
        fatal=self.fatal,
        **snapshot.backend_kwargs,
    )
    restored_backend.bind_fatal(self.fatal)
    if restored_backend.uses_voice_bundles:
        restored_backend.validate_refs()

    restored_model: Optional[PreTrainedModel] = None
    restored_model_name = snapshot.model_name
    if snapshot.loaded and snapshot.current_voice is not None:
        restored_model, restored_model_name = _load_backend_voice_runtime(
            restored_backend,
            snapshot.current_voice,
        )

    snapshot.backend = restored_backend
    snapshot.model = cast(Optional[BackendModel], restored_model)
    snapshot.model_name = restored_model_name


def _rebuild_reload_snapshot_vc_runtime(self, snapshot: ReloadSnapshot) -> None:
    """Recreate the previous VC backend runtime from a rollback snapshot."""
    if snapshot.restorable_vc_backend_spec is None:
        return

    if snapshot.vc_backend is not None:
        close_backend(snapshot.vc_backend)
    restored_vc_backend = resolve_vc_backend(
        self,
        snapshot.restorable_vc_backend_spec,
        log=self.log_callback,
    )
    if available("f0_condition", obj=restored_vc_backend):
        restored_vc_backend.f0_condition = self.vc_f0_condition
    if snapshot.loaded:
        restored_vc_backend.preload_models()
    snapshot.vc_backend = restored_vc_backend


def _resolve_voice_state(
    backend: CeluneBackend,
    preferred_voice: Optional[str] = None,
) -> tuple[
    tuple[str, ...],
    Optional[str],
    Optional[str],
    Optional[CEVoicePersona],
    bool,
]:
    """Resolve voices and character metadata for one backend and active bundle."""
    if backend.uses_voice_bundles:
        loader = default_loader()
        if loader is not None:
            voices = loader.bundle.voice_order
            configured_default = loader.bundle.metadata.get("default_voice")
            default_voice = (
                configured_default
                if isinstance(configured_default, str)
                else backend.default_voice
            )
            current_voice = (
                preferred_voice
                if preferred_voice in voices
                else default_voice
                if default_voice in voices
                else voices[0]
                if voices
                else None
            )
            return (
                voices,
                current_voice,
                bundle_character_name(loader.bundle),
                persona_metadata_from_manifest(loader.bundle.metadata),
                bundle_matches_default_pack_checksum(loader.bundle.path),
            )

    voices = tuple(backend.voices)
    current_voice = (
        preferred_voice
        if preferred_voice in voices
        else backend.default_voice
        if backend.default_voice in voices
        else voices[0]
        if voices
        else None
    )
    return voices, current_voice, None, None, True


def _load_backend_voice_runtime(
    backend: CeluneBackend,
    voice: str,
) -> tuple[Optional[PreTrainedModel], str]:
    """Load the TTS runtime for one backend and voice without disturbing the previous runtime."""
    model_name = backend.model_id_for_voice(voice)
    try:
        model = cast(PreTrainedModel, backend.load_model(model_name))
    except Exception as error:
        if is_cuda_out_of_memory(error):
            with contextlib.suppress(Exception):
                backend.unload_model()
            release_cuda_after_oom()
            raise
        if not backend.quantization_requested:
            raise
        backend.disable_runtime_quantization()
        backend.unload_model()
        try:
            model = cast(PreTrainedModel, backend.load_model(model_name))
        except Exception as recovery_error:
            if is_cuda_out_of_memory(recovery_error):
                with contextlib.suppress(Exception):
                    backend.unload_model()
                release_cuda_after_oom()
            fatal = getattr(backend, "_fatal_callback", None)
            if callable(fatal):
                fatal()
            raise
    backend.model = model
    return model, model_name


def reload_backend(
    self,
    backend_spec: CoreBackendSpec,
    preferred_voice: Optional[str] = None,
) -> bool:
    """Switch backend while reserving the shared model lifecycle resource."""
    acquired, component_lease = lifecycle.acquire_model_loading_lease(
        self, f"backend-reload:{time.monotonic_ns()}"
    )
    if not acquired:
        _release_unstarted_reload(self)
        return False
    try:
        return _hot_reload_backend_impl(self, backend_spec, preferred_voice)
    finally:
        if component_lease is not None:
            component_lease.release()


def _hot_reload_backend_impl(
    self,
    backend_spec: CoreBackendSpec,
    preferred_voice: Optional[str] = None,
) -> bool:
    """Synchronously switch to a new backend family with rollback on failure."""
    snapshot: Optional[ReloadSnapshot] = None
    candidate_kwargs: dict[str, JSONSerializable] = {}
    candidate_backend: Optional[CeluneBackend] = None
    candidate_vc_backend: Optional[CeluneVCBackend] = None
    candidate_model: Optional[PreTrainedModel] = None
    candidate_voice: Optional[str] = None
    requested_name = (
        backend_spec.name
        if isinstance(backend_spec, (CeluneBackend, CeluneVCBackend))
        else str(backend_spec)
    )

    try:
        snapshot = _capture_reload_snapshot(self)
        preset = resolve_vram_preset(self.config)
        backend_target, normalized_backend_spec = _core_backend_target(
            backend_spec,
            self.log_callback,
            self.input_mode,
        )
        if backend_target == "tts":
            candidate_kwargs = _backend_reload_kwargs(
                self, cast(TTSBackendSpec, normalized_backend_spec)
            )
        self.log(
            string(
                "celune.switching_backend",
                app_name=APP_NAME,
                backend=requested_name,
            )
        )
        self._ready_announced = False
        self.status_callback(string("status.reloading_backend"))
        self.progress_callback(None, None)
        self.cur_state = "reloading"

        previous_backend = self.backend
        previous_vc_backend = self.vc_backend
        previous_voice = snapshot.current_voice

        if backend_target == "tts":
            candidate_backend = resolve_tts_backend(
                self,
                cast(TTSBackendSpec, normalized_backend_spec),
                log=self.log_callback,
                fatal=self.fatal,
                **candidate_kwargs,
            )
            if not _track_reload_backend(self, candidate_backend):
                close_backend(candidate_backend)
                return False
            candidate_backend.bind_fatal(self.fatal)
            validate_backend_against_preset(candidate_backend, preset)
            if candidate_backend.uses_voice_bundles:
                candidate_backend.validate_refs()
            if previous_vc_backend is not None:
                dispose_backend(previous_vc_backend)
            if previous_backend is not candidate_backend:
                dispose_backend(previous_backend)
            candidate_backend.preload_models()

            (
                candidate_voices,
                candidate_voice,
                candidate_character,
                candidate_persona,
                candidate_bundle_is_default,
            ) = _resolve_voice_state(candidate_backend, preferred_voice)
            if candidate_voice is None:
                raise BackendError("no voices found")

            model, model_name = _load_backend_voice_runtime(
                candidate_backend,
                candidate_voice,
            )
            candidate_model = model

            from .loading import warmup

            if not warmup(
                self,
                fatal_on_failure=False,
                backend=candidate_backend,
                model=model,
                voice=candidate_voice,
            ):
                raise_warmup_error(self, "warmup failed after backend reload")
            candidate_model = cast(Optional[PreTrainedModel], candidate_backend.model)

            self.backend = candidate_backend
            self.vc_backend = None
            self._vc_backend_spec = None
            self.voice_conversion_backend = ""
            self.input_mode = "text_to_speech"
            self.tts_backend = candidate_backend.name
            self.model = model
            self.model_name = model_name
            self.voices = candidate_voices
            self.current_voice = candidate_voice
            self.current_character = candidate_character
            self.current_character_persona = candidate_persona
            self.voice_bundle_is_default = candidate_bundle_is_default
            self._backend_spec = (
                cast(Union[str, type[CeluneBackend]], normalized_backend_spec)
                if not isinstance(normalized_backend_spec, CeluneBackend)
                else type(candidate_backend)
            )
            self._backend_kwargs = dict(candidate_kwargs)
        else:
            candidate_vc_backend = resolve_vc_backend(
                self,
                cast(VCBackendSpec, normalized_backend_spec),
                log=self.log_callback,
            )
            if not _track_reload_backend(self, candidate_vc_backend):
                close_backend(candidate_vc_backend)
                return False
            if available("f0_condition", obj=candidate_vc_backend):
                candidate_vc_backend.f0_condition = self.vc_f0_condition
            if previous_backend is not None:
                dispose_backend(previous_backend)
            if (
                previous_vc_backend is not None
                and previous_vc_backend is not candidate_vc_backend
            ):
                dispose_backend(previous_vc_backend)
            candidate_vc_backend.preload_models()

            self.vc_backend = candidate_vc_backend
            self._vc_backend_spec = (
                cast(
                    Union[str, type[CeluneVCBackend]],
                    normalized_backend_spec,
                )
                if not isinstance(normalized_backend_spec, CeluneVCBackend)
                else type(candidate_vc_backend)
            )
            self.voice_conversion_backend = candidate_vc_backend.name
            self.input_mode = "voice_conversion"
            self.model = None
            self.model_name = ""

        _release_reload_snapshot(self, snapshot)
        if backend_target == "tts" and self.use_normalization:
            lifecycle.unload_normalizer_components(self)
            self.load_normalizer()
        elif backend_target == "vc":
            lifecycle.unload_normalizer_components(self)
        self.loaded = True
        self.log(string("celune.switched_backend", backend=requested_name))
        self.progress_callback(1, 1)
        self.cur_state = "idle"
        self.status_callback(string("status.idle"))
        self._model_ready.set()
        if candidate_backend is not None:
            if candidate_voice is None:
                raise BackendError("no voices found")
            self.voice_changed_callback(candidate_voice)
            if previous_voice != candidate_voice:
                self.emit_event(
                    "voice_changed",
                    VoiceChangedEvent(
                        celune=self,
                        old_voice=previous_voice or candidate_voice,
                        new_voice=candidate_voice,
                    ),
                )
        return True
    except Exception as error:
        self.log(
            format_error_message(
                tagged_string("celune.reload_error", "RELOAD ERROR"),
                error,
                self.log_level,
            ),
            "error",
        )
        if self.exit_requested:
            if candidate_backend is not None:
                close_backend(candidate_backend)
            if candidate_vc_backend is not None:
                close_backend(candidate_vc_backend)
            return False
        self.status_callback(string("status.restoring_backend"))
        self.progress_callback(None, None)
        if (
            candidate_backend is not None
            and snapshot is not None
            and candidate_backend is not snapshot.backend
        ):
            shutdown_backend(candidate_backend, release_cuda_cache=True)
        elif (
            candidate_model is not None
            and snapshot is not None
            and candidate_model is not snapshot.model
        ):
            release_loaded_object(candidate_model)
        if (
            candidate_vc_backend is not None
            and snapshot is not None
            and candidate_vc_backend is not snapshot.vc_backend
        ):
            shutdown_backend(candidate_vc_backend, release_cuda_cache=True)
        rollback_succeeded = True
        if snapshot is not None:
            try:
                if snapshot.input_mode == "voice_conversion":
                    if (
                        snapshot.loaded
                        and snapshot.restorable_vc_backend_spec is not None
                    ):
                        _rebuild_reload_snapshot_vc_runtime(self, snapshot)
                    _restore_reload_snapshot(self, snapshot)
                elif snapshot.backend.model is None and snapshot.loaded:
                    _rebuild_reload_snapshot_runtime(self, snapshot)
                    _restore_reload_snapshot(self, snapshot)
                else:
                    _restore_reload_snapshot(self, snapshot)
            except Exception as restore_error:
                rollback_succeeded = False
                self.log(
                    format_error_message(
                        string("celune.backend_restore_error"),
                        restore_error,
                        self.log_level,
                    ),
                    "error",
                )
                _restore_reload_snapshot(self, snapshot)
                self.model = None
                self.backend.model = None
                if self.input_mode == "voice_conversion":
                    self.vc_backend = None
                    self._vc_backend_spec = None
                    self.voice_conversion_backend = ""
                self.loaded = False
                self.model_name = ""
                self.cur_state = "idle"
        else:
            self.cur_state = "idle"
        self._last_warmup_error = None
        self.status_callback(string("status.idle"))
        self.progress_callback(1, 1)
        if rollback_succeeded:
            self.log(string("celune.backend_restore_failed"), "warning")
        self.error_callback(string("status.could_not_reload", app_name=APP_NAME))
        return False
    finally:
        _clear_reload_backend(
            self,
            candidate_backend
            if candidate_backend is not None
            else candidate_vc_backend,
        )
        self._reload_pending = False
        self._model_ready.set()
        self._last_component_busy = None
        self.change_input_state_callback(locked=False)
        self.change_voice_lock_state_callback(locked=len(self.voices) < 2)


def _track_reload_backend(
    self,
    backend: Union[CeluneBackend, CeluneVCBackend],
) -> bool:
    """Publish a reload candidate to shutdown before it performs work."""
    with self._model_lock:
        if self._closed:
            return False
        self._reload_backend = backend
        return True


def _clear_reload_backend(
    self,
    backend: Optional[Union[CeluneBackend, CeluneVCBackend]],
) -> None:
    """Drop one reload candidate without clearing a newer candidate."""
    if backend is None:
        return
    with self._model_lock:
        if self._reload_backend is backend:
            self._reload_backend = None


def reload_cevoice(
    self,
    bundle: Optional[Union[str, Path]],
    preferred_voice: Optional[str] = None,
) -> bool:
    """Switch voice bundle while reserving the model lifecycle resource."""
    acquired, component_lease = lifecycle.acquire_model_loading_lease(
        self, f"cevoice-reload:{time.monotonic_ns()}"
    )
    if not acquired:
        _release_unstarted_reload(self)
        return False
    try:
        return _hot_reload_cevoice_impl(self, bundle, preferred_voice)
    finally:
        if component_lease is not None:
            component_lease.release()


def _hot_reload_cevoice_impl(
    self,
    bundle: Optional[Union[str, Path]],
    preferred_voice: Optional[str] = None,
) -> bool:
    """Synchronously switch to a new CEVOICE bundle with rollback on failure."""
    snapshot: Optional[ReloadSnapshot] = None
    previous_bundle = active_bundle_path()
    loaded_model: Optional[PreTrainedModel] = None

    try:
        snapshot = _capture_reload_snapshot(self)
        previous_loader = default_loader()
        previous_bundle = (
            Path(previous_loader.bundle.path)
            if previous_loader is not None
            else active_bundle_path()
        )
        self.log(string("celune.reloading_character", app_name=APP_NAME))
        self._ready_announced = False
        self.status_callback(string("status.reloading_character"))
        self.progress_callback(None, None)
        self.cur_state = "reloading"

        select_voice_bundle(bundle)
        if self.backend.uses_voice_bundles:
            self.backend.validate_refs()

        (
            candidate_voices,
            candidate_voice,
            candidate_character,
            candidate_persona,
            candidate_bundle_is_default,
        ) = _resolve_voice_state(self.backend, preferred_voice)
        if candidate_voice is None:
            raise BackendError("no voices found in current CEVOICE")

        self.voices = candidate_voices
        self.current_voice = candidate_voice
        self.current_character = candidate_character
        self.current_character_persona = candidate_persona
        self.voice_bundle_is_default = candidate_bundle_is_default

        if snapshot.current_character != self.current_character or str(
            previous_bundle
        ) != str(active_bundle_path()):
            self.reset_persona_conversation()

        if not lifecycle.is_voice_conversion_mode(self):
            model, model_name = _load_backend_voice_runtime(
                self.backend,
                candidate_voice,
            )
            loaded_model = model
            previous_model = snapshot.model
            self.model = model
            self.model_name = model_name
            from .loading import warmup

            if not warmup(self, fatal_on_failure=False):
                raise_warmup_error(self, "warmup failed after CEVOICE reload")
            if previous_model is not None and previous_model is not model:
                release_loaded_object(cast(ReleasableObject, previous_model))

        self.loaded = True
        self.log(
            string(
                "celune.switched_character",
                character=(bundle if bundle is not None else previous_bundle.name),
            )
        )
        self.progress_callback(1, 1)
        self.cur_state = "idle"
        self.status_callback(string("status.idle"))
        self._model_ready.set()
        self.voice_changed_callback(candidate_voice)
        emit_character_event_transition(
            self,
            snapshot.current_character,
            str(previous_bundle),
            self.current_character,
            str(active_bundle_path()),
            self.voice_bundle_is_default,
        )
        if snapshot.current_voice != candidate_voice:
            self.emit_event(
                "voice_changed",
                VoiceChangedEvent(
                    celune=self,
                    old_voice=snapshot.current_voice or candidate_voice,
                    new_voice=candidate_voice,
                ),
            )
        return True
    except Exception as error:
        self.log(
            format_error_message(
                tagged_string("celune.reload_error", "RELOAD ERROR"),
                error,
                self.log_level,
            ),
            "error",
        )
        if (
            loaded_model is not None
            and snapshot is not None
            and loaded_model is not snapshot.model
        ):
            release_loaded_object(loaded_model)
        select_voice_bundle(previous_bundle)
        if snapshot is not None:
            _restore_reload_snapshot(self, snapshot)
        else:
            self.cur_state = "idle"
        self.status_callback(string("status.idle"))
        self.progress_callback(1, 1)
        self.log(
            string("celune.character_restore_failed"),
            "warning",
        )
        return False
    finally:
        self._reload_pending = False
        self._model_ready.set()
        self._last_component_busy = None
        self.change_input_state_callback(locked=False)
        self.change_voice_lock_state_callback(locked=len(self.voices) < 2)


def _release_unstarted_reload(self) -> None:
    """Restore readiness after a reload could not acquire model ownership."""
    self._reload_pending = False
    self._model_ready.set()
    self._last_component_busy = None
    self.change_input_state_callback(locked=False)
    self.change_voice_lock_state_callback(locked=len(self.voices) < 2)


def raise_warmup_error(self, message: str) -> None:
    """Raise a Celune warmup error while preserving any original cause."""
    if self._last_warmup_error is not None:
        raise WarmupError(message) from self._last_warmup_error
    raise WarmupError(message)


@contextlib.contextmanager
def with_backend(
    self,
    backend_spec: CoreBackendSpec,
    timeout: float = 30.0,
):
    """Temporarily switch Celune to another backend within a context block.

    Args:
        backend_spec: The backend name, type, or instance to activate temporarily.
        timeout: Maximum seconds to wait while switching or restoring the backend.

    Raises:
        BackendError: Celune could not switch to or restore the requested backend.
    """
    restore_backend = restorable_active_backend_spec(self)
    restore_voice = self.current_voice
    if not self.wait_until_idle(timeout=timeout):
        raise BackendError("timed out switching backend")
    if not reload_backend(self, backend_spec, self.current_voice):
        raise BackendError("failed to switch backend")

    try:
        yield self
    finally:
        self.wait_until_idle(timeout=timeout)
        if not reload_backend(self, restore_backend, restore_voice):
            raise BackendError("failed to restore old backend")


@contextlib.contextmanager
def with_cevoice(
    self,
    bundle: Optional[Union[str, Path]],
    timeout: float = 30.0,
):
    """Temporarily switch Celune to another CEVOICE bundle within a context block.

    Args:
        bundle: The CEVOICE bundle name or path to activate temporarily.
        timeout: Maximum seconds to wait while switching or restoring the CEVOICE pack.

    Raises:
        BackendError: Celune could not switch to or restore the requested CEVOICE pack.
    """
    previous_loader = default_loader()
    restore_bundle = (
        Path(previous_loader.bundle.path)
        if previous_loader is not None
        else active_bundle_path()
    )
    restore_voice = self.current_voice
    if not self.wait_until_idle(timeout=timeout):
        raise BackendError("timed out switching character")
    if not reload_cevoice(self, bundle, None):
        raise BackendError("failed to switch character")

    try:
        yield self
    finally:
        self.wait_until_idle(timeout=timeout)
        if not reload_cevoice(self, restore_bundle, restore_voice):
            raise BackendError("failed to restore character")


def wait_until_idle(
    self,
    timeout: float = 30.0,
    *,
    wait_for_speech: Optional[bool] = None,
) -> bool:
    """Wait until the model and speech pipeline are ready.

    Args:
        timeout: Maximum time to wait for readiness.
        wait_for_speech: Whether active speech must finish before returning.
    """
    # don't wait a timeout while Celune is downloading a model
    ok = self._model_ready.wait(timeout=timeout)
    if not ok:
        self.log(string("celune.ready_wait_timeout"), "warning")
        self.log(string("celune.ready_wait_reason"), "warning")
        self.log(string("celune.ready_wait_not_fatal"), "warning")
        return False

    if not self.loaded:
        self.log(string("celune.model_unloaded_while_waiting"), "warning")
        return False

    if wait_for_speech is None:
        from .voice import speech_playback_active

        wait_for_speech = speech_playback_active(self)

    if wait_for_speech:
        ok = self._playback_done.wait(timeout=timeout)
        if not ok:
            self.log(
                string("celune.playback_idle_timeout"),
                "warning",
            )
            return False

    with self._say_lock:
        return (not self.locked) and self.loaded


def reload_model_for_language(
    engine: Celune,
    target_language: Optional[str],
) -> None:
    """Reload a backend model when the requested language needs a new variant.

    Args:
        engine: Runtime that owns the active model and backend.
        target_language: Language requested by the current speech operation.

    Raises:
        NotAvailableError: A required model reference or active voice is missing.
        Exception: The backend fails while unloading or loading the model.
    """
    backend = engine.backend
    should_reload = getattr(backend, "should_reload_for_language", None)
    if engine.model is None and not callable(should_reload):
        raise NotAvailableError("cannot generate without a model reference")
    if not callable(should_reload) or (
        engine.model is not None and not should_reload(target_language)
    ):
        return

    active_voice = engine.current_voice or backend.default_voice
    if active_voice is None:
        raise NotAvailableError("cannot switch language without an active voice")

    model_id = backend.model_id_for_voice(active_voice)
    engine.log(
        f"[RELOAD] Loading {model_id} for language: {target_language}",
        loglevel="verbose",
    )
    engine.model = None
    backend.unload_model()
    engine.model = backend.load_model(model_id, lang=target_language)
    engine.model_name = model_id


if TYPE_CHECKING:
    from ..celune import Celune
