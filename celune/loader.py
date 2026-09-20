# SPDX-License-Identifier: Apache-2.0
"""Celune core loading methods."""

from __future__ import annotations

import contextlib
import gc
import os
import threading
import time
from typing import Optional, cast

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .cevoice import announce_default_bundle, bundle_display_name, default_loader
from .backends.tts import CeluneBackend
from .constants import APP_NAME, NORMALIZER_MODEL_ID
from .dataclasses.events import ReadyEvent
from .exceptions import BackendError, NotAvailableError, RuntimeCheckError, WarmupError
from .i18n import string, tagged_string
from .modeling import load_normalizer_components, normalizer_device
from .persona.impl import persona_enabled
from .pipeline import (
    force_stop_speech as force_stop_pipeline,
    saved_output_speech_seconds,
)
from .runtime import log_runtime_banner, validate_runtime
from .threads import run_in_daemon_thread
from .typing.celune import Generative, NormalizerTokenizer
from .utils import (
    custom_assert,
    discard,
    format_error,
    format_error_message,
    format_number,
    is_port_usable,
)
from .vram import resolve_vram_preset, validate_vram_preset
from .binding import install_class_functions

__all__ = (
    "_start_configured_api",
    "_warmup",
    "enter_sleep_mode_async",
    "force_stop_speech_async",
    "load",
    "load_normalizer",
    "normalize",
    "wake_from_sleep_async",
)


async def force_stop_speech_async(self) -> bool:
    """Forcefully stop Celune from speaking without blocking an async caller.

    Returns:
        ``True`` when an active utterance was interrupted.
    """
    return await run_in_daemon_thread(force_stop_pipeline, self)


async def enter_sleep_mode_async(self) -> bool:
    """Put Celune to sleep without blocking the caller's event loop.

    Returns:
        ``True`` when Celune enters sleep mode successfully.
    """
    await run_in_daemon_thread(self._async_runtime_lock.acquire)
    try:
        return await run_in_daemon_thread(self.enter_sleep_mode)
    finally:
        self._async_runtime_lock.release()


async def wake_from_sleep_async(self) -> bool:
    """Wake Celune without blocking the caller's event loop.

    Returns:
        ``True`` when Celune wakes successfully.
    """
    await run_in_daemon_thread(self._async_runtime_lock.acquire)
    try:
        return await run_in_daemon_thread(self.wake_from_sleep)
    finally:
        self._async_runtime_lock.release()


def load(self, raise_on_error: bool = False, skip_runtime_check: bool = False) -> bool:
    """Load and initialize Celune.

    Args:
        raise_on_error: Whether Celune should raise for failures to load or signal fatal states.
        skip_runtime_check: Whether Celune should skip runtime checks during startup.

    Returns:
        bool: ``True`` when initialization completed successfully, otherwise ``False``.

    Raises:
        NotAvailableError: If no usable voice or backend is available.
        Exception: If an unexpected loading failure occurs and `raise_on_error` is enabled.
        RuntimeCheckError: If the runtime environment is unsupported.
        BackendError: If backend initialization fails.
    """
    if self._startup_callback is not None:
        self._startup_callback(string("ui.startup_initializing_core"))
    log_runtime_banner(
        self._emit_runtime_banner_line,
        self.vc_backend or self.backend,
        self.backend_mode,
    )
    self._startup_banner_emitted = True
    self._flush_startup_logs()
    self.historical_generated_speech_seconds = saved_output_speech_seconds()

    if self.backend_mode == "ui_test" and self.backend.is_fake:
        self.log(string("celune.test_mode_active", app_name=APP_NAME))
        return True

    if not self.load_available_voices():
        self.fatal()
        self.log(string("celune.no_voices_loaded"), "error")
        self.error_callback(string("celune.no_voices_loaded_short"))
        self.progress_callback(0, 1)
        if raise_on_error:
            raise NotAvailableError("no voices are available")
        return False

    if self.backend.uses_voice_bundles:
        announced_character = announce_default_bundle(self.log)
        character = self.current_character or announced_character
        self.current_character = character
        loader = default_loader()
        display_character = (
            bundle_display_name(loader.bundle) if loader is not None else character
        )
        self.log(string("celune.current_character", character=display_character))

    if self.backend_mode == "normal":
        self.setup_extensions()

    vram_message = validate_vram_preset(self.config)
    if vram_message:
        self.log(vram_message, "warning")

    effective_vram_preset = resolve_vram_preset(self.config)
    self.log(
        string(
            "celune.current_vram_preset",
            preset=effective_vram_preset.tier.title(),
        )
    )

    self.progress_callback(None, None)
    if self._is_voice_conversion_mode():
        if self.vc_backend is None:
            self.fatal()
            self.log(string("celune.no_vc_backend"), "error")
            self.error_callback(string("celune.no_valid_vc_backend"))
            self.progress_callback(0, 1)
            if raise_on_error:
                raise NotAvailableError(
                    "requested VC mode, but no VC backend was loaded"
                )
            return False

        self.vc_backend.preload_models()
        self.model = None
        self.model_name = ""
        self.log(string("celune.ready_for_vc"))
    else:
        self.backend.preload_models()

        self.log(string("celune.all_voices_available"))
        try:
            self.model = self.backend.load_default_model()
            active_voice = self.current_voice or self.voices[0]
            self.model_name = self.backend.model_id_for_voice(active_voice)
        except Exception as e:
            self.fatal()
            self.log(
                format_error_message(
                    string("celune.default_model_load_failed", app_name=APP_NAME),
                    e,
                    self.log_level,
                ),
                "error",
            )
            self.error_callback(string("celune.default_model_failed_short"))
            self.progress_callback(0, 1)
            if raise_on_error:
                raise
            return False

    pipeline_thread = threading.Thread(target=self._run_pipeline_jobs, daemon=True)
    self._generation_thread = None
    self._playback_thread = pipeline_thread
    pipeline_thread.start()

    if not skip_runtime_check and not validate_runtime(
        log=self.log,
        error=self.error_callback,
        set_state=lambda state: setattr(self, "cur_state", state),
        glow_connect_failed=self.glow.connect_failed,
        format_error=format_error,
        log_level=self.log_level,
        backend_name=self._active_runtime_backend_name(),
    ):
        self.fatal()
        self._stop_pipeline_jobs()
        if raise_on_error:
            raise RuntimeCheckError("runtime check failed")
        return False

    warmup_ok = True
    if not self._is_voice_conversion_mode():
        warmup_ok = self._warmup()

    if warmup_ok:
        self.loaded = True
        self._model_ready.set()
        self._release_pipeline()
        self.glow.enter()  # Celune has entered your PC
    else:
        self.fatal()
        self.log(tagged_string("celune.warmup_failed", "WARMUP"), "error")
        self._stop_pipeline_jobs()
        if raise_on_error:
            raise BackendError("warmup failed")
        return False

    self._start_persona_background_load()

    if self.use_normalization:
        self.load_normalizer()

    if self.backend_mode == "normal":
        self._start_configured_api()

    if persona_enabled(self.config) and self.vision is None:
        self.log(
            string("celune.personas_unavailable", app_name=APP_NAME),
            "warning",
        )

    if self.backend_mode == "normal" and not self._try_play_signal("readiness"):
        self.log(
            "Could not play the readiness signal.",
            "warning",
            loglevel="verbose",
        )

    self._emit_event("ready", ReadyEvent(celune=self))

    return True


def _start_configured_api(self) -> None:
    """Start the API from config without blocking Celune startup."""
    enabled, host, port, token, requests_per_minute = self._api_settings()
    if not enabled or self._api_thread is not None:
        return

    if not is_port_usable(port):
        self.log(f"API port {port} is already in use.", "warning")
        self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
        return

    try:
        from .api import start_api
    except ModuleNotFoundError as package:
        self.log(
            string(
                "celune.required_package_missing",
                package=package.name,
            ),
            "warning",
        )
        self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
        return
    except Exception as e:
        self.log(
            format_error_message(
                string("celune.package_import_failed"),
                e,
                self.log_level,
            ),
            "warning",
        )
        self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
        return

    try:
        self._api_thread = start_api(
            self,
            host=host,
            port=port,
            token=token,
            requests_per_minute=requests_per_minute,
        )
    except Exception as error:
        self.log(
            format_error_message(
                string("celune.internal_error"),
                error,
                self.log_level,
            ),
            "warning",
        )
        self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
        return


def load_normalizer(self) -> None:
    """Load the normalizer LLM."""
    load_epoch = self._normalizer_load_epoch + 1
    self._normalizer_load_epoch = load_epoch

    def _worker():
        loaded_tokenizer: Optional[PreTrainedTokenizerBase] = None
        loaded_llm: Optional[PreTrainedModel] = None

        discard(loaded_tokenizer)
        discard(loaded_llm)
        try:
            loaded_tokenizer, loaded_llm = load_normalizer_components(
                self.log,
                self.backend,
                self.config,
                progress_callback=None,
            )
            with self._model_lock:
                if (
                    self._normalizer_load_epoch != load_epoch
                    or self.sleeping
                    or self.exit_requested
                ):
                    discard(loaded_llm)
                    discard(loaded_tokenizer)
                    gc.collect()
                    if torch.cuda.is_available():
                        with contextlib.suppress(Exception):
                            torch.cuda.synchronize()
                        with contextlib.suppress(Exception):
                            torch.cuda.empty_cache()
                    self.log(
                        "[NORMALIZER] Discarded stale normalizer load.",
                        loglevel="verbose",
                    )
                    return

                self.tokenizer = loaded_tokenizer
                self.llm = loaded_llm
            self.log(string("celune.normalizer_loaded"))
        except Exception as e:
            self.log(
                format_error_message(
                    "[NORMALIZER ERROR]",
                    e,
                    self.log_level,
                ),
                "error",
            )
            self.log(string("celune.normalizer_failed"), "warning")
            self.log(string("celune.normalization_unavailable"), "warning")

    with self._model_lock:
        if self.persona_ready or self.persona_loading:
            return

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    self.log(
        f"Loading normalizer {NORMALIZER_MODEL_ID} "
        f"on {normalizer_device(self.config)}..."
    )


def _warmup(
    self,
    fatal_on_failure: bool = True,
    backend: Optional[CeluneBackend] = None,
    model: Optional[PreTrainedModel] = None,
    voice: Optional[str] = None,
) -> bool:
    """Warm up Celune's speech capabilities."""
    self.log(tagged_string("celune.warmup_start", "WARMUP"))
    self.status_callback(string("status.warming_up"))
    self.progress_callback(None, None)
    warmup_text = "Hello. I am here, and I am listening carefully."
    self._last_warmup_error = None
    active_backend = backend if backend is not None else self.backend
    active_model = model if model is not None else self.model
    active_voice = voice if voice is not None else self.current_voice

    forced_error = os.getenv("CELUNE_FORCE_ERROR") in {
        "1",
        "true",
        "on",
        "yes",
        "enabled",
    }

    if forced_error:
        raise WarmupError("forced warmup failure")

    def run_warmup(candidate_model: Optional[PreTrainedModel]) -> None:
        """Run one speech probe against a loaded model."""
        with self._model_lock:
            if candidate_model is None:
                raise WarmupError("cannot warm up a null model")

            for _, _, _ in active_backend.generate_stream(
                candidate_model,
                text=warmup_text,
                language=self.language,
                chunk_size=self.chunk_size,
                instruct=self.effective_voice_prompt(),
                voice=active_voice,
            ):
                pass

    try:
        warmup_start = time.perf_counter()
        run_warmup(active_model)

        warmup_end = time.perf_counter()
        warmup_took = warmup_end - warmup_start
        self.log(
            f"[WARMUP] done, took {format_number(warmup_took, 2)} seconds",
            loglevel="verbose",
        )

        self.progress_callback(1, 1)
        return True
    except Exception as error:
        warmup_error = error
        quantization_recovery_failed = False
        if active_backend.quantization_active:
            try:
                active_backend.disable_runtime_quantization()
                active_backend.unload_model()
                if active_voice is None:
                    raise WarmupError(
                        "cannot recover a quantized model without a voice"
                    ) from error
                fallback_model = active_backend.load_model(
                    active_backend.model_id_for_voice(active_voice),
                    lang=self.language,
                )
                active_backend.model = fallback_model
                active_model = cast(PreTrainedModel, fallback_model)
                run_warmup(active_model)
                if active_backend is self.backend:
                    self.model = active_model
                self._last_warmup_error = None
                self.progress_callback(1, 1)
                return True
            except Exception as fallback_error:
                warmup_error = fallback_error
                quantization_recovery_failed = True
        self._last_warmup_error = warmup_error
        self.log(
            format_error_message(
                tagged_string("celune.warmup_error", "WARMUP ERROR"),
                warmup_error,
                self.log_level,
            ),
            "error",
        )
        self.progress_callback(0, 1)
        if fatal_on_failure or quantization_recovery_failed:
            self.fatal()
            self.error_callback(string("celune.warmup_failed_app", app_name=APP_NAME))
        return False


def normalize(self, text: str) -> Optional[str]:
    """Normalize input text using CeluneNorm.

    Args:
        text: The raw text to normalize before speech generation.

    Returns:
        Optional[str]: The normalized text, the original text for blank input, or ``None`` when normalization is
        unavailable or has failed.
    """

    if not self.use_normalization:
        return None

    if not text or not text.strip():
        return text

    if self.llm is None or self.tokenizer is None:
        return None

    llm = cast(Generative, self.llm)
    tokenizer = cast(NormalizerTokenizer, self.tokenizer)

    def _run_inference() -> Optional[str]:
        inf_start = time.perf_counter()
        try:
            bad_text = text.strip()
            norm_token = "<NORM>"

            # Are we using CeluneNorm?
            norm_token_id = tokenizer.convert_tokens_to_ids(norm_token)
            custom_assert(
                norm_token_id is not None, ValueError("not a CeluneNorm normalizer")
            )
            assert norm_token_id is not None

            custom_assert(
                norm_token_id != tokenizer.unk_token_id,
                ValueError("not a CeluneNorm normalizer"),
            )
            assert norm_token_id != tokenizer.unk_token_id

            prompt = f"{bad_text}{norm_token}"

            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

            device = next(llm.parameters()).device
            inputs = tokens.to(device)
            token_ids = cast(torch.Tensor, tokens["input_ids"])
            len_tokens = token_ids.shape[1]

            self.log(f"Tokens to normalize: {len_tokens}")
            if len_tokens > 512:
                self.log(string("celune.input_too_long_to_normalize"), "warning")
                return None

            with torch.inference_mode():
                output_ids = llm.generate(
                    **inputs,
                    # CeluneNorm will likely return less, unless you use up your whole context allowance
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_ids = inputs["input_ids"]

            if not isinstance(input_ids, torch.Tensor):
                self.log("Normalizer output was not a tensor.", "warning")
                return None

            prompt_len = input_ids.shape[1]
            new_ids = output_ids[0][prompt_len:]

            # CeluneNorm shouldn't do this, but if it does happen, stop Celune from saying nothing
            if new_ids.numel() == 0:
                self.log("Normalizer returned no tokens.", "warning")
                return None

            out = tokenizer.decode(new_ids, skip_special_tokens=True)

            # fix type checker
            if isinstance(out, list):
                out = out[0] if out else ""

            # too many <NORM>'s can break splitting
            if "<NORM>" in out:
                out = out.split("<NORM>", 1)[0].strip()

            # are we absolutely sure CeluneNorm did produce something before Celune gets to say it?
            if not out:
                self.log("Normalizer did not produce normal output.", "warning")
                return None

            inf_total = time.perf_counter() - inf_start
            self.log(f"Normalized text: {out}")
            self.log(f"Normalization took {format_number(inf_total, 2)} seconds.")

            return out

        except Exception as e:
            self.log(
                format_error_message(
                    "[NORMALIZATION ERROR]",
                    e,
                    self.log_level,
                ),
                "error",
            )
            return None

    return _run_inference()  # blocks the generation thread, but Celune doesn't mind it since the main thread is up


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(target, {name: globals()[name] for name in __all__})
