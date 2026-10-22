# SPDX-License-Identifier: Apache-2.0
"""Celune core loading methods."""

from __future__ import annotations

import gc
import os
import time
import threading
import contextlib
from typing import Optional, cast
from functools import partial

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import lifecycle
from ..i18n import string, tagged_string
from ..vram import vram_profile_fits, resolve_vram_preset, validate_vram_preset
from ..utils import (
    discard,
    format_error,
    custom_assert,
    format_number,
    is_port_usable,
    format_error_message,
)
from ..cevoice import default_loader, bundle_display_name, announce_default_bundle
from ..runtime import validate_runtime, log_runtime_banner
from ..modeling import normalizer_device, load_normalizer_components
from ..constants import APP_NAME
from ..exceptions import (
    WarmupError,
    BackendError,
    NotAvailableError,
    RuntimeCheckError,
    ModelConsentRequired,
)
from .concurrency import run_in_daemon_thread
from ..backends.tts import CeluneBackend
from ..persona.impl import persona_enabled
from ..typing.celune import Generative, NormalizerTokenizer
from ..registry.other import NORMALIZER_MODEL_ID
from ..pipeline.output import saved_output_speech_seconds
from ..pipeline.playback import force_stop_speech as force_stop_pipeline
from ..dataclasses.events import ReadyEvent


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
    from .callbacks import _emit_runtime_banner_line

    log_runtime_banner(
        partial(_emit_runtime_banner_line, self),
        self.vc_backend or self.backend,
        self.backend_mode,
    )
    self._startup_banner_emitted = True
    from .callbacks import _flush_startup_logs

    _flush_startup_logs(self)
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

    if vram_profile_fits(self.config) is False:
        message = string("vram.profile_exceeds_budget", app_name=APP_NAME)
        self.fatal()
        self.log(message, "error")
        self.error_callback(message)
        self.progress_callback(0, 1)
        if raise_on_error:
            raise BackendError(message)
        return False

    effective_vram_preset = resolve_vram_preset(self.config)
    self.log(
        string(
            "celune.current_vram_preset",
            preset=effective_vram_preset.tier.title(),
        )
    )

    self.progress_callback(None, None)
    if self.is_voice_conversion_mode():
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
        except ModelConsentRequired:
            raise
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

    pipeline_thread = threading.Thread(target=self.run_pipeline_jobs, daemon=True)
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
        backend_name=lifecycle.active_runtime_backend_name(self),
    ):
        self.fatal()
        lifecycle.stop_pipeline_jobs(self)
        if raise_on_error:
            raise RuntimeCheckError("runtime check failed")
        return False

    warmup_ok = True
    if not self.is_voice_conversion_mode():
        warmup_ok = warmup(self)

    if warmup_ok:
        self.loaded = True
        self._model_ready.set()
        self.release_pipeline()
        self.glow.enter()  # Celune has entered your PC
    else:
        self.fatal()
        self.log(tagged_string("celune.warmup_failed", "WARMUP"), "error")
        lifecycle.stop_pipeline_jobs(self)
        if raise_on_error:
            raise BackendError("warmup failed")
        return False

    lifecycle.start_persona_background_load(self)

    if self.use_normalization:
        self.load_normalizer()

    if self.config.get("captions") is True:
        from ..pipeline.captions import preload_caption_aligner

        preload_caption_aligner(self.log, self.log_level)

    if self.backend_mode == "normal":
        _start_configured_api(self)

    if persona_enabled(self.config) and self.vision is None:
        self.log(
            string("celune.personas_unavailable", app_name=APP_NAME),
            "warning",
        )

    from .voice import try_play_signal

    if self.backend_mode == "normal" and not try_play_signal(self, "readiness"):
        self.log(
            "Could not play the readiness signal.",
            "warning",
            loglevel="verbose",
        )

    self.emit_event("ready", ReadyEvent(celune=self))

    return True


def _start_configured_api(self) -> None:
    """Start the API from config without blocking Celune startup."""
    enabled, host, port, token, requests_per_minute, access_level = (
        lifecycle.api_settings(self)
    )
    if not enabled or self._api_thread is not None:
        return

    if not is_port_usable(port):
        self.log(f"API port {port} is already in use.", "warning")
        self.log(string("celune.api_unavailable", app_name=APP_NAME), "warning")
        return

    try:
        from ..api import start_api
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

    on_started = None
    if access_level == "everyone":
        from ..api import CloudflaredQuickTunnel

        def start_public_tunnel(api_host: str, api_port: int) -> None:
            """Start the public tunnel after the local API socket is ready."""
            tunnel = self._api_tunnel
            if tunnel is None or not tunnel.start(api_port):
                self.log(
                    string(
                        "celune.api_started",
                        app_name=APP_NAME,
                        url=f"http://{api_host}:{api_port}",
                    )
                )

        self._api_tunnel = CloudflaredQuickTunnel(self.log)
        on_started = start_public_tunnel

    try:
        self._api_thread = start_api(
            self,
            host=host,
            port=port,
            token=token,
            requests_per_minute=requests_per_minute,
            on_started=on_started,
            defer_startup_log=access_level == "everyone",
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


def warmup(
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
    self._last_warmup_error = None
    active_backend = backend if backend is not None else self.backend
    active_model = model if model is not None else self.model
    active_voice = voice if voice is not None else self.current_voice
    warmup_text = "Hello. I am here, and I am listening carefully."
    if active_backend.singing:
        warmup_text = "[bpm:120,key:c,scale:major][lang:en][a4:4][ɑ] Ah"

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

            backend_text = (
                active_backend.cesn_to_model(warmup_text)
                if active_backend.singing
                else warmup_text
            )
            for _, _, _ in active_backend.generate_stream(
                candidate_model,
                text=backend_text,
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
