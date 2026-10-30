# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backend implementation for Celune."""

import time
import inspect
import contextlib
from typing import Optional
from collections.abc import Callable, Iterator

import torch
from huggingface_hub import snapshot_download
from faster_qwen3_tts import FasterQwen3TTS, __version__ as qwen3_ver

from .base import CeluneBackend, local_hf_offline_mode, cached_hf_snapshot_path
from ...paths import (
    huggingface_progress,
    configure_numba_cache,
    huggingface_hub_cache_dir,
)
from ...modeling import (
    GraphQuantizedKVCache,
    QuantizedKVCache,
    create_graph_quantized_kv_cache,
    create_quantized_kv_cache,
)
from ...i18n import string
from ...typing.aliases import AudioChunk
from ...utils import available, custom_assert
from ...cevoice import CEVoiceLoader, default_loader
from ...registry.consent import require_model_consent
from ...registry.tts import TTS_DEFAULT_VOICE, QWEN3_DEFAULT_MODEL_ID


def _quantized_talker_cache(
    model: FasterQwen3TTS,
    enabled: bool,
) -> Optional[tuple[torch.nn.Module, QuantizedKVCache]]:
    """Prepare a cache and talker only when the installed Qwen path supports it."""
    try:
        if (
            "parity_mode"
            not in inspect.signature(model.generate_voice_clone_streaming).parameters
        ):
            return None
    except (TypeError, ValueError):
        return None

    current: Optional[object] = model
    for _ in range(4):
        if current is None:
            return None
        talker = getattr(current, "talker", None)
        if isinstance(talker, torch.nn.Module):
            config = getattr(getattr(current, "config", None), "talker_config", None)
            if config is None:
                config = getattr(talker, "config", None)
            try:
                device = next(talker.parameters()).device
            except StopIteration:
                return None
            if config is not None:
                try:
                    cache = create_quantized_kv_cache(
                        config=config,
                        device=device,
                        enabled=enabled,
                    )
                except (TypeError, ValueError):
                    return None
                if cache is not None:
                    return talker, cache
        current = getattr(current, "model", None)
    return None


def _install_graph_quantized_cache(
    model: FasterQwen3TTS,
    enabled: bool,
) -> bool:
    """Install fixed-capacity quantized storage into Qwen's talker graph."""
    talker_graph = getattr(model, "talker_graph", None)
    talker_model = getattr(talker_graph, "model", None)
    config = getattr(talker_model, "config", None)
    max_cache_len = getattr(talker_graph, "max_seq_len", None)
    prefill = getattr(talker_graph, "prefill_kv", None)
    if (
        not enabled
        or config is None
        or not isinstance(max_cache_len, int)
        or not callable(prefill)
    ):
        return False

    try:
        device = torch.device(model.device)
    except (AttributeError, TypeError, RuntimeError):
        return False
    cache = create_graph_quantized_kv_cache(
        config=config,
        device=device,
        max_cache_len=max_cache_len,
        enabled=enabled,
    )
    if cache is None:
        return False

    original_prefill = prefill
    talker_graph.static_cache = cache

    def copy_quantized_prefill(past_key_values):
        """Transfer Celune's quantized prompt into the graph cache directly."""
        if isinstance(past_key_values, QuantizedKVCache):
            return cache.copy_from(past_key_values)
        cache.reset()
        return original_prefill(past_key_values)

    talker_graph.prefill_kv = copy_quantized_prefill
    return True


class Qwen3(CeluneBackend[FasterQwen3TTS]):
    """Celune Qwen3-TTS backend."""

    name: str = "qwen3"

    uses_voice_bundles: bool = True
    chunk_rate: float = 12.5
    max_new_tokens: int = 512

    # setting this parameter will lock in identity, but expression may be reduced
    x_vector_only: bool = True
    supported_languages: tuple[str, ...] = (
        "zh-cn",
        "en",
        "ja",
        "ko",
        "de",
        "fr",
        "ru",
        "pt",
        "es",
        "it",
    )

    # this may be reassigned to a different size model as needed
    # low VRAM presets use 0.6B model
    # medium and above VRAM presets use 1.7B model
    clone_model: str = QWEN3_DEFAULT_MODEL_ID
    default_voice: Optional[str] = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        x_vector_only: bool = False,
        clone_model_id: Optional[str] = None,
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
        quantize_kv_cache: bool = True,
    ) -> None:
        super().__init__(
            log=log,
            fatal=fatal,
            quantize=quantize,
            quantize_kv_cache=quantize_kv_cache,
        )
        self.x_vector_only = x_vector_only
        self.model_name = clone_model_id or self.clone_model
        self._validate_refs()
        self.clone_model_id = clone_model_id or self.clone_model

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for the Qwen3 backend module."""
        return default_loader()

    def _validate_refs(self) -> None:
        """Validate Qwen3 reference audio files from the active CEVOICE/CECHAR pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def default_model_id(self) -> str:
        """Return the model loaded by default for Qwen3 cloning.

        Returns:
            str: The default Qwen3 model identifier.
        """
        return self.clone_model_id

    @property
    def all_model_ids(self) -> list[str]:
        """Return every model required by Qwen3 cloning.

        Returns:
            list[str]: The model identifiers needed by the backend.
        """
        return [self.clone_model_id]

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack.

        Returns:
            list[str]: The list of available voices to use from current CEVOICE/CECHAR pack.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice from the active pack to the shared Qwen3 clone model.

        Args:
            voice: The voice name to resolve.

        Returns:
            str: A resolved model name for this voice.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return self.clone_model_id
        _, voice_names = compatible_bundle
        custom_assert(
            voice in voice_names,
            ValueError(f"{self.name} cannot resolve a model for voice '{voice}'"),
        )
        assert voice in voice_names

        return self.clone_model_id

    def prepare_model_loading(self) -> None:
        """Import Qwen's lazy model dependencies before request execution."""
        from qwen_tts import Qwen3TTSModel

        del Qwen3TTSModel

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.
            lang: The language identifier for differentiating models by language.

        Returns:
            tuple[bool, Optional[str]]: A cache availability flag and the resolved snapshot path when present.
        """
        return cached_hf_snapshot_path(
            model,
            [
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "tokenizer_config.json",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> FasterQwen3TTS:
        """Load the given voice model.

        Args:
            model_id: The Qwen3 model repository ID to load.
            kwargs: Additional keyword arguments to use.

        Returns:
            FasterQwen3TTS: The loaded Qwen3 TTS model instance.
        """
        metadata = require_model_consent(model_id)
        available, path = self.model_is_available_locally(model_id)
        if not available or path is None:
            self.log(string("tts.model_download_start"), "info")
            with huggingface_progress(self.report_progress):
                path = snapshot_download(
                    repo_id=model_id,
                    revision=metadata.revision,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )
        configure_numba_cache()

        with local_hf_offline_mode(), huggingface_progress(self.report_progress):
            self.model = FasterQwen3TTS.from_pretrained(path)
        _install_graph_quantized_cache(self.model, self.quantize_kv_cache)
        self.model = self.apply_runtime_quantization(self.model, model_id)
        return self.model

    def generate_stream(
        self, model: FasterQwen3TTS, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded Qwen3 model instance.
            kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of Qwen3 streaming audio chunks.

        Raises:
            ValueError: The requested voice is unsupported, or input text is empty.
        """
        if not kwargs.get("text", None):
            raise ValueError("expected text to say")

        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        self._apply_seed()

        # if faster_qwen3_tts >= 0.2.5 use instructions, else remove this arg
        if tuple(int(num) for num in qwen3_ver.split(".")[:3]) < (0, 2, 5):
            kwargs.pop("instruct", None)

        voice = kwargs.pop("voice", self.default_voice)

        try:
            compatible_bundle = self._require_compatible_bundle()
            if compatible_bundle is None:
                return
            loader, _ = compatible_bundle
            ref_wav = self._truncate_reference(loader.materialize(voice, "wav"))
            configured_ref_text = loader.bundle.voices[voice].get("reference_text")
            ref_text = (
                configured_ref_text.strip()
                if isinstance(configured_ref_text, str)
                else ""
            )
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        stream = None
        first_chunk_time: Optional[float] = None
        cache_target = _quantized_talker_cache(model, self.quantize_kv_cache)
        original_talker_forward = None
        cache_injected = False
        try:
            stream_kwargs = {}
            if cache_target is not None:
                talker, cache = cache_target
                original_forward = talker.forward
                original_talker_forward = original_forward

                def forward_with_quantized_cache(*args, **call_kwargs):
                    """Supply a fresh compact cache to Qwen's dynamic decode path."""
                    nonlocal cache_injected
                    if call_kwargs.get("use_cache") and not cache_injected:
                        call_kwargs["past_key_values"] = cache
                        cache_injected = True
                    return original_forward(*args, **call_kwargs)

                talker.forward = forward_with_quantized_cache
                stream_kwargs["parity_mode"] = not isinstance(
                    getattr(getattr(model, "talker_graph", None), "static_cache", None),
                    GraphQuantizedKVCache,
                )
            stream = model.generate_voice_clone_streaming(
                ref_audio=ref_wav,
                ref_text=ref_text,
                non_streaming_mode=False,  # VERY IMPORTANT ON >=0.2.5
                xvec_only=self.x_vector_only,
                **stream_kwargs,
                **kwargs,
            )

            for chunk in stream:
                audio_chunk, sample_rate, timing = chunk
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                if timing is not None:
                    timing = dict(timing)
                    total_steps = timing.get("total_steps_so_far")
                    if timing.get("is_final") and isinstance(total_steps, int):
                        timing["missing_eos"] = total_steps >= self.max_new_tokens
                    timing["first_chunk_time"] = first_chunk_time
                else:
                    timing = {"first_chunk_time": first_chunk_time}
                yield audio_chunk, sample_rate, timing
        finally:
            if stream is not None and available("close", obj=stream):
                with contextlib.suppress(Exception):
                    stream.close()
            if cache_target is not None and original_talker_forward is not None:
                cache_target[0].forward = original_talker_forward
