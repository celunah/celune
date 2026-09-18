# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 backend implementation for Celune."""

import os
import sys
import time
import shutil
import zipfile
import tempfile
import threading
import contextlib
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol, cast
from collections.abc import Callable, Iterator, Generator

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from transformers.cache_utils import Cache

from .base import (
    CeluneBackend,
    _to_numpy_audio,
    local_hf_offline_mode,
    cached_hf_snapshot_path,
)
from ...paths import (
    runtime_data_dir,
    huggingface_progress,
    huggingface_hub_cache_dir,
)
from ...i18n import string
from ...typing.aliases import AudioChunk
from ...typing.backends import BackendModel
from ...cevoice import CEVoiceLoader, default_loader

__all__ = ["FireRedTTS3"]

_FIRERED_ATTENTION_IMPLEMENTATION = "sdpa"


class _FireRedModel(BackendModel, Protocol):
    """Subset of FireRedTTS3 used by the Celune streaming adapter."""

    redae: "_FireRedRedAE"
    tts_core: "_FireRedCore"
    spk_extractor: "_FireRedSpeakerExtractor"
    device: torch.device

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize FireRed text on the model's device."""
        raise NotImplementedError("protocol not defined")


class _FireRedRedAE(Protocol):
    """RedAE methods required for incremental FireRed decoding."""

    sample_rate: int
    downsample_rate: int
    decoder: "_FireRedAudioDecoder"

    def pad_to_multiple_of(self, audio: torch.Tensor, multiple_of: int) -> torch.Tensor:
        """Pad audio to a RedAE-compatible length."""
        raise NotImplementedError("protocol not defined")

    def encode(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        """Encode prompt audio into RedAE latents."""
        raise NotImplementedError("protocol not defined")

    def decode(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Decode a complete prompt-plus-generated latent prefix."""
        raise NotImplementedError("protocol not defined")


class _FireRedSpeakerExtractor(Protocol):
    """Speaker encoder methods required by the FireRed generation path."""

    def forward(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        """Extract a speaker embedding from prompt audio."""
        raise NotImplementedError("protocol not defined")


class _FireRedCore(Protocol):
    """FireRed autoregressive core methods required for streaming."""

    patch_size: int

    def generate_stream(
        self,
        *,
        spk_emb: torch.Tensor,
        text_tokens: torch.Tensor,
        prompt_latents: torch.Tensor,
        n_timesteps: int,
        inference_cfg: float,
        stop_threshold: float,
        min_gen_steps: int,
        max_gen_steps: Optional[int],
    ) -> Iterator[torch.Tensor]:
        """Yield generated RedAE latent patches."""
        raise NotImplementedError("protocol not defined")


class _FireRedQwenOutput(Protocol):
    """Cached Qwen3 decoder output required by the incremental RedAE path."""

    last_hidden_state: torch.Tensor
    past_key_values: Optional["Cache"]


class _FireRedQwenConfig(Protocol):
    """Qwen3 decoder configuration values used by the latent projection."""

    hidden_size: int


class _FireRedQwenModel(Protocol):
    """Qwen3 decoder call surface used by RedAE."""

    config: _FireRedQwenConfig

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional["Cache"],
        use_cache: bool,
    ) -> _FireRedQwenOutput:
        """Decode embeddings while updating the KV cache."""
        raise NotImplementedError("protocol not defined")


class _FireRedISTFT(Protocol):
    """ISTFT configuration and window used by FireRed's decoder head."""

    window: torch.Tensor
    n_fft: int
    hop_length: int


class _FireRedISTFTHead(Protocol):
    """Spectral projection and ISTFT surface used by RedAE."""

    out: Callable[[torch.Tensor], torch.Tensor]
    istft: _FireRedISTFT


class _FireRedAudioDecoder(Protocol):
    """Decoder modules required for cached FireRed audio reconstruction."""

    in_proj: Callable[[torch.Tensor], torch.Tensor]
    qwen3: _FireRedQwenModel
    istft_head: _FireRedISTFTHead


class _FireRedIncrementalDecoder:
    """Decode FireRed latent batches without re-running the completed prefix."""

    def __init__(self, redae: _FireRedRedAE) -> None:
        """Initialize cached Qwen3 and overlap-add decoder state."""
        self._decoder = redae.decoder
        istft = self._decoder.istft_head.istft
        self._n_fft = istft.n_fft
        self._hop_length = istft.hop_length
        self._padding = (self._n_fft - self._hop_length) // 2
        self._window = istft.window
        self._cache: Optional[Cache] = None
        self._frame_count = 0
        self._raw_cursor = 0
        self._ola: Optional[torch.Tensor] = None
        self._envelope: Optional[torch.Tensor] = None

    def _decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent batches into windowed inverse-FFT frames."""
        hidden_size = self._decoder.qwen3.config.hidden_size
        embeddings = self._decoder.in_proj(latents)
        embeddings = embeddings.reshape(embeddings.shape[0], -1, hidden_size)
        outputs = self._decoder.qwen3(
            inputs_embeds=embeddings,
            attention_mask=None,
            past_key_values=self._cache,
            use_cache=True,
        )
        self._cache = outputs.past_key_values

        projected = self._decoder.istft_head.out(outputs.last_hidden_state).transpose(
            1, 2
        )
        magnitude, phase = projected.chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clip(max=1e2)
        spectrum = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        frames = torch.fft.irfft(
            spectrum,
            self._n_fft,
            dim=1,
            norm="backward",
        )
        window = self._window.to(device=frames.device, dtype=frames.dtype)
        return frames * window[None, :, None]

    @staticmethod
    def _fold_frames(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
        """Overlap-add a batch of frames using the upstream RedAE layout."""
        frame_count = frames.shape[-1]
        output_size = (frame_count - 1) * hop_length + frames.shape[1]
        return torch.nn.functional.fold(
            frames,
            output_size=(1, output_size),
            kernel_size=(1, frames.shape[1]),
            stride=(1, hop_length),
        )[:, 0, 0]

    def _append_frames(self, frames: torch.Tensor) -> None:
        """Add windowed frames and their normalization envelope to the buffer."""
        frame_count = frames.shape[-1]
        if frame_count == 0:
            return

        window = self._window.to(device=frames.device, dtype=frames.dtype)
        audio = self._fold_frames(frames, self._hop_length)
        envelope_frames = (
            window.square()
            .expand(
                frames.shape[0],
                frame_count,
                frames.shape[1],
            )
            .transpose(1, 2)
        )
        envelope = self._fold_frames(envelope_frames, self._hop_length)

        frame_start = self._frame_count * self._hop_length
        required_length = frame_start + audio.shape[-1]
        ola = self._ola
        stored_envelope = self._envelope
        if ola is None or stored_envelope is None:
            ola = audio.new_zeros(audio.shape[0], required_length)
            stored_envelope = envelope.new_zeros(
                envelope.shape[0],
                required_length,
            )
        elif ola.shape[-1] < required_length:
            extension = required_length - ola.shape[-1]
            ola = torch.cat([ola, audio.new_zeros(audio.shape[0], extension)], dim=-1)
            stored_envelope = torch.cat(
                [
                    stored_envelope,
                    envelope.new_zeros(envelope.shape[0], extension),
                ],
                dim=-1,
            )

        ola[:, frame_start:required_length] += audio
        stored_envelope[:, frame_start:required_length] += envelope
        self._ola = ola
        self._envelope = stored_envelope
        self._frame_count += frame_count

    def _drain(self, raw_end: int) -> tuple[Optional[AudioChunk], int]:
        """Return samples no future ISTFT frame can modify."""
        ola = self._ola
        envelope = self._envelope
        if ola is None or envelope is None:
            return None, 0

        end = min(raw_end, ola.shape[-1])
        start = max(self._raw_cursor, self._padding)
        if end <= start:
            return None, start - self._padding

        audio = ola[:, start:end] / envelope[:, start:end].clamp_min(1e-11)
        self._raw_cursor = end
        return _to_numpy_audio(audio), start - self._padding

    def push(self, latents: torch.Tensor) -> tuple[Optional[AudioChunk], int]:
        """Decode one latent batch and return its stable audio segment."""
        self._append_frames(self._decode_frames(latents))
        return self._drain(self._frame_count * self._hop_length)

    def finish(self) -> tuple[Optional[AudioChunk], int]:
        """Flush the final ISTFT tail after no more latent frames remain."""
        return self._drain(self._frame_count * self._hop_length + self._padding)


def _create_firered_model(
    model_root: str,
    *,
    log: Optional[Callable[[str, str], None]] = None,
    report_progress: Optional[
        Callable[[Optional[float], Optional[float]], None]
    ] = None,
) -> _FireRedModel:
    """Construct FireRedTTS3 with Celune's supported attention path."""
    from transformers import Qwen3Config

    from fireredtts3.llm import fireredtts3_base
    from fireredtts3.redae import redae as redae_module
    from fireredtts3.core import FireRedTTS3 as FireRedTTS3Model

    def announce(message_key: str, step: int) -> None:
        """Publish one visible model-construction stage."""
        if log is not None:
            log(string(message_key), "info")
        if report_progress is not None:
            report_progress(step, 4)

    class BFloat16RedAE(redae_module.RedAE):
        """Load RedAE checkpoints in BF16 for the Celune worker."""

        @classmethod
        def from_pretrained(cls, pretrained_model_dir: str) -> "BFloat16RedAE":
            announce("fireredtts3.loading_redae", 1)
            model = super().from_pretrained(
                pretrained_model_dir,
                dtype=torch.bfloat16,
            )
            announce("fireredtts3.redae_loaded", 2)
            return cast(
                "BFloat16RedAE",
                model,
            )

    class BFloat16FireRedTTS3BaseCore(fireredtts3_base.FireRedTTS3BaseCore):
        """Load FireRedTTS3 transformer checkpoints in BF16 and stream latents."""

        @classmethod
        def from_pretrained(
            cls, pretrained_model_dir: str
        ) -> "BFloat16FireRedTTS3BaseCore":
            announce("fireredtts3.loading_core", 2)
            model = super().from_pretrained(
                pretrained_model_dir,
                dtype=torch.bfloat16,
            )
            announce("fireredtts3.core_loaded", 3)
            return cast(
                "BFloat16FireRedTTS3BaseCore",
                model,
            )

        def generate_stream(
            self,
            *,
            spk_emb: torch.Tensor,
            text_tokens: torch.Tensor,
            prompt_latents: torch.Tensor,
            n_timesteps: int = 10,
            inference_cfg: float = 2.0,
            stop_threshold: float = 0.5,
            min_gen_steps: int = 6,
            max_gen_steps: Optional[int] = None,
        ) -> Iterator[torch.Tensor]:
            """Yield each FireRed latent patch as soon as autoregression creates it."""
            with torch.inference_mode():
                device = text_tokens.device

                input_embeds = self.backbone_llm.embed_tokens(text_tokens)
                patch_prompt_latents = self.patch_encoder(prompt_latents)
                spk_embs_llm = self.spk_proj_llm(spk_emb)
                input_embeds = torch.cat(
                    [
                        spk_embs_llm.unsqueeze(1),
                        input_embeds,
                        patch_prompt_latents,
                    ],
                    dim=1,
                )

                t_span = torch.linspace(0, 1, n_timesteps + 1).to(device)
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
                latents_gen = fireredtts3_base.F.pad(
                    prompt_latents,
                    (0, 0, self.history_length, 0),
                )
                dit_spk_cond = self.spk_proj_dit(spk_emb)
                backbone_cond = input_embeds.new_zeros(
                    1,
                    self.history_patches,
                    input_embeds.shape[-1],
                )
                backbone_cache = None

                max_gen_steps = 400 if max_gen_steps is None else max_gen_steps
                for step_index in range(max_gen_steps):
                    backbone_out, backbone_cache = self._backbone_one_step(
                        input_embeds,
                        cache=backbone_cache,
                    )
                    stop_logits = self.stop_head(backbone_out[:, -1]).squeeze(-1)
                    stop_score = torch.sigmoid(stop_logits).item()
                    if stop_score >= stop_threshold:
                        if min_gen_steps is not None:
                            if step_index >= min_gen_steps:
                                break
                        else:
                            break

                    if step_index == 0:
                        one_backbone_out = backbone_out[
                            :, -patch_prompt_latents.shape[1] :
                        ]
                    else:
                        one_backbone_out = backbone_out[:, -1:]
                    backbone_cond = torch.cat(
                        [backbone_cond, one_backbone_out],
                        dim=1,
                    )

                    one_latents = self._flow_one_step(
                        hist_latents=latents_gen[:, -self.history_length :],
                        backbone_cond=self.dit_head(
                            backbone_cond[:, -(self.history_patches + 1) :]
                        ),
                        spk_cond=dit_spk_cond,
                        t_span=t_span,
                        inference_cfg=inference_cfg,
                    )
                    input_embeds = self.patch_encoder(one_latents)
                    latents_gen = torch.cat([latents_gen, one_latents], dim=1)
                    yield one_latents

    original_attention = fireredtts3_base.Qwen3_1_7B_ConfigDict["attn_implementation"]
    original_redae = fireredtts3_base.RedAE
    original_core = fireredtts3_base.FireRedTTS3BaseCore
    original_qwen3_config = redae_module.Qwen3Config
    qwen3_config_factory = cast(Callable[..., Qwen3Config], original_qwen3_config)

    def sdpa_qwen3_config(*args: object, **kwargs: object) -> Qwen3Config:
        """Replace FireRed's unsupported FA2 config with PyTorch SDPA."""
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = _FIRERED_ATTENTION_IMPLEMENTATION
        return qwen3_config_factory(*args, **kwargs)

    fireredtts3_base.Qwen3_1_7B_ConfigDict["attn_implementation"] = (
        _FIRERED_ATTENTION_IMPLEMENTATION
    )
    fireredtts3_base.RedAE = BFloat16RedAE
    fireredtts3_base.FireRedTTS3BaseCore = BFloat16FireRedTTS3BaseCore
    redae_module.Qwen3Config = sdpa_qwen3_config
    try:
        announce("fireredtts3.model_loading", 0)
        return cast(
            _FireRedModel,
            FireRedTTS3Model(
                model_root,
                use_fasttext=False,
                use_llm_tn=False,
                use_wetext=False,
            ),
        )
    finally:
        fireredtts3_base.Qwen3_1_7B_ConfigDict["attn_implementation"] = (
            original_attention
        )
        fireredtts3_base.RedAE = original_redae
        fireredtts3_base.FireRedTTS3BaseCore = original_core
        redae_module.Qwen3Config = original_qwen3_config


class FireRedTTS3(CeluneBackend[_FireRedModel]):
    """Celune FireRedTTS3 zero-shot voice-cloning backend."""

    name: str = "fireredtts3"
    uses_voice_bundles: bool = True
    # 4 RedAE frames @ 25 Hz
    chunk_rate: float = 6.25
    max_new_tokens: int = 400
    default_voice: Optional[str] = "balanced"
    model_repo: str = "FireRedTeam/FireRedTTS3"
    supported_languages: tuple[str, ...] = (
        "ar",
        "yue",
        "zh-cn",
        "cs",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "pl",
        "pt",
        "ro",
        "ru",
        "es",
        "th",
        "tr",
        "uk",
        "vi",
        "ZH_Anhui",
        "ZH_Fujian",
        "ZH_Gansu",
        "ZH_Guizhou",
        "ZH_Hebei",
        "ZH_Henan",
        "ZH_Hubei",
        "ZH_Hunan",
        "ZH_Jiangxi",
        "ZH_Liaoning",
        "ZH_Minnan",
        "ZH_Ningxia",
        "ZH_Shaanxi",
        "ZH_Shandong",
        "ZH_Shanghai",
        "ZH_Shanxi",
        "ZH_Sichuan",
        "ZH_Tianjin",
        "ZH_Wenzhou",
        "ZH_Wu",
        "ZH_Yunnan",
    )

    _source_archive_url: str = (
        "https://github.com/FireRedTeam/FireRedTTS3/archive/refs/heads/main.zip"
    )
    _managed_source_dir_name: str = "fireredtts3"
    _source_download_lock = threading.Lock()
    _model_files: tuple[str, ...] = (
        "redae/config.json",
        "redae/model.safetensors",
        "fireredtts3_base/config.json",
        "fireredtts3_base/model.safetensors",
        "campp/campplus_voxceleb.bin",
        "text_tokenizer/tokenizer.json",
        "text_tokenizer/tokenizer_config.json",
        "text_tokenizer/vocab.json",
    )
    _language_aliases: ClassVar[dict[str, str]] = {
        "arabic": "Arabic",
        "cantonese": "Cantonese",
        "chinese": "Chinese",
        "czech": "Czech",
        "dutch": "Dutch",
        "english": "English",
        "finnish": "Finnish",
        "french": "French",
        "german": "German",
        "greek": "Greek",
        "hindi": "Hindi",
        "indonesian": "Indonesian",
        "italian": "Italian",
        "japanese": "Japanese",
        "korean": "Korean",
        "polish": "Polish",
        "portuguese": "Portuguese",
        "romanian": "Romanian",
        "russian": "Russian",
        "spanish": "Spanish",
        "thai": "Thai",
        "turkish": "Turkish",
        "ukrainian": "Ukrainian",
        "vietnamese": "Vietnamese",
        "ar": "Arabic",
        "yue": "Cantonese",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "cs": "Czech",
        "nl": "Dutch",
        "en": "English",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "hi": "Hindi",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "es": "Spanish",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
    }
    _dialect_aliases: ClassVar[dict[str, str]] = {
        "zh-anhui": "ZH_Anhui",
        "zh-fujian": "ZH_Fujian",
        "zh-gansu": "ZH_Gansu",
        "zh-guizhou": "ZH_Guizhou",
        "zh-hebei": "ZH_Hebei",
        "zh-henan": "ZH_Henan",
        "zh-hubei": "ZH_Hubei",
        "zh-hunan": "ZH_Hunan",
        "zh-jiangxi": "ZH_Jiangxi",
        "zh-liaoning": "ZH_Liaoning",
        "zh-minnan": "ZH_Minnan",
        "zh-ningxia": "ZH_Ningxia",
        "zh-shaanxi": "ZH_Shaanxi",
        "zh-shandong": "ZH_Shandong",
        "zh-shanghai": "ZH_Shanghai",
        "zh-shanxi": "ZH_Shanxi",
        "zh-sichuan": "ZH_Sichuan",
        "zh-tianjin": "ZH_Tianjin",
        "zh-wenzhou": "ZH_Wenzhou",
        "zh-wu": "ZH_Wu",
        "zh-yunnan": "ZH_Yunnan",
    }

    def __init__(
        self,
        log: Callable[[str, str], None],
        model_id: Optional[str] = None,
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(
            log=log,
            model_name=model_id or self.model_repo,
            fatal=fatal,
        )
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for FireRedTTS3."""
        return default_loader()

    def _validate_refs(self) -> None:
        """Validate FireRedTTS3 reference audio files in the active voice pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice from the active pack to the shared FireRedTTS3 model."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return self.default_model_id
        _, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(string("celune.unknown_voice", voice=voice))
        return self.default_model_id

    def resolve_generation_language(self, lang: Optional[str]) -> Optional[str]:
        """Normalize Celune language identifiers to FireRedTTS3 language tags."""
        if lang is None:
            return None
        normalized = lang.strip().lower().replace("_", "-")
        if not normalized or normalized == "auto":
            return None
        if normalized in self._language_aliases:
            return self._language_aliases[normalized]
        if normalized in self._dialect_aliases:
            return self._dialect_aliases[normalized]
        base_language = normalized.split("-", 1)[0]
        return self._language_aliases.get(base_language)

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check whether the FireRedTTS3 model snapshot has all required files."""
        del lang
        return cached_hf_snapshot_path(model, list(self._model_files))

    @classmethod
    def _managed_source_root(cls, create: bool = False) -> Path:
        """Return the Celune-managed FireRedTTS3 source directory."""
        return runtime_data_dir(create=create) / cls._managed_source_dir_name

    @staticmethod
    def _source_is_available(root: Path) -> bool:
        """Return whether a source tree contains the FireRedTTS3 Python API."""
        return (root / "fireredtts3/core.py").is_file()

    @classmethod
    def _download_source_tree(cls, destination: Path) -> Path:
        """Download the official source-only FireRedTTS3 repository."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.partial")
        with cls._source_download_lock:
            if cls._source_is_available(destination):
                return destination
            if partial.exists():
                shutil.rmtree(partial)

            try:
                with tempfile.TemporaryDirectory(
                    prefix="fireredtts3-", dir=str(destination.parent)
                ) as temp_dir:
                    temp_root = Path(temp_dir)
                    archive_path = temp_root / "source.zip"
                    request = urllib.request.Request(
                        cls._source_archive_url,
                        headers={"User-Agent": "Celune"},
                    )
                    with (
                        urllib.request.urlopen(request, timeout=120) as response,
                        archive_path.open("wb") as archive_file,
                    ):
                        shutil.copyfileobj(response, archive_file)

                    extraction_root = temp_root / "extracted"
                    extraction_root.mkdir()
                    with zipfile.ZipFile(archive_path) as archive:
                        extraction_root_resolved = extraction_root.resolve()
                        for member in archive.infolist():
                            member_path = (extraction_root / member.filename).resolve()
                            if (
                                member_path != extraction_root_resolved
                                and extraction_root_resolved not in member_path.parents
                            ):
                                raise RuntimeError(
                                    string("fireredtts3.source_archive_invalid")
                                )
                        archive.extractall(extraction_root)

                    source_root = next(
                        (
                            candidate
                            for candidate in extraction_root.iterdir()
                            if cls._source_is_available(candidate)
                        ),
                        None,
                    )
                    if source_root is None:
                        raise RuntimeError(string("fireredtts3.source_archive_invalid"))
                    shutil.copytree(source_root, partial)

                if destination.exists():
                    shutil.rmtree(destination)
                partial.replace(destination)
            except Exception as error:
                with contextlib.suppress(OSError):
                    shutil.rmtree(partial)
                if isinstance(error, RuntimeError) and str(error) == string(
                    "fireredtts3.source_archive_invalid"
                ):
                    raise
                raise RuntimeError(
                    string("fireredtts3.source_download_failed")
                ) from error

        return destination

    @classmethod
    def _ensure_source_root(cls) -> Path:
        """Return a usable local FireRedTTS3 source tree."""
        destination = cls._managed_source_root(create=True)
        if cls._source_is_available(destination):
            return destination
        return cls._download_source_tree(destination)

    @staticmethod
    @contextlib.contextmanager
    def _source_context(root: Path) -> Generator[None, None, None]:
        """Expose the source-only FireRedTTS3 package to Python imports."""
        source_path = str(root)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        yield

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress third-party stdout/stderr while preserving CEDTS diagnostics."""
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            yield

    suppress_backend_output = _suppress_backend_output

    @staticmethod
    def _preload_source_modules() -> None:
        """Import FireRedTTS3 dependencies before the CEDTS read loop starts."""
        from transformers import Qwen3Config

        from fireredtts3.llm import fireredtts3_base
        from fireredtts3.redae import redae as redae_module
        from fireredtts3.core import FireRedTTS3 as FireRedTTS3Model

        _ = (Qwen3Config, fireredtts3_base, FireRedTTS3Model, redae_module)

    @staticmethod
    def _configure_bfloat16(model: _FireRedModel) -> _FireRedModel:
        """Use BF16 for FireRedTTS3's memory-heavy transformer components."""
        tts_core = getattr(model, "tts_core", None)
        redae = getattr(model, "redae", None)
        components = (
            ("tts_core.backbone_llm", getattr(tts_core, "backbone_llm", None)),
            ("tts_core.stop_head", getattr(tts_core, "stop_head", None)),
            ("redae.encoder", getattr(redae, "encoder", None)),
        )
        for component_name, component in components:
            to = getattr(component, "to", None)
            if not callable(to):
                raise TypeError(
                    f"FireRedTTS3 did not expose the expected {component_name} module"
                )
            to(dtype=torch.bfloat16)
        return model

    @staticmethod
    def _stream_latents(
        model: _FireRedModel,
        *,
        language: str,
        prompt_text: str,
        prompt_audio: torch.Tensor,
        prompt_audio_sr: int,
        text: str,
        stop_threshold: float,
        n_timesteps: int,
        inference_cfg: float,
        seed: Optional[int],
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, int]]:
        """Yield FireRed latent patches with prompt trimming metadata."""
        with torch.inference_mode():
            input_text = f"<|{language}|><|sot|>{prompt_text}{text}<|eot|>"
            text_tokens = model._tokenize_text(input_text)

            prompt_audio = prompt_audio[:1]
            prompt_audio = torchaudio.functional.resample(
                prompt_audio,
                prompt_audio_sr,
                model.redae.sample_rate,
            )
            prompt_audio = model.redae.pad_to_multiple_of(
                prompt_audio,
                model.redae.downsample_rate * model.tts_core.patch_size,
            )
            prompt_audio = prompt_audio.to(model.device)
            prompt_latents = model.redae.encode(
                prompt_audio,
                model.redae.sample_rate,
            ).to(torch.float32)
            spk_emb = model.spk_extractor.forward(
                prompt_audio,
                model.redae.sample_rate,
            ).to(model.device)

            if seed is not None:
                torch.manual_seed(seed)

            for latent_patch in model.tts_core.generate_stream(
                spk_emb=spk_emb,
                text_tokens=text_tokens,
                prompt_latents=prompt_latents,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                stop_threshold=stop_threshold,
                min_gen_steps=6,
                max_gen_steps=None,
            ):
                yield (
                    prompt_latents,
                    latent_patch,
                    int(prompt_audio.shape[1]),
                    int(model.redae.sample_rate),
                )

    def prepare_model_loading(self) -> None:
        """Prepare the source-only FireRedTTS3 package before CEDTS readiness."""
        source_root = self._ensure_source_root()
        with (
            self._source_context(source_root),
            self._suppress_backend_output(),
        ):
            self._preload_source_modules()

    def load_model(self, model_id: str, **kwargs) -> _FireRedModel:
        """Load FireRedTTS3 model assets and configure its BF16 path."""
        del kwargs
        available, path = self.model_is_available_locally(model_id)
        if not available or path is None:
            self.log(string("tts.model_download_start"), "info")
            with huggingface_progress(self.report_progress):
                path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )

        source_root = self._ensure_source_root()
        self.report_progress(0, 4)
        with (
            local_hf_offline_mode() if available else contextlib.nullcontext(),
            self._source_context(source_root),
            self._suppress_backend_output(),
        ):
            model = _create_firered_model(
                path,
                log=self.log,
                report_progress=self.report_progress,
            )
        self.log(string("fireredtts3.finalizing_model"), "info")
        self.report_progress(3, 4)
        model = self._configure_bfloat16(model)
        self.report_progress(4, 4)
        self.log(string("fireredtts3.model_ready"), "info")
        return model

    def generate_stream(
        self, model: _FireRedModel, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate progressive Celune-compatible audio chunks with FireRedTTS3."""
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(string("tts.text_required"))

        voice = kwargs.pop("voice", self.default_voice)
        if not isinstance(voice, str):
            raise TypeError(string("celune.unknown_voice", voice=voice))
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(string("celune.unknown_voice", voice=voice))

        # FireRed consumes the prompt transcript together with the complete
        # reference waveform. Truncating only the waveform leaves part of the
        # transcript unmatched, so the model can regenerate that prompt text
        # before it reaches the requested utterance.
        reference_wav = loader.materialize(voice, "wav")
        reference_text = loader.bundle.voices[voice].get("reference_text")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError(string("celune.unknown_voice", voice=voice))
        prompt_audio, prompt_audio_sr = torchaudio.load(str(reference_wav))

        language = self.resolve_generation_language(kwargs.pop("language", None))
        chunk_size = max(1, int(kwargs.pop("chunk_size", 1)))
        kwargs.pop("instruct", None)
        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        kwargs.pop("repetition_penalty", None)
        stop_threshold = float(kwargs.pop("stop_threshold", 0.5))
        n_timesteps = int(kwargs.pop("n_timesteps", 10))
        inference_cfg = float(kwargs.pop("inference_cfg", 2.0))

        self._apply_seed()
        seed_value = kwargs.pop("seed", self.current_seed)
        seed = cast(Optional[int], seed_value)
        chunks_per_batch = chunk_size
        pending_audio: Optional[AudioChunk] = None
        pending_steps = 0
        chunk_index = 0
        total_steps = 0
        first_chunk_time: Optional[float] = None

        def stream_decoded_audio() -> Iterator[tuple[AudioChunk, int, int]]:
            """Yield generated audio using one cached incremental decoder."""
            decoder: Optional[_FireRedIncrementalDecoder] = None
            prompt_samples = 0
            sample_rate = 0
            audio_batch: list[AudioChunk] = []

            for (
                prompt_latents,
                latent_patch,
                prompt_samples,
                sample_rate,
            ) in self._stream_latents(
                model,
                language=language or "English",
                prompt_text=reference_text.strip(),
                prompt_audio=prompt_audio,
                prompt_audio_sr=int(prompt_audio_sr),
                text=text,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
            ):
                if decoder is None:
                    decoder = _FireRedIncrementalDecoder(model.redae)
                    decoder.push(prompt_latents)

                audio, audio_start = decoder.push(latent_patch)
                if audio is None:
                    continue
                trim = max(0, prompt_samples - audio_start)
                if trim >= audio.shape[-1]:
                    continue
                audio_batch.append(audio[trim:])
                if len(audio_batch) < chunks_per_batch:
                    continue
                yield np.concatenate(audio_batch), sample_rate, len(audio_batch)
                audio_batch.clear()

            if decoder is not None:
                audio, audio_start = decoder.finish()
                if audio is not None:
                    trim = max(0, prompt_samples - audio_start)
                    if trim < audio.shape[-1]:
                        audio_batch.append(audio[trim:])

            if audio_batch:
                yield np.concatenate(audio_batch), sample_rate, len(audio_batch)

        with (
            self._suppress_backend_output(),
            # "I only speak BF16. Do not give me FP32." - she says
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for audio, sample_rate, batch_steps in stream_decoded_audio():
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                if pending_audio is not None:
                    total_steps += pending_steps
                    yield (
                        pending_audio,
                        int(sample_rate),
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": pending_steps,
                            "total_steps_so_far": total_steps,
                            "first_chunk_time": first_chunk_time,
                            "is_final": False,
                        },
                    )
                    chunk_index += 1
                pending_audio = audio
                pending_steps = batch_steps

            if pending_audio is not None:
                total_steps += pending_steps
                yield (
                    pending_audio,
                    int(sample_rate),
                    {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": pending_steps,
                        "total_steps_so_far": total_steps,
                        "first_chunk_time": first_chunk_time,
                        "is_final": True,
                    },
                )
