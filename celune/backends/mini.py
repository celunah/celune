# SPDX-License-Identifier: MIT
"""Pocket TTS backend implementation for Celune."""

import tempfile
from pathlib import Path
from collections.abc import Iterator, Mapping
from typing import Callable, Optional, Final, Protocol, cast

import yaml
import torch
import numpy as np
import numpy.typing as npt
from pocket_tts import TTSModel
from huggingface_hub import snapshot_download

from .base import CeluneBackend, cached_hf_snapshot_path, BackendModel
from ..cevoice import default_loader
from ..exceptions import BackendError

type MiniPromptState = dict[str, dict[str, torch.Tensor]]


class MiniModel(Protocol):
    """Pocket TTS model surface used by Celune's mini backend."""

    sample_rate: int

    def get_state_for_audio_prompt(self, audio_conditioning: str) -> MiniPromptState:
        """Return a reusable prompt state for one reference audio path.

        Args:
            audio_conditioning: The audio conditioning string value.

        Raises:
            NotImplementedError: The protocol was called directly.
        """
        raise NotImplementedError("protocol not defined")

    def generate_audio_stream(
        self,
        model_state: MiniPromptState,
        text_to_generate: str,
    ) -> Iterator[torch.Tensor]:
        """Yield streamed audio chunks for one prompt state and text.

        Args:
            model_state: The current prompt state.
            text_to_generate: The text to be generated.

        Raises:
            NotImplementedError: The protocol was called directly.
        """
        raise NotImplementedError("protocol not defined")


class Mini(CeluneBackend):
    """Celune Mini (Pocket TTS) backend."""

    name: Final[str] = "mini"
    uses_voice_bundles: Final[bool] = True
    chunk_rate: Final[float] = 12.5
    supported_languages: Final[tuple[str, ...]] = ("en", "fr", "de", "it", "pt", "es")

    voice_models: Final[Mapping[str, str]] = {
        "balanced": "lunahr/pocket-tts-ungated",
        "calm": "lunahr/pocket-tts-ungated",
        "bold": "lunahr/pocket-tts-ungated",
        "upbeat": "lunahr/pocket-tts-ungated",
    }
    default_voice: Final[str] = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self._validate_refs()
        self._voice_states: dict[str, MiniPromptState] = {}
        self._generated_config_path: Optional[Path] = None

    @staticmethod
    def _resolve_language_name(lang: str = "en") -> str:
        """Return the Pocket TTS language variant expected for this backend."""
        code_to_model: Final[Mapping[str, str]] = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "es": "spanish",
        }

        return code_to_model[lang]

    def _resolve_snapshot_language_dir(self, snapshot_path: str) -> Path:
        """Return the model language directory from a local Pocket TTS snapshot."""
        language_name = self._resolve_language_name()
        language_dir = Path(snapshot_path) / "languages" / language_name
        if not language_dir.is_dir():
            raise BackendError(
                f"invalid Pocket TTS snapshot: languages/{language_name} not found"
            )
        return language_dir

    def _build_generated_config_path(self, snapshot_path: str) -> Path:
        """Create a temporary Pocket TTS YAML config targeting the snapshot files."""
        from pocket_tts.utils.config import CONFIGS_DIR

        language_name = self._resolve_language_name()
        template_path = CONFIGS_DIR / f"{language_name}.yaml"
        if not template_path.is_file():
            raise BackendError(
                f"invalid Pocket TTS snapshot: template config {template_path.name} not found"
            )

        language_dir = self._resolve_snapshot_language_dir(snapshot_path)
        model_path = language_dir / "model.safetensors"
        tokenizer_path = language_dir / "tokenizer.model"

        if not model_path.exists():
            raise BackendError(
                f"invalid Pocket TTS snapshot: {model_path.relative_to(snapshot_path)} not found"
            )
        if not tokenizer_path.exists():
            raise BackendError(
                f"invalid Pocket TTS snapshot: {tokenizer_path.relative_to(snapshot_path)} not found"
            )

        with open(template_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["weights_path"] = str(model_path)
        config["weights_path_without_voice_cloning"] = str(model_path)
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_path)

        temp_dir = Path(tempfile.gettempdir()) / "celune-pocket-tts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        generated_path = temp_dir / f"{language_name}-{Path(snapshot_path).name}.yaml"
        with open(generated_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return generated_path

    def _reference_voice_path(self, voice: str) -> Path:
        """Return the active reference WAV path for a Celune voice."""
        loader = default_loader()
        if loader is not None:
            return loader.materialize(voice, "wav")
        return self._reference_wave_path(voice)

    def _get_voice_state(self, model: MiniModel, voice: str) -> MiniPromptState:
        """Return a cached Pocket TTS voice state for the selected voice."""
        if voice in self._voice_states:
            return self._voice_states[voice]

        try:
            voice_path = self._reference_voice_path(voice)
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        voice_state = model.get_state_for_audio_prompt(str(voice_path))
        self._voice_states[voice] = voice_state
        return voice_state

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if a Pocket TTS model snapshot is already available locally.

        Args:
            model: The model ID to validate.

        Returns:
            tuple[bool, Optional[str]]: Whether a model by this ID is available, and any applicable path.
        """
        return cached_hf_snapshot_path(
            model,
            [
                "languages/english/model.safetensors",
                "languages/english/tokenizer.model",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> Optional[BackendModel]:
        """Load the configured Pocket TTS model snapshot.

        Args:
            model_id: The Pocket TTS model ID to load.
            kwargs: Additional keyword arguments to use while loading the model.

        Returns:
            Optional[BackendModel]: A Celune-compatible Pocket TTS model object.
        """
        available, snapshot_path = self.model_is_available_locally(model_id)
        if not available or snapshot_path is None:
            self.log("Downloading TTS model...", "info")
            snapshot_path = snapshot_download(repo_id=model_id)

        generated_config_path = self._build_generated_config_path(snapshot_path)
        self._generated_config_path = generated_config_path
        self.model = TTSModel.load_model(config=generated_config_path)
        self._voice_states.clear()
        return self.model

    def unload_model(self) -> None:
        """Release the loaded model and cached voice states."""
        self._voice_states.clear()
        self._generated_config_path = None
        super().unload_model()

    def generate_stream(
        self, model: BackendModel, **kwargs
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune-compatible audio chunks from Pocket TTS.

        Args:
            model: The model to perform inference with.
            kwargs: Keyword arguments to use for generation.

        Raises:
            ValueError: Received an empty input to be generated.
        """
        text = kwargs.pop("text", None)
        if not text:
            raise ValueError("expected text to say")

        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        chunk_size = kwargs.pop("chunk_size", 1)

        if instruct:
            text = f"({instruct}) {text}"

        self._apply_seed()
        mini_model = cast(MiniModel, model)
        voice_state = self._get_voice_state(mini_model, voice)
        chunks_per_batch = max(1, round(chunk_size * self.chunk_rate))

        batch: list[npt.NDArray[np.float32]] = []
        pending_audio: Optional[npt.NDArray[np.float32]] = None
        pending_steps = 0
        chunk_index = 0
        total_steps = 0

        for chunk in mini_model.generate_audio_stream(voice_state, text):
            chunk_array = chunk.detach().cpu().float().numpy()
            batch.append(chunk_array)

            if len(batch) < chunks_per_batch:
                continue

            if pending_audio is not None:
                total_steps += pending_steps
                yield (
                    pending_audio,
                    int(mini_model.sample_rate),
                    {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": pending_steps,
                        "total_steps_so_far": total_steps,
                        "is_final": False,
                    },
                )
                chunk_index += 1

            pending_audio = np.concatenate(batch)
            pending_steps = len(batch)
            batch.clear()

        if batch:
            if pending_audio is not None:
                total_steps += pending_steps
                yield (
                    pending_audio,
                    int(mini_model.sample_rate),
                    {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": pending_steps,
                        "total_steps_so_far": total_steps,
                        "is_final": False,
                    },
                )
                chunk_index += 1

            total_steps += len(batch)
            yield (
                np.concatenate(batch),
                int(mini_model.sample_rate),
                {
                    "backend": self.name,
                    "chunk_index": chunk_index,
                    "chunk_steps": len(batch),
                    "total_steps_so_far": total_steps,
                    "is_final": True,
                },
            )
        elif pending_audio is not None:
            total_steps += pending_steps
            yield (
                pending_audio,
                int(mini_model.sample_rate),
                {
                    "backend": self.name,
                    "chunk_index": chunk_index,
                    "chunk_steps": pending_steps,
                    "total_steps_so_far": total_steps,
                    "is_final": True,
                },
            )
