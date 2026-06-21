# SPDX-License-Identifier: MIT
"""Pocket TTS backend implementation for Celune."""

import tempfile
import contextlib
from pathlib import Path
from typing import Callable, Optional, cast
from collections.abc import Iterator, Mapping

import yaml
import numpy as np
import numpy.typing as npt
from pocket_tts import TTSModel
from huggingface_hub import snapshot_download

from ..paths import temp_data_dir
from ..utils import custom_assert
from ..cevoice import default_loader, CEVoiceLoader
from ..typing.backends import MiniModel, MiniPromptState
from .base import CeluneBackend, cached_hf_snapshot_path


class Mini(CeluneBackend[TTSModel]):
    """Celune Mini (Pocket TTS) backend."""

    name: str = "mini"
    uses_voice_bundles: bool = True
    chunk_rate: float = 12.5
    supported_languages: tuple[str, ...] = ("en", "fr", "de", "it", "pt", "es")

    voice_models: Optional[Mapping[str, str]] = {
        "balanced": "lunahr/pocket-tts-ungated",
        "calm": "lunahr/pocket-tts-ungated",
        "bold": "lunahr/pocket-tts-ungated",
        "upbeat": "lunahr/pocket-tts-ungated",
    }
    default_voice: Optional[str] = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self._validate_refs()
        self._voice_states: dict[str, MiniPromptState] = {}
        self._generated_config_path: Optional[Path] = None
        self._loaded_language = "en"

    @staticmethod
    def _require_compatible_bundle() -> tuple[CEVoiceLoader, tuple[str, ...]]:
        """Return the active CEVOICE/CECHAR loader and its usable voice names."""
        loader = default_loader()
        custom_assert(
            loader is not None,
            FileNotFoundError(
                "backend 'mini' requires a compatible CEVOICE/CECHAR package "
                "with at least one valid voice identifier"
            ),
        )
        assert loader is not None

        voice_names = tuple(
            voice
            for voice in loader.bundle.voice_order
            if (
                isinstance(voice, str)
                and voice.strip()
                and voice in loader.bundle.voices
                and isinstance(loader.bundle.voices[voice].get("reference_text"), str)
                and bool(str(loader.bundle.voices[voice]["reference_text"]).strip())
            )
        )
        custom_assert(
            bool(voice_names),
            FileNotFoundError(
                "backend 'mini' requires a compatible CEVOICE/CECHAR package "
                "with at least one valid voice identifier"
            ),
        )
        assert bool(voice_names)

        return loader, voice_names

    def _validate_refs(self) -> None:
        """Validate Mini reference audio files from the active CEVOICE/CECHAR pack."""
        loader, voice_names = self._require_compatible_bundle()
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack.

        Returns:
            list[str]: The list of available voices to use from current CEVOICE/CECHAR pack.
        """
        _, voice_names = self._require_compatible_bundle()
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice from the active pack to the shared Mini model.

        Args:
            voice: The voice name to resolve.

        Returns:
            str: A resolved model name for this voice.
        """
        _, voice_names = self._require_compatible_bundle()
        custom_assert(
            voice in voice_names,
            ValueError(f"{self.name} cannot resolve a model for voice '{voice}'"),
        )
        assert voice in voice_names
        return self.default_model_id

    def resolve_generation_language(self, lang: Optional[str]) -> str:
        """Normalize a requested language to one of Pocket TTS's supported variants.

        Args:
            lang: The language identifier for differentiating models by language.

        Returns:
            str: A language-specific model identifier, or ``"en"`` if no match was found.
        """
        alias_to_code: Mapping[str, str] = {
            "english": "en",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "spanish": "es",
        }
        fallback = "en"

        if not lang:
            return fallback

        normalized = lang.strip().lower().replace("_", "-")
        if not normalized or normalized == "auto":
            return fallback

        if normalized in alias_to_code:
            return alias_to_code[normalized]

        if normalized in self.supported_languages:
            return normalized

        if "-" in normalized:
            base = normalized.split("-", 1)[0]
            if base in self.supported_languages:
                return base

        return fallback

    def _resolve_language_name(self, lang: Optional[str] = "en") -> str:
        """Return the Pocket TTS language variant expected for this backend."""
        code_to_model: Mapping[str, str] = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "es": "spanish",
        }

        return code_to_model[self.resolve_generation_language(lang)]

    def _resolve_snapshot_language_dir(
        self, snapshot_path: str, lang: str = "en"
    ) -> Path:
        """Return the model language directory from a local Pocket TTS snapshot."""
        language_name = self._resolve_language_name(lang)
        languages_dir = Path(snapshot_path) / "languages"
        candidates = [languages_dir / language_name]
        candidates.extend(sorted(languages_dir.glob(f"{language_name}_*")))
        for language_dir in candidates:
            if language_dir.is_dir():
                return language_dir

        if not languages_dir.is_dir():
            raise FileNotFoundError(
                "invalid Pocket TTS snapshot: languages directory not found"
            )
        available = ", ".join(sorted(path.name for path in languages_dir.iterdir()))
        raise FileNotFoundError(
            f"invalid Pocket TTS snapshot: languages/{language_name} not found"
            + (f" (available: {available})" if available else "")
        )

    def _resolve_template_config_path(self, lang: str = "en") -> Path:
        """Return the best matching Pocket TTS template config for one language."""
        from pocket_tts.utils.config import CONFIGS_DIR

        language_code = self.resolve_generation_language(lang)
        language_name = self._resolve_language_name(lang)
        candidates = (
            CONFIGS_DIR / f"{language_name}.yaml",
            CONFIGS_DIR / f"{language_code}.yaml",
        )
        for template_path in candidates:
            if template_path.is_file():
                return template_path

        prefixed_matches = sorted(CONFIGS_DIR.glob(f"{language_name}_*.yaml"))
        if prefixed_matches:
            return prefixed_matches[0]

        code_matches = sorted(CONFIGS_DIR.glob(f"{language_code}_*.yaml"))
        if code_matches:
            return code_matches[0]

        raise FileNotFoundError(
            f"invalid Pocket TTS snapshot: template config for {language_name} not found"
        )

    resolve_language_name = _resolve_language_name
    resolve_snapshot_language_dir = _resolve_snapshot_language_dir
    resolve_template_config_path = _resolve_template_config_path

    def _build_generated_config_path(
        self, snapshot_path: str, lang: str = "en"
    ) -> Path:
        """Create a temporary Pocket TTS YAML config targeting the snapshot files."""
        language_name = self._resolve_language_name(lang)
        template_path = self._resolve_template_config_path(lang)
        language_dir = self._resolve_snapshot_language_dir(snapshot_path, lang)
        model_path = language_dir / "model.safetensors"
        tokenizer_path = language_dir / "tokenizer.model"

        if not model_path.exists():
            raise FileNotFoundError(
                f"invalid Pocket TTS snapshot: {model_path.relative_to(snapshot_path)} not found"
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"invalid Pocket TTS snapshot: {tokenizer_path.relative_to(snapshot_path)} not found"
            )

        with open(template_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["weights_path"] = str(model_path)
        config["weights_path_without_voice_cloning"] = str(model_path)
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_path)

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="celune-pocket-tts-",
                dir=str(temp_data_dir(create=True)),
            )
        )
        generated_path = temp_dir / f"{language_name}-{Path(snapshot_path).name}.yaml"
        with open(generated_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return generated_path

    def _reference_voice_path(self, voice: str) -> Path:
        """Return the active reference WAV path for a Celune voice."""
        loader, _ = self._require_compatible_bundle()
        return loader.materialize(voice, "wav")

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

        voice_state = model.get_state_for_audio_prompt(
            str(self._truncate_reference(voice_path))
        )
        self._voice_states[voice] = voice_state
        return voice_state

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = "en"
    ) -> tuple[bool, Optional[str]]:
        """Check if a Pocket TTS model snapshot is already available locally.

        Args:
            model: The model ID to validate.
            lang: The language identifier for differentiating models by language.

        Returns:
            tuple[bool, Optional[str]]: Whether a model by this ID is available, and any applicable path.
        """
        model_name = self._resolve_language_name(lang)

        return cached_hf_snapshot_path(
            model,
            [
                f"languages/{model_name}/model.safetensors",
                f"languages/{model_name}/tokenizer.model",
            ],
        )

    def should_reload_for_language(self, lang: Optional[str]) -> bool:
        """Return whether the loaded Pocket TTS language differs from ``lang``.

        Args:
            lang: The language identifier for differentiating models by language.

        Returns:
            bool: Whether Celune should reload a new Pocket TTS language model.
        """
        return self.resolve_generation_language(lang) != self._loaded_language

    def load_model(self, model_id: str, **kwargs) -> TTSModel:
        """Load the configured Pocket TTS model snapshot.

        Args:
            model_id: The Pocket TTS model ID to load.
            kwargs: Additional keyword arguments to use while loading the model.

        Returns:
            Optional[TTSModel]: A Celune-compatible Pocket TTS model object.
        """
        requested_language = self.resolve_generation_language(
            cast(Optional[str], kwargs.pop("lang", kwargs.pop("language", None)))
        )
        available, snapshot_path = self.model_is_available_locally(
            model_id, requested_language
        )
        if not available or snapshot_path is None:
            self.log("Downloading TTS model...", "info")
            snapshot_path = snapshot_download(repo_id=model_id)

        generated_config_path = self._build_generated_config_path(
            snapshot_path, requested_language
        )
        self._generated_config_path = generated_config_path
        self.model = TTSModel.load_model(config=generated_config_path)
        self._loaded_language = requested_language
        self._voice_states.clear()
        return self.model

    def unload_model(self) -> None:
        """Release the loaded model and cached voice states."""
        generated_config_path = self._generated_config_path
        self._voice_states.clear()
        self._generated_config_path = None
        if generated_config_path is not None:
            with contextlib.suppress(OSError):
                generated_config_path.unlink(missing_ok=True)
                generated_config_path.parent.rmdir()
        super().unload_model()

    def generate_stream(
        self, model: TTSModel, **kwargs
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
        kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        chunk_size = kwargs.pop("chunk_size", 1)

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
                    mini_model.sample_rate,
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
                    mini_model.sample_rate,
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
                mini_model.sample_rate,
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
                mini_model.sample_rate,
                {
                    "backend": self.name,
                    "chunk_index": chunk_index,
                    "chunk_steps": pending_steps,
                    "total_steps_so_far": total_steps,
                    "is_final": True,
                },
            )
