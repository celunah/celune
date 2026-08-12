# SPDX-License-Identifier: MIT
"""GPT-SoVITS backend implementation for Celune."""

import contextlib
import hashlib
import importlib
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable, Generator, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Optional, Union, cast

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from ...cevoice import CEVoiceLoader, default_loader
from ...i18n import string
from ...paths import (
    huggingface_hub_cache_dir,
    project_root,
    runtime_data_dir,
)
from ...typing.aliases import AudioChunk, AudioChunkNonNormalized, RuntimeValue
from ...typing.backends import GPTSoVITSPipeline, _GPTSoVITSConfig
from ...typing.common import JSONSerializable
from ...utils import custom_assert
from .base import CeluneBackend


class _GPTSoVITSRuntime:
    """Celune-owned wrapper around one official GPT-SoVITS pipeline."""

    def __init__(
        self,
        pipeline: GPTSoVITSPipeline,
        variant: str,
        sample_rate: int,
    ) -> None:
        self.pipeline: Optional[GPTSoVITSPipeline] = pipeline
        self.variant = variant
        self.sample_rate = sample_rate

    def close(self) -> None:
        """Stop GPT-SoVITS and release the pipeline reference."""
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is not None:
            with contextlib.suppress(Exception):
                pipeline.stop()


class GPTSoVITS(CeluneBackend[_GPTSoVITSRuntime]):
    """Celune GPT-SoVITS few-shot voice-cloning backend."""

    name: str = "gpt-sovits"
    uses_voice_bundles: bool = True
    chunk_rate: float = 6.25
    max_new_tokens: int = 512
    supported_languages: tuple[str, ...] = ("zh", "en", "ja", "ko", "yue")
    default_voice: Optional[str] = "balanced"

    _source_archive_url: str = (
        "https://github.com/RVC-Boss/GPT-SoVITS/archive/refs/heads/main.zip"
    )
    _model_repo_id: str = "lj1995/GPT-SoVITS"
    _managed_source_dir_name: str = "gpt_sovits"
    _download_lock = threading.Lock()
    _source_context_lock = threading.RLock()
    _model_snapshot_patterns: tuple[str, ...] = (
        "chinese-hubert-base/**",
        "chinese-roberta-wwm-ext-large/**",
        "gsv-v2final-pretrained/**",
        "gsv-v4-pretrained/**",
        "models--nvidia--bigvgan_v2_24khz_100band_256x/**",
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "s1v3.ckpt",
        "s2G488k.pth",
        "s2Gv3.pth",
        "sv/**",
        "v2Pro/**",
    )
    _cached_model_directories: tuple[str, ...] = (
        "chinese-hubert-base",
        "chinese-roberta-wwm-ext-large",
        "gsv-v2final-pretrained",
        "gsv-v4-pretrained",
        "models--nvidia--bigvgan_v2_24khz_100band_256x",
        "sv",
        "v2Pro",
    )
    _nltk_resources: Mapping[str, tuple[str, ...]] = {
        "averaged_perceptron_tagger": (
            "taggers/averaged_perceptron_tagger.zip",
            "taggers/averaged_perceptron_tagger/",
        ),
        "averaged_perceptron_tagger_eng": (
            "taggers/averaged_perceptron_tagger_eng/",
            "taggers/averaged_perceptron_tagger_eng.zip",
        ),
        "cmudict": ("corpora/cmudict.zip", "corpora/cmudict/"),
    }
    _fast_langdetect_model_name: str = "lid.176.bin"
    _reference_frame_ms: int = 20
    _reference_silence_threshold: float = 0.01
    _reference_silence_padding_ms: int = 80

    _variant_order: tuple[str, ...] = (
        "v2ProPlus",
        "v2Pro",
        "v4",
        "v3",
    )
    _fragment_streaming_variants: frozenset[str] = frozenset(("v3", "v4"))
    _variant_aliases: Mapping[str, str] = {
        "v2pro": "v2Pro",
        "v2proplus": "v2ProPlus",
        "v3": "v3",
        "v4": "v4",
    }
    _variant_files: Mapping[str, tuple[str, ...]] = {
        "v2Pro": (
            "s1v3.ckpt",
            "v2Pro/s2Gv2Pro.pth",
            "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        ),
        "v2ProPlus": (
            "s1v3.ckpt",
            "v2Pro/s2Gv2ProPlus.pth",
            "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        ),
        "v3": (
            "s1v3.ckpt",
            "s2Gv3.pth",
            "models--nvidia--bigvgan_v2_24khz_100band_256x/config.json",
        ),
        "v4": (
            "s1v3.ckpt",
            "gsv-v4-pretrained/s2Gv4.pth",
            "gsv-v4-pretrained/vocoder.pth",
        ),
    }
    _common_files: tuple[str, ...] = (
        "chinese-hubert-base/config.json",
        "chinese-roberta-wwm-ext-large/config.json",
    )

    def __init__(
        self,
        log: Callable[[str, str], None],
        root: Optional[str] = None,
        variant: Optional[str] = None,
        t2s_weights_path: Optional[str] = None,
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        self.root = self._resolve_root(root)
        requested_variant = self._normalize_variant(variant)
        self._requested_variant = requested_variant
        self._custom_t2s_weights_path = self._resolve_custom_t2s_weights_path(
            t2s_weights_path
        )
        self._model_snapshot: Optional[Path] = None
        self.variant = requested_variant or self._variant_order[0]
        super().__init__(log=log, model_name=self.variant, fatal=fatal)
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for GPT-SoVITS."""
        return default_loader()

    @staticmethod
    def _normalize_variant(variant: Optional[str]) -> Optional[str]:
        """Normalize a configured GPT-SoVITS variant name."""
        if variant is None or not variant.strip() or variant.strip().lower() == "auto":
            return None
        try:
            return GPTSoVITS._variant_aliases[variant.strip().lower()]
        except KeyError as error:
            choices = ", ".join(GPTSoVITS._variant_order)
            raise ValueError(
                string(
                    "gpt_sovits.unknown_variant",
                    variant=variant,
                    available=choices,
                )
            ) from error

    @staticmethod
    def _resolve_custom_t2s_weights_path(
        path: Optional[str],
    ) -> Optional[Path]:
        """Resolve an optional custom text-to-semantic checkpoint path."""
        if path is None or not path.strip():
            return None
        return Path(path).expanduser().resolve()

    @classmethod
    def _managed_source_root(cls, create: bool = False) -> Path:
        """Return the Celune-managed GPT-SoVITS source directory."""
        return runtime_data_dir(create=create) / cls._managed_source_dir_name

    @staticmethod
    def _source_is_available(root: Path) -> bool:
        """Return whether a GPT-SoVITS source checkout has the inference API."""
        return (root / "GPT_SoVITS/TTS_infer_pack/TTS.py").is_file()

    @classmethod
    def _download_source_tree(cls, destination: Path) -> Path:
        """Download the official GPT-SoVITS source into a managed directory."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.partial")
        with cls._download_lock:
            if cls._source_is_available(destination):
                return destination
            if partial.exists():
                shutil.rmtree(partial)

            try:
                with tempfile.TemporaryDirectory(
                    prefix="gpt-sovits-", dir=str(destination.parent)
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
                                    string("gpt_sovits.source_archive_invalid")
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
                        raise RuntimeError(string("gpt_sovits.source_archive_invalid"))
                    shutil.copytree(source_root, partial)

                if destination.exists():
                    shutil.rmtree(destination)
                partial.replace(destination)
            except Exception as error:
                with contextlib.suppress(OSError):
                    shutil.rmtree(partial)
                if isinstance(error, RuntimeError) and str(error) == string(
                    "gpt_sovits.source_archive_invalid"
                ):
                    raise
                raise RuntimeError(
                    string("gpt_sovits.source_download_failed", error=str(error))
                ) from error

        return destination

    @staticmethod
    def _candidate_roots(configured_root: Optional[str]) -> Iterator[Path]:
        """Yield possible GPT-SoVITS source roots in preference order."""
        candidates: list[Path] = []
        if configured_root is not None and configured_root.strip():
            candidates.append(Path(configured_root).expanduser())

        environment_root = os.getenv("GPT_SOVITS_ROOT")
        if environment_root:
            candidates.append(Path(environment_root).expanduser())

        candidates.extend(
            [
                GPTSoVITS._managed_source_root(),
                project_root() / "GPT-SoVITS",
                project_root() / "GPT_SoVITS",
                Path.cwd() / "GPT-SoVITS",
                Path.cwd() / "GPT_SoVITS",
            ]
        )

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.name == "GPT_SoVITS":
                resolved = resolved.parent
            if resolved in seen:
                continue
            seen.add(resolved)
            if GPTSoVITS._source_is_available(resolved):
                yield resolved

    @classmethod
    def _resolve_root(cls, configured_root: Optional[str]) -> Path:
        """Resolve or download the GPT-SoVITS source tree used by the backend."""
        existing_root = next(cls._candidate_roots(configured_root), None)
        if existing_root is not None:
            return existing_root
        return cls._download_source_tree(cls._managed_source_root(create=True))

    @classmethod
    def _variant_is_available(
        cls,
        model_root: Path,
        variant: str,
        custom_t2s_weights_path: Optional[Path] = None,
    ) -> bool:
        """Return whether one variant has all required cached assets."""
        variant_files = cls._variant_files[variant]
        if custom_t2s_weights_path is not None:
            variant_files = variant_files[1:]
        required = (*cls._common_files, *variant_files)
        return all((model_root / relative_path).exists() for relative_path in required)

    def _select_variant(
        self, model_root: Path, requested_variant: Optional[str]
    ) -> str:
        """Select an explicit variant or the best available cached variant."""
        if requested_variant is not None:
            if not self._variant_is_available(
                model_root,
                requested_variant,
                self._custom_t2s_weights_path,
            ):
                raise FileNotFoundError(
                    string(
                        "gpt_sovits.variant_not_found",
                        variant=requested_variant,
                        root=model_root,
                    )
                )
            return requested_variant

        for candidate in self._variant_order:
            if self._variant_is_available(
                model_root,
                candidate,
                self._custom_t2s_weights_path,
            ):
                return candidate

        choices = ", ".join(self._variant_order)
        raise FileNotFoundError(
            string(
                "gpt_sovits.no_variant",
                root=model_root,
                available=choices,
            )
        )

    def _validate_refs(self) -> None:
        """Validate GPT-SoVITS reference audio files in the active voice pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def default_model_id(self) -> str:  # noqa
        """Return the selected GPT-SoVITS variant identifier.

        Returns:
            str: The selected GPT-SoVITS variant identifier.
        """
        return self.variant

    @property
    def all_model_ids(self) -> list[str]:  # noqa
        """Return the selected GPT-SoVITS variant identifier.

        Returns:
            list[str]: A one-item list containing the selected variant identifier.
        """
        return [self.variant]

    @property
    def voices(self) -> list[str]:  # noqa
        """Return voice names exposed by the active CEVOICE/CECHAR pack.

        Returns:
            list[str]: Voice names exposed by the active voice pack.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a CEVOICE/CECHAR voice to the selected GPT-SoVITS model.

        Args:
            voice: Voice name to resolve.

        Returns:
            str: The selected GPT-SoVITS variant identifier.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is not None:
            _, voice_names = compatible_bundle
            custom_assert(
                voice in voice_names,
                ValueError(f"{self.name} cannot resolve a model for voice '{voice}'"),
            )
        return self.variant

    def _ensure_model_snapshot(self) -> Path:
        """Download or resolve the GPT-SoVITS pretrained-model cache snapshot."""
        snapshot = self._model_snapshot
        if snapshot is not None:
            return snapshot

        with self._download_lock:
            snapshot = self._model_snapshot
            if snapshot is None:
                try:
                    snapshot = Path(
                        snapshot_download(
                            repo_id=self._model_repo_id,
                            cache_dir=str(huggingface_hub_cache_dir(create=True)),
                            allow_patterns=list(self._model_snapshot_patterns),
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        string("gpt_sovits.model_download_failed", error=str(error))
                    ) from error
                self._model_snapshot = snapshot

        return snapshot

    @staticmethod
    def _nltk_resource_available(
        nltk_module: ModuleType, paths: tuple[str, ...]
    ) -> bool:
        """Return whether any known path for one NLTK resource is available."""
        data = nltk_module.data
        find = data.find
        for path in paths:
            try:
                find(path)
            except LookupError:
                continue
            return True
        return False

    def _ensure_nltk_data(self) -> None:
        """Download NLTK resources required by GPT-SoVITS English frontend."""
        import nltk

        data_dir = runtime_data_dir(create=True) / "nltk_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(data_dir)
        if data_path not in nltk.data.path:
            nltk.data.path.insert(0, data_path)
        os.environ.setdefault("NLTK_DATA", data_path)

        for package, paths in self._nltk_resources.items():
            if self._nltk_resource_available(nltk, paths):
                continue
            if not nltk.download(
                package,
                download_dir=data_path,
                quiet=True,
            ) or not self._nltk_resource_available(nltk, paths):
                raise RuntimeError(
                    string("gpt_sovits.nltk_download_failed", package=package)
                )

    def _ensure_fast_langdetect_data(self) -> None:
        """Download the GPT-SoVITS language-detection model into Celune data."""
        cache_dir = runtime_data_dir(create=True) / "fast_langdetect"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / self._fast_langdetect_model_name

        source_cache = self.root / "GPT_SoVITS/pretrained_models/fast_langdetect"
        if (
            source_cache.exists()
            and source_cache.resolve() != cache_dir.resolve()
            and not source_cache.is_symlink()
        ):
            raise RuntimeError(
                string("gpt_sovits.cache_path_occupied", path=source_cache)
            )
        source_cache.unlink()
        if not source_cache.exists():
            source_cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(cache_dir, source_cache, target_is_directory=True)
            except OSError as error:
                if os.name != "nt":
                    raise RuntimeError(
                        string("gpt_sovits.cache_link_failed", error=str(error))
                    ) from error
                command = f'mklink /J "{source_cache}" "{cache_dir}"'
                try:
                    subprocess.run(
                        ["cmd.exe", "/d", "/c", command],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as junction_error:
                    raise RuntimeError(
                        string(
                            "gpt_sovits.cache_link_failed",
                            error=str(junction_error),
                        )
                    ) from junction_error

        if model_path.is_file():
            return

        try:
            from fast_langdetect.infer import (
                FASTTEXT_LARGE_MODEL_URL,
                ModelDownloader,
            )

            ModelDownloader().download(FASTTEXT_LARGE_MODEL_URL, model_path)
        except Exception as error:
            raise RuntimeError(
                string("gpt_sovits.fast_langdetect_download_failed", error=str(error))
            ) from error

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check whether a GPT-SoVITS variant is complete in the HF cache.

        Args:
            model: GPT-SoVITS variant identifier to check.
            lang: Optional language discriminator; GPT-SoVITS uses one shared cache snapshot.

        Returns:
            tuple[bool, Optional[str]]: Availability flag and cache snapshot path when the variant is complete.
        """
        del lang
        normalized_model = self._normalize_variant(model)
        if normalized_model is None:
            return False, None
        snapshot = self._model_snapshot
        if snapshot is None:
            return False, None
        available = self._variant_is_available(
            snapshot,
            normalized_model,
            self._custom_t2s_weights_path,
        )
        return available, str(snapshot) if available else None

    def preload_models(self) -> None:
        """Download GPT-SoVITS pretrained assets into Celune's HF cache.

        Raises:
            FileNotFoundError: If the selected variant is not in the downloaded model snapshot.
            RuntimeError: If the model snapshot cannot be downloaded.
        """
        self.log(string("gpt_sovits.models_downloading"), "info")
        snapshot = self._ensure_model_snapshot()
        self.log(string("gpt_sovits.nltk_downloading"), "info")
        self._ensure_nltk_data()
        self._ensure_fast_langdetect_data()
        self.variant = self._select_variant(snapshot, self._requested_variant)
        self.model_name = self.variant
        self.log(string("gpt_sovits.variant_available", variant=self.variant), "info")

    @contextlib.contextmanager
    def _source_context(self) -> Generator[None, None, None]:
        """Expose the official source tree while loading or running GPT-SoVITS."""
        source_path = str(self.root)
        package_path = str(self.root / "GPT_SoVITS")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

        with self._source_context_lock:
            previous_cwd = Path.cwd()
            os.chdir(self.root)
            try:
                yield
            finally:
                os.chdir(previous_cwd)

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress GPT-SoVITS diagnostic output while preserving Celune logs."""
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            yield

    suppress_backend_output = _suppress_backend_output

    def _model_config(self, variant: str) -> dict[str, JSONSerializable]:
        """Build an absolute-path TTS_Config payload for one cached variant."""
        model_root = self._ensure_model_snapshot()
        paths = self._variant_files[variant]
        t2s_path = self._custom_t2s_weights_path or model_root / paths[0]
        sovits_path = model_root / paths[1]
        return {
            "custom": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": torch.cuda.is_available(),
                "version": variant,
                "t2s_weights_path": str(t2s_path),
                "vits_weights_path": str(sovits_path),
                "cnhuhbert_base_path": str(model_root / "chinese-hubert-base"),
                "bert_base_path": str(model_root / "chinese-roberta-wwm-ext-large"),
            }
        }

    def _ensure_cached_model_links(self) -> None:
        """Bridge GPT-SoVITS relative asset paths to the HF cache."""
        model_root = self._ensure_model_snapshot()
        link_root = self.root / "GPT_SoVITS/pretrained_models"
        managed_root = self._managed_source_root().resolve()
        for relative_path in self._cached_model_directories:
            target = model_root / relative_path
            link = link_root / relative_path
            if not target.is_dir():
                continue
            if link.exists():
                if link.resolve() == target.resolve():
                    continue
                if not link.is_symlink() and self.root.resolve() != managed_root:
                    continue
                if link.is_symlink():
                    link.unlink()
                else:
                    raise RuntimeError(
                        string("gpt_sovits.cache_path_occupied", path=link)
                    )
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError as error:
                if os.name != "nt":
                    raise RuntimeError(
                        string("gpt_sovits.cache_link_failed", error=str(error))
                    ) from error
                command = f'mklink /J "{link}" "{target}"'
                try:
                    subprocess.run(
                        ["cmd.exe", "/d", "/c", command],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as junction_error:
                    raise RuntimeError(
                        string(
                            "gpt_sovits.cache_link_failed",
                            error=str(junction_error),
                        )
                    ) from junction_error

    def load_model(self, model_id: str, **kwargs) -> _GPTSoVITSRuntime:
        """Load one GPT-SoVITS variant through ``TTS_Config`` and ``TTS``.

        Args:
            model_id: GPT-SoVITS variant identifier to load.
            kwargs: Additional backend-specific load options, currently ignored.

        Returns:
            _GPTSoVITSRuntime: Loaded official GPT-SoVITS runtime wrapper.

        Raises:
            ValueError: If `model_id` is not a known GPT-SoVITS variant.
            FileNotFoundError: If required cached assets or a custom checkpoint are missing.
        """
        del kwargs
        variant = self._normalize_variant(model_id)
        if variant is None:
            raise ValueError(string("gpt_sovits.invalid_model", model=model_id))
        snapshot = self._ensure_model_snapshot()
        self.variant = variant
        self.model_name = variant
        available = self._variant_is_available(
            snapshot,
            variant,
            self._custom_t2s_weights_path,
        )
        if not available:
            raise FileNotFoundError(
                string(
                    "gpt_sovits.variant_not_found",
                    variant=variant,
                    root=snapshot,
                )
            )

        if (
            self._custom_t2s_weights_path is not None
            and not self._custom_t2s_weights_path.is_file()
        ):
            raise FileNotFoundError(
                string(
                    "gpt_sovits.custom_t2s_weights_not_found",
                    path=self._custom_t2s_weights_path,
                )
            )

        self._ensure_cached_model_links()
        self._ensure_nltk_data()
        with self._suppress_backend_output(), self._source_context():
            module = importlib.import_module("GPT_SoVITS.TTS_infer_pack.TTS")
            config_type = cast(
                Callable[[dict[str, JSONSerializable]], _GPTSoVITSConfig],
                module.TTS_Config,
            )
            pipeline_type = cast(
                Callable[[_GPTSoVITSConfig], GPTSoVITSPipeline],
                module.TTS,
            )
            config = config_type(self._model_config(variant))
            setattr(
                config,
                "configs_path",
                str(self.root / "GPT_SoVITS/configs/tts_infer.yaml"),
            )
            pipeline = pipeline_type(config)

        sample_rate = 48000 if variant == "v4" else 32000
        return _GPTSoVITSRuntime(pipeline, variant, sample_rate)

    @staticmethod
    def _normalize_language(language: Optional[str]) -> str:
        """Normalize Celune language identifiers to GPT-SoVITS language tags."""
        if language is None:
            return "auto"
        normalized = language.strip().lower().replace("_", "-")
        aliases: Mapping[str, str] = {
            "auto": "auto",
            "english": "en",
            "chinese": "zh",
            "cantonese": "yue",
            "japanese": "ja",
            "korean": "ko",
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.startswith("zh"):
            return "zh"
        if normalized.startswith("en"):
            return "en"
        if normalized.startswith("ja"):
            return "ja"
        if normalized.startswith("ko"):
            return "ko"
        if normalized.startswith("yue"):
            return "yue"
        if normalized in {"all-zh", "all-ja", "all-yue", "all-ko"}:
            return normalized.replace("-", "_")
        return "auto"

    def resolve_generation_language(self, lang: Optional[str]) -> str:
        """Normalize a requested language for GPT-SoVITS text preprocessing.

        Args:
            lang: Requested Celune language identifier, or None for automatic detection.

        Returns:
            str: GPT-SoVITS language tag used for text preprocessing.
        """
        return self._normalize_language(lang)

    @staticmethod
    def _resolve_text_language(language: str, text: str) -> str:
        """Use explicit English phonemization for unambiguously Latin text."""
        if language != "auto":
            return language
        has_latin = any(char.isascii() and char.isalpha() for char in text)
        has_cjk = any(
            "\u3040" <= char <= "\u30ff"
            or "\u3400" <= char <= "\u4dbf"
            or "\u4e00" <= char <= "\u9fff"
            or "\uac00" <= char <= "\ud7a3"
            for char in text
        )
        return "en" if has_latin and not has_cjk else "auto"

    @staticmethod
    def _prompt_language(reference_text: str, configured: Optional[str]) -> str:
        """Resolve a prompt language from pack metadata or simple script detection."""
        if configured:
            language = GPTSoVITS._normalize_language(configured)
            return GPTSoVITS._resolve_text_language(language, reference_text)
        if any("\uac00" <= char <= "\ud7a3" for char in reference_text):
            return "ko"
        if any("\u3040" <= char <= "\u30ff" for char in reference_text):
            return "ja"
        if any("\u4e00" <= char <= "\u9fff" for char in reference_text):
            return "zh"
        return "en"

    @staticmethod
    def _validate_reference_audio(voice: str, reference_wav: Path) -> None:
        """Reject reference audio that GPT-SoVITS cannot use reliably."""
        duration = float(sf.info(reference_wav).duration)
        if duration < 3.0:
            raise ValueError(string("gpt_sovits.reference_too_short", voice=voice))

    def _trim_reference_silence(self, reference_wav: Path) -> Path:
        """Remove only quiet lead-in and tail from a GPT-SoVITS reference WAV."""
        sample_rate = int(sf.info(reference_wav).samplerate)
        audio, _ = sf.read(reference_wav, dtype="float32", always_2d=True)
        if audio.size == 0:
            return reference_wav

        mono = np.mean(audio, axis=1)
        frame_size = max(1, int(sample_rate * self._reference_frame_ms / 1000))
        frame_energy = np.sqrt(
            np.convolve(
                mono * mono,
                np.ones(frame_size, dtype=np.float32) / frame_size,
                mode="same",
            )
        )
        active_samples = np.flatnonzero(
            frame_energy > self._reference_silence_threshold
        )
        if active_samples.size == 0:
            return reference_wav

        padding = int(sample_rate * self._reference_silence_padding_ms / 1000)
        start = max(0, int(active_samples[0]) - padding)
        end = min(len(audio), int(active_samples[-1]) + padding + 1)
        if start == 0 and end == len(audio):
            return reference_wav

        digest = hashlib.sha1(str(reference_wav.resolve()).encode("utf-8")).hexdigest()[
            :12
        ]
        trimmed_path = reference_wav.with_name(
            f"{reference_wav.stem}-{digest}-trimmed.wav"
        )
        if not trimmed_path.exists():
            sf.write(trimmed_path, audio[start:end], sample_rate)
        self._truncated_reference_paths.add(trimmed_path)
        return trimmed_path

    @staticmethod
    def _to_numpy_audio(
        audio: Union[AudioChunk, AudioChunkNonNormalized, torch.Tensor],  # noqa
    ) -> AudioChunk:
        """Convert one GPT-SoVITS output chunk to mono float32 audio."""
        if isinstance(audio, torch.Tensor):
            values = audio.detach().cpu().numpy()
        else:
            values = np.asarray(audio)
        if np.issubdtype(values.dtype, np.integer):
            return (values.astype(np.float32) / 32768.0).reshape(-1)
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def _run_pipeline(
        self,
        pipeline: GPTSoVITSPipeline,
        request: dict[str, JSONSerializable],
    ) -> Iterator[RuntimeValue]:
        """Run GPT-SoVITS while its relative runtime paths are active."""
        with self._suppress_backend_output(), self._source_context():
            yield from pipeline.run(request)

    @staticmethod
    def _refresh_prompt_cache(
        pipeline: GPTSoVITSPipeline,
        reference_wav: Path,
        prompt_text: str,
        prompt_language: str,
    ) -> None:
        """Invalidate official prompt state when a voice reference changes."""
        prompt_cache = getattr(pipeline, "prompt_cache", None)
        if not isinstance(prompt_cache, dict):
            return
        if (
            prompt_cache.get("ref_audio_path") == str(reference_wav)
            and prompt_cache.get("prompt_text") == prompt_text
            and prompt_cache.get("prompt_lang") == prompt_language
        ):
            return
        prompt_cache.update(
            {
                "ref_audio_path": None,
                "prompt_semantic": None,
                "refer_spec": [],
                "prompt_text": None,
                "prompt_lang": None,
                "phones": None,
                "bert_features": None,
                "norm_text": None,
                "aux_ref_audio_paths": [],
            }
        )

    def generate_stream(
        self, model: _GPTSoVITSRuntime, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune-compatible streaming audio with GPT-SoVITS.

        Args:
            model: Loaded GPT-SoVITS runtime wrapper.
            kwargs: Generation options including text, voice, language, and sampling controls.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: Audio chunks, sample rates, and timing metadata.

        Raises:
            ValueError: If text, voice, or model identifier is invalid.
            RuntimeError: If the runtime has already been unloaded.
        """
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(string("gpt_sovits.text_required"))

        voice = kwargs.pop("voice", self.default_voice)
        language = self.resolve_generation_language(kwargs.pop("language", None))
        language = self._resolve_text_language(language, text)
        kwargs.pop("chunk_size", None)
        kwargs.pop("instruct", None)
        top_k = int(kwargs.pop("top_k", 15))
        top_p = float(kwargs.pop("top_p", 1.0))
        temperature = float(kwargs.pop("temperature", 1.0))
        repetition_penalty = float(kwargs.pop("repetition_penalty", 1.35))

        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(string("celune.unknown_voice", voice=voice))

        reference_wav = self._trim_reference_silence(loader.materialize(voice, "wav"))
        reference_wav = self._truncate_reference(reference_wav)
        self._validate_reference_audio(voice, reference_wav)
        voice_data = loader.bundle.voices[voice]
        reference_text = str(voice_data["reference_text"]).strip()
        configured_prompt_language = voice_data.get("gpt_sovits_prompt_language")
        prompt_language = self._prompt_language(
            reference_text,
            configured_prompt_language
            if isinstance(configured_prompt_language, str)
            else None,
        )

        self._apply_seed()
        if self.current_seed is None:
            self.current_seed = secrets.randbits(32)
            self._apply_seed()
        seed = self.current_seed
        fragment_streaming = model.variant in self._fragment_streaming_variants
        request: dict[str, JSONSerializable] = {
            "text": text,
            "text_lang": language,
            "ref_audio_path": str(reference_wav),
            "prompt_lang": prompt_language,
            "prompt_text": reference_text,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": seed,
            "parallel_infer": True,
            "repetition_penalty": repetition_penalty,
            "sample_steps": 32,
            "return_fragment": fragment_streaming,
            "streaming_mode": not fragment_streaming,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }

        pipeline = model.pipeline
        if pipeline is None:
            raise RuntimeError(string("gpt_sovits.runtime_unloaded"))
        self._refresh_prompt_cache(
            pipeline,
            reference_wav,
            reference_text,
            prompt_language,
        )

        first_chunk_time: Optional[float] = None
        generated = iter(self._run_pipeline(pipeline, request))
        pending: Optional[tuple[int, AudioChunk]] = None
        try:
            for raw_chunk in generated:
                if not isinstance(raw_chunk, tuple) or len(raw_chunk) != 2:
                    continue
                raw_sr, raw_audio = raw_chunk
                if not isinstance(raw_sr, int):
                    continue
                audio = self._to_numpy_audio(
                    cast(Union[AudioChunk, torch.Tensor], raw_audio)
                )
                if pending is not None:
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    yield (
                        pending[1],
                        pending[0],
                        {
                            "backend": self.name,
                            "variant": self.variant,
                            "first_chunk_time": first_chunk_time,
                            "is_final": False,
                        },
                    )
                pending = (raw_sr, audio)

            if pending is not None:
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                yield (
                    pending[1],
                    pending[0],
                    {
                        "backend": self.name,
                        "variant": self.variant,
                        "first_chunk_time": first_chunk_time,
                        "is_final": True,
                    },
                )
        finally:
            close = getattr(generated, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
