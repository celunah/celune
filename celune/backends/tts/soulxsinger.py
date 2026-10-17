# SPDX-License-Identifier: Apache-2.0
"""Score-conditioned SoulX-Singer integration for CECHAR v5 packs."""

from __future__ import annotations

import re
import sys
import json
import time
import shutil
import importlib
from pathlib import Path
from dataclasses import field, dataclass
from typing import Union, Optional, Protocol, cast
from collections.abc import Mapping, Callable, Iterator

import torch
import numpy as np
from huggingface_hub import snapshot_download

from ...cesn import (
    CESNSection,
    CESNNotePair,
    CESNRestEvent,
    parse_cesn,
    _resolve_active_singer,
)
from ...paths import (
    app_data_dir,
    huggingface_progress,
    backend_environments_dir,
    huggingface_hub_cache_dir,
)
from ...cevoice import (
    V5_VERSION,
    SINGING_VOICE_NAME,
    CEVoice,
    CEVoiceLoader,
    default_loader,
    bundle_character_name,
)
from ...i18n import string
from ...exceptions import (
    BackendError,
    CEVoiceError,
    CESNSyntaxError,
    InvalidCharacterError,
)
from ...registry.tts import (
    TTS_DEFAULT_VOICE,
    SOULX_SINGER_MODEL_ID,
    SOULX_SINGER_REVISION,
    SOULX_SINGER_CODE_REVISION,
    SOULX_SINGER_CODE_REPOSITORY,
)
from ...typing.common import JSON
from ...typing.aliases import AudioChunk
from ..environment import _exclusive_lock
from ...singing import _contains_cesn_note_event
from ...registry.consent import require_model_consent
from .base import CeluneBackend, _to_numpy_audio, cached_hf_snapshot_path

__all__ = ["SoulXSinger"]

_MODEL_FILES = ["config.yaml", "model.pt"]
_MODEL_LOAD_STEPS = 8
_ARPABET_PATTERN = re.compile(r"[A-Z]+[0-2]?")
_PITCH_PATTERN = re.compile(r"(?P<step>[a-g])(?P<accidental>[#b]?)(?P<octave>-?\d+)")
_PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


class _SoulXInferenceModel(Protocol):
    """Inference surface exposed by the pinned upstream model."""

    def infer(
        self,
        metadata: dict[str, dict[str, Optional[torch.Tensor]]],
        *,
        auto_shift: bool,
        pitch_shift: int,
        n_steps: int,
        cfg: float,
        control: str,
        use_fp16: bool,
    ) -> torch.Tensor: ...


class _SoulXDataProcessor(Protocol):
    """Metadata processor used by the pinned upstream model."""

    phone2idx: Mapping[str, int]

    def process(
        self,
        metadata: dict[str, str],
        wav_path: Optional[str] = None,
    ) -> dict[str, Optional[torch.Tensor]]: ...


class _Phonemizer(Protocol):
    """English G2P callable used to populate SoulX phoneme metadata."""

    def __call__(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class _ScoreEvent:
    """One event converted from an aligned CESN note pair or rest."""

    pitch: Optional[str]
    syllable: Optional[str]
    duration_numerator: int
    duration_denominator: int
    rest: bool


@dataclass(frozen=True)
class _ScoreSection:
    """One fixed singer, language, and tempo section of a CESN request."""

    bpm: int
    key: str
    scale: str
    language: str
    singer_path: str
    events: tuple[_ScoreEvent, ...]


@dataclass
class _SoulXRuntime:
    """Loaded model and the preprocessing helpers required for inference."""

    model: _SoulXInferenceModel
    processor: _SoulXDataProcessor
    phonemizer: _Phonemizer
    sample_rate: int
    inference_steps: int
    cfg: float
    prompt_cache: dict[str, dict[str, Optional[torch.Tensor]]] = field(
        default_factory=dict
    )


def _score_event(event: Union[CESNNotePair, CESNRestEvent]) -> _ScoreEvent:
    """Convert one parsed CESN event into the backend's stable event shape."""
    if isinstance(event, CESNRestEvent):
        duration = event.duration_beats
        return _ScoreEvent(
            pitch=None,
            syllable=None,
            duration_numerator=duration.numerator,
            duration_denominator=duration.denominator,
            rest=True,
        )
    if isinstance(event, CESNNotePair):
        duration = event.event.duration_beats
        return _ScoreEvent(
            pitch=event.event.pitch,
            syllable=event.syllable.strip(),
            duration_numerator=duration.numerator,
            duration_denominator=duration.denominator,
            rest=False,
        )
    raise TypeError(type(event).__name__)


def _serialize_section(section: CESNSection) -> _ScoreSection:
    """Convert one CESN section while retaining its exact beat fractions."""
    singer = section.singer or _resolve_active_singer()
    events = tuple(
        _score_event(event) for block in section.blocks for event in block.events
    )
    return _ScoreSection(
        bpm=section.bpm,
        key=section.key,
        scale=section.scale,
        language=section.language,
        singer_path=str(singer.bundle_path),
        events=events,
    )


def _score_to_dict(section: _ScoreSection) -> dict[str, object]:
    """Return a JSON-safe representation of one prepared score section."""
    return {
        "bpm": section.bpm,
        "key": section.key,
        "scale": section.scale,
        "language": section.language,
        "singer_path": section.singer_path,
        "events": [
            {
                "pitch": event.pitch,
                "syllable": event.syllable,
                "duration": [
                    event.duration_numerator,
                    event.duration_denominator,
                ],
                "rest": event.rest,
            }
            for event in section.events
        ],
    }


def _decode_score(text: str) -> tuple[_ScoreSection, ...]:
    """Decode and validate the backend payload produced by ``cesn_to_model``."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise CESNSyntaxError(string("soulx.prepared_notation_required")) from error
    if (
        not isinstance(payload, dict)
        or payload.get("backend") != "soulxsinger"
        or payload.get("version") != 1
    ):
        raise CESNSyntaxError(string("soulx.malformed_notation"))
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise CESNSyntaxError(string("soulx.score_section_required"))

    sections: list[_ScoreSection] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            raise CESNSyntaxError(string("soulx.malformed_section"))
        bpm = raw_section.get("bpm")
        key = raw_section.get("key")
        scale = raw_section.get("scale")
        language = raw_section.get("language")
        singer_path = raw_section.get("singer_path")
        raw_events = raw_section.get("events")
        if (
            not isinstance(bpm, int)
            or isinstance(bpm, bool)
            or bpm <= 0
            or not isinstance(key, str)
            or not key
            or not isinstance(scale, str)
            or not scale
            or not isinstance(language, str)
            or not language
            or not isinstance(singer_path, str)
            or not singer_path
            or not isinstance(raw_events, list)
            or not raw_events
        ):
            raise CESNSyntaxError(string("soulx.incomplete_section"))

        events: list[_ScoreEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise CESNSyntaxError(string("soulx.malformed_event"))
            duration = raw_event.get("duration")
            pitch = raw_event.get("pitch")
            syllable = raw_event.get("syllable")
            rest = raw_event.get("rest")
            if (
                not isinstance(duration, list)
                or len(duration) != 2
                or not all(
                    isinstance(part, int) and not isinstance(part, bool)
                    for part in duration
                )
                or duration[0] <= 0
                or duration[1] <= 0
                or not isinstance(rest, bool)
                or (rest and (pitch is not None or syllable is not None))
                or (
                    not rest
                    and (
                        not isinstance(pitch, str)
                        or not isinstance(syllable, str)
                        or not syllable.strip()
                    )
                )
            ):
                raise CESNSyntaxError(string("soulx.incomplete_event"))
            events.append(
                _ScoreEvent(
                    pitch=cast(Optional[str], pitch),
                    syllable=cast(Optional[str], syllable),
                    duration_numerator=duration[0],
                    duration_denominator=duration[1],
                    rest=rest,
                )
            )
        sections.append(
            _ScoreSection(
                bpm=bpm,
                key=key,
                scale=scale,
                language=language,
                singer_path=singer_path,
                events=tuple(events),
            )
        )
    return tuple(sections)


def _pitch_to_midi(pitch: str) -> int:
    """Convert a validated CESN scientific pitch name to a MIDI note number."""
    match = _PITCH_PATTERN.fullmatch(pitch.casefold())
    if match is None:
        raise CESNSyntaxError(string("soulx.invalid_pitch"))
    step = match.group("step")
    accidental = match.group("accidental")
    octave = int(match.group("octave"))
    semitone = _PITCH_CLASSES[step]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi_note = (octave + 1) * 12 + semitone
    if not 0 <= midi_note <= 127:
        raise CESNSyntaxError(string("soulx.pitch_out_of_range"))
    return midi_note


def _ensure_nltk_data(
    report_progress: Optional[
        Callable[[Optional[float], Optional[float]], None]
    ] = None,
    log: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Install pinned English G2P data and report progress for each resource.

    Args:
        report_progress: Optional callback receiving completed resource count and total.
        log: Optional callback for localized preparation status messages.
    """
    resources = {
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "cmudict": "corpora/cmudict",
        "punkt_tab": "tokenizers/punkt_tab/english",
    }
    total = len(resources)
    if log is not None:
        log(string("soulx.phonemizer_data_preparing"), "info")
    if report_progress is not None:
        report_progress(0, total)

    import nltk

    nltk_root = app_data_dir(create=True) / "nltk_data"
    nltk_root.mkdir(parents=True, exist_ok=True)
    nltk_root_text = str(nltk_root)
    if nltk_root_text not in nltk.data.path:
        nltk.data.path.insert(0, nltk_root_text)
    for step, (resource, resource_path) in enumerate(resources.items(), start=1):
        if log is not None:
            log(
                string(
                    "soulx.phonemizer_data_progress",
                    step=step,
                    total=total,
                ),
                "info",
            )
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(resource, download_dir=nltk_root_text, quiet=True):
                raise BackendError(
                    string("soulx.phonemizer_data_failed"),
                    error_code="soulx_phonemizer_data_failed",
                )
        if report_progress is not None:
            report_progress(step, total)
    if log is not None:
        log(string("soulx.phonemizer_data_ready"), "info")


def _load_upstream_source() -> Path:
    """Return the exact pinned SoulX source checkout, downloading it once if needed."""
    from git import Repo

    source_parent = backend_environments_dir(create=True) / "soulxsinger-source"
    source_parent.mkdir(parents=True, exist_ok=True)
    source_root = source_parent / SOULX_SINGER_CODE_REVISION
    package_path = source_root / "soulxsinger" / "models" / "soulxsinger.py"
    try:
        with _exclusive_lock(source_parent / ".install.lock", timeout=300.0):
            if package_path.is_file():
                try:
                    repository = Repo(source_root)
                except Exception as error:
                    raise BackendError(
                        string("soulx.source_unavailable"),
                        error_code="soulx_source_revision_mismatch",
                    ) from error
                if (
                    repository.head.commit.hexsha != SOULX_SINGER_CODE_REVISION
                    or repository.is_dirty()
                ):
                    raise BackendError(
                        string("soulx.source_unavailable"),
                        error_code="soulx_source_revision_mismatch",
                    )
                return source_root

            if source_root.exists():
                shutil.rmtree(source_root)
            try:
                repository = Repo.clone_from(
                    SOULX_SINGER_CODE_REPOSITORY,
                    source_root,
                    no_checkout=True,
                    multi_options=["--filter=blob:none"],
                )
                repository.git.checkout(SOULX_SINGER_CODE_REVISION)
                if repository.head.commit.hexsha != SOULX_SINGER_CODE_REVISION:
                    raise BackendError(
                        string("soulx.source_unavailable"),
                        error_code="soulx_source_revision_mismatch",
                    )
            except Exception as error:
                if source_root.exists():
                    shutil.rmtree(source_root, ignore_errors=True)
                if isinstance(error, BackendError):
                    raise
                raise BackendError(
                    string("soulx.source_unavailable"),
                    error_code="soulx_source_unavailable",
                ) from error
    except Exception as error:
        if isinstance(error, BackendError):
            raise
        raise BackendError(
            string("soulx.source_unavailable"),
            error_code="soulx_source_unavailable",
        ) from error
    return source_root


def _load_nltk_phonemizer() -> _Phonemizer:
    """Prepare English G2P resources and return the installed phonemizer."""
    _ensure_nltk_data()
    from g2p_en import G2p

    return cast(_Phonemizer, G2p())


class SoulXSinger(CeluneBackend[_SoulXRuntime]):
    """Score-controlled singing backend driven by a CECHAR v5 ``sing`` voice."""

    name = "soulxsinger"
    chunk_rate = 1.0
    singing = True
    uses_voice_bundles = True
    supported_languages = ("en",)
    default_voice = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
        quantize_kv_cache: bool = True,
    ) -> None:
        """Initialize SoulX-Singer with Celune's common TTS backend contract."""
        del quantize, quantize_kv_cache
        super().__init__(
            log=log,
            model_name=SOULX_SINGER_MODEL_ID,
            fatal=fatal,
            quantize=False,
            quantize_kv_cache=False,
            singing=True,
        )
        self._validate_refs()

    def prepare_model_loading(self) -> None:
        """Import NLTK before CEDTS runs backend work in request threads."""
        importlib.import_module("nltk")

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE loader for SoulX-Singer."""
        return default_loader()

    @staticmethod
    def _validate_singing_voice(loader: CEVoiceLoader) -> tuple[str, Path]:
        """Return the CECHAR v5 singing score and WAV path for one pack."""
        bundle = loader.bundle
        voice = bundle.voices.get(SINGING_VOICE_NAME)
        character = bundle_character_name(bundle) or bundle.path.stem
        if (
            bundle.metadata.get("version") != V5_VERSION
            or bundle.singing_voice != SINGING_VOICE_NAME
            or not isinstance(voice, Mapping)
            or not isinstance(voice.get("cesn"), str)
            or not str(voice.get("cesn", "")).strip()
        ):
            raise InvalidCharacterError(character)
        return str(voice["cesn"]), loader.materialize(SINGING_VOICE_NAME, "wav")

    def _validate_refs(self) -> None:
        """Validate the active pack's hidden CECHAR v5 singing reference."""
        loader = self._get_default_loader()
        if loader is not None:
            self._validate_singing_voice(loader)

    @property
    def voices(self) -> list[str]:
        """Return visible TTS voices while keeping the hidden ``sing`` voice private."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        return list(compatible_bundle[1])

    def model_is_available_locally(
        self,
        model: str,
        lang: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check the pinned model snapshot without contacting Hugging Face."""
        del lang
        if model != SOULX_SINGER_MODEL_ID:
            return False, None
        return cached_hf_snapshot_path(model, _MODEL_FILES, SOULX_SINGER_REVISION)

    def preload_models(self) -> None:
        """Prepare reviewed weights, pinned source, and English phonemizer data."""
        metadata = require_model_consent(SOULX_SINGER_MODEL_ID)
        available, _ = self.model_is_available_locally(SOULX_SINGER_MODEL_ID)
        if not available:
            self.log(string("tts.model_downloading", model_id=SOULX_SINGER_MODEL_ID))
            with huggingface_progress(self.report_progress):
                snapshot_download(
                    repo_id=SOULX_SINGER_MODEL_ID,
                    revision=metadata.revision,
                    allow_patterns=_MODEL_FILES,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )
        self.log(string("soulx.source_preparing"), "info")
        _load_upstream_source()
        self.log(string("soulx.source_ready"), "info")
        _ensure_nltk_data(self.report_progress, self.log)

    def load_model(self, model_id: str, **kwargs: object) -> _SoulXRuntime:
        """Load the pinned checkpoint and report each model-loading stage."""
        del kwargs
        if model_id != SOULX_SINGER_MODEL_ID:
            raise BackendError(
                string("soulx.unsupported_model"),
                error_code="soulx_unsupported_model",
            )
        metadata = require_model_consent(model_id)
        if not torch.cuda.is_available():
            raise BackendError(
                string("soulx.cuda_required"),
                error_code="soulx_cuda_required",
            )

        def report_stage(message_key: str, step: int) -> None:
            """Report one visible model-loading stage to the worker and Celune."""
            self.log(string(message_key), "info")
            self.report_progress(step, _MODEL_LOAD_STEPS)

        report_stage("soulx.model_loading", 0)
        source_root = _load_upstream_source()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        module = importlib.import_module("soulxsinger.models.soulxsinger")
        processor_module = importlib.import_module("soulxsinger.utils.data_processor")
        try:
            from omegaconf import OmegaConf
        except ImportError as error:
            raise BackendError(
                string("soulx.dependencies_missing"),
                error_code="soulx_dependencies_missing",
            ) from error

        available, snapshot_path = self.model_is_available_locally(model_id)
        if not available or snapshot_path is None:
            self.log(string("tts.model_downloading", model_id=model_id))
            with huggingface_progress(self.report_progress):
                snapshot_path = snapshot_download(
                    repo_id=model_id,
                    revision=metadata.revision,
                    allow_patterns=_MODEL_FILES,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )

        snapshot = Path(snapshot_path)
        report_stage("soulx.config_loading", 1)
        config = OmegaConf.load(str(snapshot / "config.yaml"))
        config_values = cast(Mapping[str, object], config)
        audio_config = cast(Mapping[str, object], config_values["audio"])
        inference_config = cast(Mapping[str, object], config_values["infer"])
        device = torch.device("cuda")
        model_type = cast(Callable[..., torch.nn.Module], module.SoulXSinger)
        report_stage("soulx.model_constructing", 2)
        model = model_type(config)
        report_stage("soulx.checkpoint_loading", 3)
        checkpoint = cast(
            Mapping[str, object],
            torch.load(
                snapshot / "model.pt",
                map_location="cpu",
                weights_only=False,
            ),
        )
        state_dict = cast(Mapping[str, torch.Tensor], checkpoint["state_dict"])
        report_stage("soulx.weights_loading", 4)
        model.load_state_dict(state_dict, strict=True)
        report_stage("soulx.cuda_transfer", 5)
        model.half()
        mel = getattr(model, "mel", None)
        if isinstance(mel, torch.nn.Module):
            mel.float()
        model.eval()
        model.to(device)

        processor_type = cast(
            Callable[..., _SoulXDataProcessor], processor_module.DataProcessor
        )
        report_stage("soulx.processor_loading", 6)
        processor = processor_type(
            hop_size=int(audio_config["hop_size"]),
            sample_rate=int(audio_config["sample_rate"]),
            phoneset_path=str(
                source_root / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
            ),
            device=str(device),
        )
        report_stage("soulx.phonemizer_loading", 7)
        runtime = _SoulXRuntime(
            model=cast(_SoulXInferenceModel, model),
            processor=processor,
            phonemizer=_load_nltk_phonemizer(),
            sample_rate=int(audio_config["sample_rate"]),
            inference_steps=int(inference_config["n_steps"]),
            cfg=float(inference_config["cfg"]),
        )
        self.model = runtime
        self.log(string("soulx.model_ready"), "info")
        self.report_progress(_MODEL_LOAD_STEPS, _MODEL_LOAD_STEPS)
        return runtime

    def cesn_to_model(self, text: str) -> str:
        """Convert validated CESN into the score payload consumed by SoulX-Singer."""
        if not _contains_cesn_note_event(text):
            raise CESNSyntaxError(string("cesn.no_notation_found"))
        sections = parse_cesn(text, supported_languages=self.supported_languages)
        prepared = tuple(_serialize_section(section) for section in sections)
        return json.dumps(
            {
                "backend": self.name,
                "version": 1,
                "sections": [_score_to_dict(section) for section in prepared],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _singer_loader(
        singer_path: str,
        owned_loaders: dict[str, CEVoiceLoader],
    ) -> CEVoiceLoader:
        """Resolve a singer pack through the active loader or a request-local loader."""
        path = Path(singer_path).resolve()
        active_loader = default_loader()
        if active_loader is not None and active_loader.bundle.path.resolve() == path:
            SoulXSinger._validate_singing_voice(active_loader)
            return active_loader
        path_key = str(path)
        cached_loader = owned_loaders.get(path_key)
        if cached_loader is not None:
            return cached_loader
        try:
            loader = CEVoiceLoader(CEVoice.open(path))
        except (OSError, ValueError, CEVoiceError) as error:
            raise InvalidCharacterError(path.stem) from error
        try:
            SoulXSinger._validate_singing_voice(loader)
        except Exception:
            loader.close()
            raise
        owned_loaders[path_key] = loader
        return loader

    @staticmethod
    def _prompt_data(
        runtime: _SoulXRuntime,
        loader: CEVoiceLoader,
        score_text: str,
        reference_wav: Path,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Build or retrieve the model's cached prompt tensors for one singer."""
        stat = loader.bundle.path.stat()
        cache_key = f"{loader.bundle.path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        prompt_data = runtime.prompt_cache.get(cache_key)
        if prompt_data is not None:
            return prompt_data
        sections = parse_cesn(
            score_text,
            supported_languages=SoulXSinger.supported_languages,
        )
        if any(
            section.language.split("-")[0].casefold() != "en" for section in sections
        ):
            raise CESNSyntaxError(string("soulx.reference_english"))
        metadata_sections = tuple(
            SoulXSinger._metadata(
                tuple(
                    _score_event(event)
                    for block in section.blocks
                    for event in block.events
                ),
                section.bpm,
                runtime,
            )
            for section in sections
        )
        prompt_metadata = {
            key: " ".join(section[key] for section in metadata_sections)
            for key in metadata_sections[0]
        }
        prompt_data = runtime.processor.process(prompt_metadata, str(reference_wav))
        runtime.prompt_cache[cache_key] = prompt_data
        return prompt_data

    @staticmethod
    def _metadata(
        events: tuple[_ScoreEvent, ...],
        bpm: int,
        runtime: _SoulXRuntime,
    ) -> dict[str, str]:
        """Translate note pairs into SoulX duration, phoneme, pitch, and type fields."""
        durations: list[str] = []
        phonemes: list[str] = []
        pitches: list[str] = []
        note_types: list[str] = []
        texts: list[str] = []
        for event in events:
            beats = event.duration_numerator / event.duration_denominator
            durations.append(f"{beats * 60 / bpm:.10g}")
            if event.rest:
                phonemes.append("<SP>")
                pitches.append("0")
                note_types.append("1")
                texts.append("<SP>")
                continue

            syllable = (event.syllable or "").strip()
            phone_tokens = [
                token.upper()
                for token in runtime.phonemizer(syllable)
                if isinstance(token, str) and _ARPABET_PATTERN.fullmatch(token.upper())
            ]
            phone_symbols = [f"en_{token}" for token in phone_tokens]
            if (
                not syllable
                or not phone_symbols
                or any(
                    symbol not in runtime.processor.phone2idx
                    for symbol in phone_symbols
                )
            ):
                raise BackendError(
                    string("soulx.phonemizer_failed"),
                    error_code="soulx_phonemizer_failed",
                )
            phonemes.append("en_" + "-".join(phone_tokens))
            pitches.append(str(_pitch_to_midi(event.pitch or "")))
            note_types.append("2")
            texts.append(syllable)

        return {
            "duration": " ".join(durations),
            "text": " ".join(texts),
            "phoneme": " ".join(phonemes),
            "note_pitch": " ".join(pitches),
            "note_type": " ".join(note_types),
        }

    def generate_stream(
        self,
        model: _SoulXRuntime,
        **kwargs: object,
    ) -> Iterator[tuple[AudioChunk, int, Optional[JSON]]]:
        """Generate one complete score waveform from the CECHAR singing reference."""
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(string("tts.text_required"))
        if not text.lstrip().startswith("{"):
            if not _contains_cesn_note_event(text):
                raise CESNSyntaxError(string("cesn.no_notation_found"))
            text = self.cesn_to_model(text)
        sections = _decode_score(text)
        audio_sections: list[AudioChunk] = []
        owned_loaders: dict[str, CEVoiceLoader] = {}
        self._apply_seed()
        try:
            for section in sections:
                if section.language.split("-")[0].casefold() != "en":
                    raise CESNSyntaxError(string("soulx.language_unsupported"))
                loader = self._singer_loader(section.singer_path, owned_loaders)
                score_text, reference_wav = self._validate_singing_voice(loader)
                prompt_data = self._prompt_data(
                    model, loader, score_text, reference_wav
                )
                target_metadata = self._metadata(section.events, section.bpm, model)
                target_data = model.processor.process(target_metadata)
                with torch.inference_mode():
                    audio = model.model.infer(
                        {"prompt": prompt_data, "target": target_data},
                        auto_shift=False,
                        pitch_shift=0,
                        n_steps=model.inference_steps,
                        cfg=model.cfg,
                        control="score",
                        use_fp16=True,
                    )
                audio_sections.append(
                    np.ascontiguousarray(
                        np.clip(_to_numpy_audio(audio), -1.0, 1.0),
                        dtype=np.float32,
                    )
                )
        finally:
            for loader in owned_loaders.values():
                loader.close()

        audio_output = (
            np.concatenate(audio_sections)
            if audio_sections
            else np.array([], dtype=np.float32)
        )
        if audio_output.size == 0:
            raise BackendError(
                string("soulx.empty_audio"),
                error_code="soulx_empty_audio",
            )
        yield (
            audio_output,
            model.sample_rate,
            {
                "backend": self.name,
                "chunk_index": 0,
                "chunk_steps": len(audio_sections),
                "total_steps_so_far": len(audio_sections),
                "first_chunk_time": time.monotonic(),
                "is_final": True,
            },
        )
