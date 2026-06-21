"""Unified Celune dataclass package with lazy re-exports."""

from typing import TYPE_CHECKING
from importlib import import_module

if TYPE_CHECKING:
    from .celune import (
        CELUNE_CONSTANT_PROPERTIES,
        CELUNE_FORWARDED_PROPERTIES,
        CeluneAudioState,
        CeluneBackendState,
        CeluneCallbackState,
        CeluneModelState,
        CelunePipelineState,
        CeluneRuntimeState,
        CeluneVoiceState,
    )
    from .extensions import CeluneContext
    from .persona import (
        ChatMessage,
        GenerateRequest,
        GenerateResponse,
    )
    from .pipeline import (
        AudioOutput,
        AudioInputRequest,
        PlaybackChunk,
        PlaybackSourceDone,
        SpeechDone,
        SpeechRequest,
        SpeechTiming,
        VoiceConversionRequest,
    )
    from .properties import (
        ConstantPropertySpec,
        ForwardedPropertySpec,
        bind_constant_properties,
        bind_forwarded_properties,
        constant_property,
        forward_property,
    )

_MODULE_EXPORTS = {
    "CELUNE_CONSTANT_PROPERTIES": "celune",
    "CELUNE_FORWARDED_PROPERTIES": "celune",
    "CeluneAudioState": "celune",
    "CeluneBackendState": "celune",
    "CeluneCallbackState": "celune",
    "CeluneContext": "extensions",
    "CeluneModelState": "celune",
    "CelunePipelineState": "celune",
    "CeluneRuntimeState": "celune",
    "CeluneVoiceState": "celune",
    "AudioOutput": "pipeline",
    "AudioInputRequest": "pipeline",
    "ChatMessage": "persona",
    "ConstantPropertySpec": "properties",
    "ForwardedPropertySpec": "properties",
    "GenerateRequest": "persona",
    "GenerateResponse": "persona",
    "PlaybackChunk": "pipeline",
    "PlaybackSourceDone": "pipeline",
    "SpeechDone": "pipeline",
    "SpeechRequest": "pipeline",
    "SpeechTiming": "pipeline",
    "VoiceConversionRequest": "pipeline",
    "bind_constant_properties": "properties",
    "bind_forwarded_properties": "properties",
    "constant_property": "properties",
    "forward_property": "properties",
}

__all__ = [
    "CELUNE_CONSTANT_PROPERTIES",
    "CELUNE_FORWARDED_PROPERTIES",
    "CeluneAudioState",
    "AudioOutput",
    "AudioInputRequest",
    "CeluneBackendState",
    "CeluneCallbackState",
    "CeluneContext",
    "CeluneModelState",
    "CelunePipelineState",
    "CeluneRuntimeState",
    "CeluneVoiceState",
    "ChatMessage",
    "ConstantPropertySpec",
    "ForwardedPropertySpec",
    "GenerateRequest",
    "GenerateResponse",
    "PlaybackChunk",
    "PlaybackSourceDone",
    "SpeechDone",
    "SpeechRequest",
    "SpeechTiming",
    "VoiceConversionRequest",
    "bind_constant_properties",
    "bind_forwarded_properties",
    "constant_property",
    "forward_property",
]


def __getattr__(name: str):
    """Resolve dataclass exports lazily to avoid package import cycles."""
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return the lazily exported package surface."""
    return __all__
