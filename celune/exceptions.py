# SPDX-License-Identifier: Apache-2.0
"""Celune exception classes."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union

from .typing.common import JSON

if TYPE_CHECKING:
    from .registry.types import ModelMetadata


class CeluneError(Exception):
    """General Celune exception."""


class ExtensionError(Exception):
    """General extension exception."""


class TaskSubscriptionClosed(RuntimeError, CeluneError):
    """A task subscription was closed."""


class RuntimeCheckError(RuntimeError, CeluneError):
    """Celune's runtime checks have failed."""


class NotAvailableError(RuntimeError, CeluneError):
    """Celune is currently unavailable."""


class BackendError(RuntimeError, CeluneError):
    """Celune backend has failed."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "backend_error",
        error_type: Optional[str] = None,
    ) -> None:
        """Initialize a backend error with inert diagnostic metadata."""
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type


class BackendEnvironmentError(RuntimeError, CeluneError):
    """A backend's isolated dependency environment cannot be used."""


class LocalManagementError(RuntimeError, CeluneError):
    """Describe a typed local-management failure and its exact target."""

    def __init__(
        self, status: str, message: str, target: Optional[Path] = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.target = target

    def to_json(self) -> JSON:
        """Serialize the failure without exposing an ambiguous target."""
        return {
            "result": self.status,
            "message": str(self),
            "target": str(self.target) if self.target is not None else None,
        }


class CEDTSError(RuntimeError, CeluneError):
    """Base class for all CEDTS transport, protocol, and payload failures."""


class CEDTSStreamError(OSError, CEDTSError):
    """CEDTS could not read from or write to its transport stream."""


class CEDTSEOFError(CEDTSStreamError):
    """CEDTS reached the end of a stream before a packet was complete."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        packet_name: Optional[str] = None,
    ) -> None:
        """Initialize an unexpected-end-of-stream error."""
        super().__init__(message or "unexpected EOF while reading stream")
        self.packet_name = packet_name


class CEDTSTimeoutError(TimeoutError, CEDTSError):
    """CEDTS did not receive a packet before its deadline."""

    def __init__(
        self,
        packet_name: str,
        timeout_seconds: float,
        *,
        message: Optional[str] = None,
    ) -> None:
        """Initialize a timeout with the packet and deadline that expired."""
        super().__init__(
            message or f"{packet_name} timed out after {timeout_seconds:g} seconds"
        )
        self.packet_name = packet_name
        self.timeout_seconds = timeout_seconds


class CEDTSProtocolError(CEDTSError):
    """CEDTS received or produced an invalid control packet."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        packet_name: Optional[str] = None,
    ) -> None:
        """Initialize a protocol error with optional packet context."""
        super().__init__(message or f"invalid packet {packet_name or 'packet'}")
        self.packet_name = packet_name


class CEDTSPayloadError(CEDTSError):
    """CEDTS received an invalid binary or typed payload descriptor."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        packet_name: Optional[str] = None,
    ) -> None:
        """Initialize a payload error with optional packet context."""
        super().__init__(
            message
            or "invalid binary payload received while processing "
            f"{packet_name or 'packet'}"
        )
        self.packet_name = packet_name


class WarmupError(RuntimeError, CeluneError):
    """Celune cannot warm up at this time."""


class AudioMismatchError(RuntimeError, CeluneError):
    """Audio pipeline received data that does not match Celune's current state."""


class BadAudioError(ValueError, CeluneError):
    """Celune cannot process audio in this format."""


class UpdateError(RuntimeError, CeluneError):
    """Celune cannot update at this time."""


class IncompleteExtensionError(NotImplementedError, ExtensionError):
    """User did not define a required extension method."""


class InvalidExtensionError(TypeError, ExtensionError):
    """Extension is not properly formed."""


class ExtensionAlreadyRegisteredError(RuntimeError, ExtensionError):
    """Extension is already registered."""


class No(Exception):
    """Celune does not want to start today."""


class CEVoiceError(RuntimeError, CeluneError):
    """CEVOICE data is malformed or unsupported."""


class InvalidCharacterError(CEVoiceError):
    """A requested character pack is missing or cannot provide a singing voice."""

    def __init__(self, character_name: str) -> None:
        """Initialize a failed character selection with its requested name."""
        self.character_name = character_name
        super().__init__(
            f"Character {character_name!r} is missing or has no usable CECHAR v5 singing voice"
        )


class CESNError(ValueError, CeluneError):
    """Base class for Celune Singing Notation parsing failures."""

    def __init__(self, message: str) -> None:
        """Initialize a notation error without an attached source location."""
        self.source_start: Optional[int] = None
        self.source_end: Optional[int] = None
        super().__init__(message)


class CESNSyntaxError(CESNError):
    """Compatibility base for typed CESN notation failures."""


class CESNValidationError(CESNSyntaxError):
    """CESN notation is malformed or violates a notation invariant."""


class CESNUnsupportedFeatureError(CESNSyntaxError):
    """CESN requests a feature unsupported by the parser or selected backend."""


class CESNBackendFormatError(CESNSyntaxError):
    """A serialized CESN score violates a backend-specific format contract."""


class CESNRangeError(CESNValidationError):
    """A CESN note is outside the selected singing voice's pitch range."""

    def __init__(
        self,
        note: str,
        midi_note: int,
        character_name: str,
        minimum_midi: int,
        maximum_midi: int,
    ) -> None:
        """Initialize a pitch-range failure with the note and voice limits."""
        self.note = note[0].upper() + note[1:]
        self.midi_note = midi_note
        self.character_name = character_name
        self.minimum_midi = minimum_midi
        self.maximum_midi = maximum_midi
        pitch_names = (
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        )
        minimum_name = f"{pitch_names[minimum_midi % 12]}{minimum_midi // 12 - 1}"
        maximum_name = f"{pitch_names[maximum_midi % 12]}{maximum_midi // 12 - 1}"
        super().__init__(
            f"note {self.note} (MIDI {midi_note}) is outside {character_name}'s "
            f"acceptable range: {minimum_name}–{maximum_name} "
            f"(MIDI {minimum_midi}–{maximum_midi})"
        )


class CESNAlignmentError(CESNValidationError):
    """The note-event count does not match the lyric syllable count."""

    def __init__(self, note_count: int, syllable_count: int) -> None:
        """Initialize an alignment error with both sequence lengths."""
        self.note_count = note_count
        self.syllable_count = syllable_count
        super().__init__(
            "note and syllable counts do not align: "
            f"{note_count} notes for {syllable_count} syllables"
        )


class CESNMatchError(CESNValidationError):
    """A note event has no corresponding lyric syllable."""

    def __init__(self, note_index: int, note_event: str, syllable_count: int) -> None:
        """Initialize an error for the first unmatched note event."""
        self.note_index = note_index
        self.note_event = note_event
        self.syllable_count = syllable_count
        super().__init__(
            f"note event {note_index} ({note_event!r}) did not match any "
            f"of {syllable_count} syllables"
        )


class CESNNotationWarning(UserWarning):
    """A CESN pitch falls outside its section's selected scale."""

    def __init__(self, note_event: str, key: str, scale: str) -> None:
        """Initialize an out-of-scale warning with its notation context."""
        from .i18n import string

        self.note_event = note_event
        self.key = key
        self.scale = scale
        super().__init__(
            string(
                "cesn.out_of_scale_note",
                note=note_event,
                tonic=key,
                scale=scale,
            )
        )


class NeedleCheckpointError(RuntimeError, CeluneError):
    """Needle checkpoint provenance, conversion, or validation failed."""


class NeedleUnsupportedConverterError(NeedleCheckpointError):
    """No safe offline converter is available for a Needle checkpoint."""


class NeedleSelectionError(ValueError, CeluneError):
    """Needle returned a selection that cannot cross the typed agent boundary."""


class ModelContractError(RuntimeError, CeluneError):
    """A TTS model artifact or loaded state violates its backend contract."""


class ModelConsentRequired(RuntimeError, CeluneError):
    """A model cannot be loaded until the user acknowledges its terms."""

    def __init__(self, metadata: "ModelMetadata") -> None:
        self.metadata = metadata
        super().__init__("model license consent is required")


class ModelConsentDeclined(ModelConsentRequired):
    """A model load was rejected because the user declined its terms."""


class ModelCommercialUseRestricted(PermissionError, CeluneError):
    """A model's reviewed terms do not permit or confirm commercial use."""

    def __init__(self, metadata: "ModelMetadata") -> None:
        from .i18n import string

        self.metadata = metadata
        super().__init__(
            string(
                "registry.commercial_use_blocked",
                model_name=metadata.display_name,
            )
        )


class _CheckpointDType(Protocol):
    """A dtype-like value that can be rendered in checkpoint diagnostics."""

    def __str__(self) -> str:
        """Return the display name of the dtype."""


class InvalidCheckpoint(ModelContractError):
    """A checkpoint failed a backend model contract validation."""

    def __init__(
        self,
        *,
        backend: str,
        filename: str,
        path: str,
        tensor_name: Optional[str] = None,
        layer_name: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[Union[str, _CheckpointDType]] = None,
        expected: Optional[Union[str, _CheckpointDType]] = None,
        actual: Optional[Union[str, _CheckpointDType]] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.filename = filename
        self.path = path
        self.tensor_name = tensor_name
        self.layer_name = layer_name
        self.shape = shape
        self.dtype = dtype
        self.expected = expected
        self.actual = actual
        self.reason = reason

        details = [f"{backend} checkpoint contract failed: {filename} ({path})"]
        if tensor_name is not None:
            details.append(f"tensor={tensor_name}")
        if layer_name is not None:
            details.append(f"layer={layer_name}")
        if shape is not None:
            details.append(f"shape={shape}")
        if dtype is not None:
            details.append(f"dtype={dtype}")
        if expected is not None or actual is not None:
            details.append(f"expected={expected}, actual={actual}")
        if reason is not None:
            details.append(reason)
        super().__init__(": ".join((details[0], "; ".join(details[1:]))))


class EmptyClassifierOutput(ValueError):
    """Identify a response that contains an empty structured-output field."""


class EspeakExecutableNotFound(FileNotFoundError):
    """Identify eSpeak-NG launch failures for a concise user warning."""


class OnnxRuntimeNotFound(ImportError):
    """Identify a missing ONNX Runtime installation for a concise user warning."""
