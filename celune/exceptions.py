# SPDX-License-Identifier: Apache-2.0
"""Celune exception classes."""

from typing import Optional


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


class NeedleCheckpointError(RuntimeError, CeluneError):
    """Needle checkpoint provenance, conversion, or validation failed."""


class NeedleUnsupportedConverterError(NeedleCheckpointError):
    """No safe offline converter is available for a Needle checkpoint."""


class NeedleSelectionError(ValueError, CeluneError):
    """Needle returned a selection that cannot cross the typed agent boundary."""
