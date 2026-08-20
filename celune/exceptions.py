# SPDX-License-Identifier: MIT
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
