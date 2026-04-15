"""Celune exception classes."""


class CeluneError(Exception):
    """General Celune exception."""


class ExtensionError(Exception):
    """General extension exception."""


class NotAvailableError(RuntimeError, CeluneError):
    """Celune is currently unavailable."""


class WarmupError(RuntimeError, CeluneError):
    """Celune cannot warm up at this time."""


class AudioMismatchError(RuntimeError, CeluneError):
    """Audio pipeline received data that does not match Celune's current state."""


class BadAudioError(ValueError, CeluneError):
    """Celune cannot process audio in this format."""


class IncompleteExtensionError(NotImplementedError, ExtensionError):
    """User did not define a required extension method."""


class InvalidExtensionError(TypeError, ExtensionError):
    """Extension is not properly formed."""


class ExtensionAlreadyRegisteredError(ExtensionError):
    """Extension is already registered."""


class No(Exception):
    """Celune does not want to start today."""
