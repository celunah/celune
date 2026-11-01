"""Celune's host-audio, resampling, and DSP implementations."""


def __getattr__(name: str) -> object:
    """Load selected audio helpers on demand from their owning modules."""
    if name == "pad_note":
        from .dsp import pad_note

        return pad_note
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
