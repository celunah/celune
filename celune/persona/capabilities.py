# SPDX-License-Identifier: Apache-2.0
"""Capability declarations for loaded Persona architectures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaCapabilities:
    """Features that the active Persona architecture can safely perform."""

    text: bool = True
    vision: bool = False
    image_uploads: bool = False
    emotion_probes: bool = False
