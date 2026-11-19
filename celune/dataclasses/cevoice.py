# SPDX-License-Identifier: Apache-2.0
"""Persona metadata records read from CEVOICE and CECHAR packs."""

from dataclasses import field, dataclass


@dataclass(frozen=True, slots=True)
class PersonaIdentity:
    """Identity details supplied by a CEVOICE/CECHAR pack."""

    name: str = ""
    age: str = ""
    gender: str = ""
    profile: str = ""


@dataclass(frozen=True, slots=True)
class PersonaStyleValues:
    """Baseline speaking-style values supplied by a CEVOICE/CECHAR pack."""

    warmth: str = ""
    directness: str = ""
    humor: str = ""
    detail: str = ""
    formality: str = ""
    enthusiasm: str = ""


@dataclass(frozen=True, slots=True)
class CEVoicePersona:
    """Persona metadata supplied by a CEVOICE/CECHAR pack."""

    identity: PersonaIdentity = field(default_factory=PersonaIdentity)
    speaking_style: str = ""
    boundaries: tuple[str, ...] = ()
    prompt_rules: tuple[str, ...] = ()
    example_dialogue: tuple[str, ...] = ()
    style: PersonaStyleValues = field(default_factory=PersonaStyleValues)
