# SPDX-License-Identifier: Apache-2.0
"""Registry of repository-bundled default voice packs."""

from types import MappingProxyType
from typing import Final
from collections.abc import Mapping

DEFAULT_CEVOICE_PACK_SHA256: Final[str] = (
    "6aa4f7928e9333c89f9836635beae159b53a861bcb207f9e5a34b81ed717426a"
)
PREVIOUS_DEFAULT_CEVOICE_PACK_SHA256: Final[str] = (
    "8422d3f0a455fd5fd89bc4a687a8fb9e7a56a1746d89e4fe3404e29a7004bc5e"
)
LEGACY_DEFAULT_CEVOICE_PACK_SHA256: Final[str] = (
    "35db55adca450062806e20d6c8ae1fc0c0ad5de569cb422321981de80e567851"
)
CLASSIC_CEVOICE_PACK_SHA256: Final[str] = (
    "646d68a70d4402cc5a9f8fdc96c171607d896dfe5424b27748193e3504530023"
)
NOL_CEVOICE_PACK_SHA256: Final[str] = (
    "c52612b94ddae2d14569e0497d6e518a9dcde19f4f6d1076cd54574f0a87844b"
)

KNOWN_DEFAULT_CEVOICE_PACKS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "default.cevoice": frozenset(
            {
                DEFAULT_CEVOICE_PACK_SHA256,
                PREVIOUS_DEFAULT_CEVOICE_PACK_SHA256,
                LEGACY_DEFAULT_CEVOICE_PACK_SHA256,
            }
        ),
        "classic.cevoice": frozenset({CLASSIC_CEVOICE_PACK_SHA256}),
        "nol.cevoice": frozenset({NOL_CEVOICE_PACK_SHA256}),
    }
)
OFFICIAL_CEVOICE_PACK_SHA256S: Final[frozenset[str]] = frozenset(
    checksum
    for checksums in KNOWN_DEFAULT_CEVOICE_PACKS.values()
    for checksum in checksums
)
