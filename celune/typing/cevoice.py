# SPDX-License-Identifier: MIT
"""CEVOICE manifest type aliases."""

from typing import Union

from .common import JSONSerializable

type ManifestValue = Union[JSONSerializable, "Manifest"]
type Manifest = dict[str, ManifestValue]
type VoiceManifest = dict[str, Manifest]
