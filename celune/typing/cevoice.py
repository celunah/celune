"""CEVOICE manifest type aliases."""

from typing import Union

from .common import JSONSerializable

ManifestValue = Union[JSONSerializable, "Manifest"]
Manifest = dict[str, ManifestValue]
VoiceManifest = dict[str, Manifest]
