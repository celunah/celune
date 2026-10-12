# SPDX-License-Identifier: Apache-2.0
"""Typed shared metadata for Celune's model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ModelCategory = Literal["tts", "llm", "asr", "other"]


@dataclass(frozen=True)
class ModelMetadata:
    """Describe a Hugging Face model and the terms Celune has reviewed.

    ``None`` for a license or restriction flag means Celune does not know that
    value. Unknown terms require an explicit acknowledgement before use.
    """

    model_id: str
    display_name: str
    category: ModelCategory
    revision: str
    license_id: Optional[str]
    license_url: Optional[str]
    non_commercial: Optional[bool]
    research_only: Optional[bool]
    repository_url: str
    roles: tuple[str, ...] = ()
    tier: Optional[Literal["standard", "smart"]] = None
    terms_review_required: bool = False

    @property
    def requires_consent(self) -> bool:
        """Return whether terms are restricted, incomplete, or unregistered."""
        return (
            self.license_id is None
            or self.license_url is None
            or self.non_commercial is None
            or self.research_only is None
            or self.non_commercial is True
            or self.research_only is True
            or self.terms_review_required
        )

    def to_dict(self, *, consented: Optional[bool] = None) -> dict[str, object]:
        """Return the public API representation of this model."""
        result: dict[str, object] = {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "category": self.category,
            "revision": self.revision,
            "license": self.license_id,
            "license_url": self.license_url,
            "non_commercial": self.non_commercial,
            "research_only": self.research_only,
            "repository_url": self.repository_url,
            "roles": list(self.roles),
            "tier": self.tier,
            "terms_review_required": self.terms_review_required,
            "requires_consent": self.requires_consent,
        }
        if consented is not None:
            result["consented"] = consented
        return result
