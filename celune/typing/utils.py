# SPDX-License-Identifier: Apache-2.0
"""Utility-layer typed dictionaries."""

from typing import TypedDict


class CallerInfo(TypedDict):
    """Caller information type annotation."""

    function: str
    filename: str
    line: int


class LanguageResult(TypedDict):
    """Language detection metadata type annotation."""

    language: str
    languages: list[str]
    probabilities: dict[str, float]
    supported: bool
