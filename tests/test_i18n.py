# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's localization database contract."""

import ast
import json
import re
from pathlib import Path


_LANGUAGE_FILE = Path("celune/lang/en.json")
_EXCEPTION_PLACEHOLDER = re.compile(r"\{(?:error|detail|reason)\}")


def _translations() -> dict[str, str]:
    """Load the English localization table used by the test suite."""
    return json.loads(_LANGUAGE_FILE.read_text(encoding="utf-8"))


def test_localization_entries_follow_size_and_content_rules() -> None:
    """Keep keys short and out of technical or exception-detail territory."""
    translations = _translations()

    assert all(len(key) <= 50 for key in translations)
    assert all(len(value) <= 100 for value in translations.values())
    assert "SPDX-License-Identifier" not in translations
    assert not any(
        key.startswith(("backends.cedts", "backends.worker")) for key in translations
    )
    assert not any(
        _EXCEPTION_PLACEHOLDER.search(value) for value in translations.values()
    )
    assert not any(re.match(r"^\[[^\]]+\]", value) for value in translations.values())


def test_literal_localization_calls_have_english_entries() -> None:
    """Ensure every static source localization reference resolves in English."""
    references: set[str] = set()
    for source_path in Path("celune").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"string", "tagged_string"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                references.add(node.args[0].value)

    assert references <= _translations().keys()
