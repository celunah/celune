# SPDX-License-Identifier: MIT
"""Tests for the docstring update script."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "update_docstrings.py"
SPEC = importlib.util.spec_from_file_location("update_docstrings_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
UPDATE_DOCSTRINGS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UPDATE_DOCSTRINGS
SPEC.loader.exec_module(UPDATE_DOCSTRINGS)


class UpdateDocstringsTests(TestCase):
    """Verify nested docstring rewriting behavior."""

    def test_local_class_methods_keep_their_docstrings(self) -> None:
        """Verify methods on classes defined inside functions are preserved."""
        source = '''def test_think_builds_persona_payload_and_queues_response(self) -> None:
    """Verify Persona request formatting without loading a Persona model.

    Raises:
        AssertionError: Persona request behavior changes unexpectedly.
    """

    class FakeResponse:
        """Fake API response class."""

        @staticmethod
        def raise_for_status() -> None:
            """Fake return of raise_for_status()."""

        @staticmethod
        def json() -> JSONSerializable:
            """Return a fake response."""
            return {"response": "I can help with that."}

    class FakeVision:
        """Fake vision API class object."""

        def __init__(self) -> None:
            self.payload: Optional[JSON] = None

        def post(self, json: JSON) -> FakeResponse:
            """Post a fake request."""
            self.payload = json
            return FakeResponse()
'''

        replacements = UPDATE_DOCSTRINGS.collect_replacements(source, ast.parse(source))
        updated = source
        for replacement in replacements:
            updated = (
                updated[: replacement.start]
                + replacement.text
                + updated[replacement.end :]
            )

        self.assertIn(
            'def raise_for_status() -> None:\n            """',
            updated,
        )
        self.assertIn(
            'def json() -> JSONSerializable:\n            """',
            updated,
        )
        self.assertIn(
            'def post(self, json: JSON) -> FakeResponse:\n            """',
            updated,
        )
        self.assertNotIn("def raise_for_status() -> None:\n            pass", updated)
        self.assertNotIn("def json() -> JSONSerializable:\n            pass", updated)
        self.assertNotIn(
            "def post(self, json: JSON) -> FakeResponse:\n            pass", updated
        )

    def test_public_docstring_wraps_generated_lines_to_max_width(self) -> None:
        """Verify generated docstrings wrap at the configured line length."""
        source = """def example(alpha, beta):\n    pass\n"""
        function = ast.parse(source).body[0]
        parsed = SimpleNamespace(
            description=(
                "This is a deliberately long summary sentence that should wrap cleanly without producing "
                "any docstring line that exceeds the configured maximum width."
            ),
            args={
                "alpha": (
                    "A deliberately verbose argument description that should wrap with a hanging indent so "
                    "that pylint line-length checks remain satisfied."
                ),
                "beta": (
                    "Another very long argument description that exists only to verify wrapping behavior "
                    "for generated docstring fields."
                ),
            },
            returns=(
                "A deliberately long return description that should also wrap cleanly when the docstring "
                "generator formats the Returns section."
            ),
            raises={
                "RuntimeError": (
                    "A deliberately long raise description that should wrap with a hanging indent rather "
                    "than overflowing the configured line length limit."
                )
            },
        )

        docstring = UPDATE_DOCSTRINGS.public_docstring(function, "    ", parsed)

        self.assertTrue(all(len(line) <= 120 for line in docstring.splitlines()))
