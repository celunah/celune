# SPDX-License-Identifier: MIT
"""Tests for Celune's package-level public surface."""

from unittest import TestCase

import celune


class PackageApiTests(TestCase):
    """Tests for package exports and interactive inspection."""

    def test_dir_only_lists_curated_public_exports(self) -> None:
        """Verify package inspection hides helper implementation names.

        Returns:
            None: This function tests whether Celune returned the intended package structure after calling dir().
        """
        self.assertEqual(
            dir(celune),
            [
                "Celune",
                "CeluneContext",
                "CeluneExtension",
                "REVISION",
                "__codename__",
                "__comment__",
                "__tagline__",
                "__version__",
            ],
        )

    def test_helper_names_are_not_public(self) -> None:
        """Verify package internals no longer appear as plain public names.

        Returns:
            None: This function tests whether Celune did not leak an unexpected attribute into the public interface.
        """
        self.assertFalse(hasattr(celune, "caller_is_repl"))
        self.assertFalse(hasattr(celune, "dirty"))
        self.assertFalse(hasattr(celune, "get_revision"))
        self.assertFalse(hasattr(celune, "local"))
        self.assertFalse(hasattr(celune, "sys"))
