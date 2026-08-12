# SPDX-License-Identifier: MIT
"""Tests for Celune's package-level public surface."""

import celune

from .support import CeluneTestCase


class TestPackageApi(CeluneTestCase):
    """Tests for package exports and interactive inspection."""

    def test_dir_only_lists_curated_public_exports(self) -> None:
        """Verify package inspection hides helper implementation names."""
        assert dir(celune) == [
            "Celune",
            "CeluneContext",
            "CeluneExtension",
            "REVISION",
            "__codename__",
            "__comment__",
            "__tagline__",
            "__version__",
            "subscribe",
        ]

    def test_helper_names_are_not_public(self) -> None:
        """Verify package internals no longer appear as plain public names."""
        assert not hasattr(celune, "caller_is_repl")
        assert not hasattr(celune, "dirty")
        assert not hasattr(celune, "get_revision")
        assert not hasattr(celune, "local")
        assert not hasattr(celune, "sys")
