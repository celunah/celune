# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-exports for UI callback protocols."""

from ..typing.ui import CeluneBaseUI, CeluneTextualUI, CeluneHeadlessBaseUI

# is this it?
__all__ = ["CeluneBaseUI", "CeluneHeadlessBaseUI", "CeluneTextualUI"]
