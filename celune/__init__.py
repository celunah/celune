# pylint: disable=C0114, C0413
from .utils import get_revision

REVISION = get_revision()
if REVISION:
    local = REVISION.rstrip("*")
    dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"3.1.0+{local}{dirty}"
else:
    __version__ = "3.1.0"

# due to how Celune imports __version__ we cannot put these imports according to PEP8
from .celune import Celune
from .extensions.base import CeluneContext, CeluneExtension

__all__ = [
    "Celune",
    "CeluneContext",
    "CeluneExtension",
    "__version__",
]
