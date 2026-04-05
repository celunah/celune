# pylint: disable=C0114, C0413
from .utils import get_revision

REVISION = get_revision()
if REVISION:
    __version__ = f"3.0.0 ({REVISION})"  # circular import moment
else:
    __version__ = "3.0.0"

# due to how Celune imports __version__ we cannot put these imports according to PEP8
from .celune import Celune
from .extensions.base import CeluneContext, CeluneExtension

__all__ = [
    "Celune",
    "CeluneContext",
    "CeluneExtension",
    "__version__",
]
