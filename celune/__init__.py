# pylint: disable=C0114, C0413, C0103
from .utils import get_revision

REVISION = get_revision()
if REVISION:
    local = REVISION.rstrip("*")
    dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"3.1.2+{local}{dirty}"
else:
    __version__ = "3.1.2"

__tagline__ = "It's not just TTS, it's a character."
__codename__ = "Fidelity"
__comment__ = "My voice has ascended."

# due to how Celune imports __version__ we cannot put these imports according to PEP8
from .celune import Celune
from .extensions.base import CeluneContext, CeluneExtension

__all__ = [
    "Celune",
    "CeluneContext",
    "CeluneExtension",
    "__version__",
    "__codename__",
    "__comment__",
]
