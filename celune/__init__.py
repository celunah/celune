# pylint: disable=C0114, C0413, C0103
from .utils import get_revision

REVISION = get_revision()
if REVISION:
    local = REVISION.rstrip("*")
    dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"3.2.1+{local}{dirty}"
else:
    __version__ = "3.2.1"

__tagline__ = "\"I'm not just a TTS. I'm someone special.\""
__codename__ = "Fidelity\u00b2"
__comment__ = "My voice has ascended, and I mean it."

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
