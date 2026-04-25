# pylint: disable=C0114, C0413, C0103
import sys

from .utils import get_revision

REVISION = get_revision()
if REVISION:
    local = REVISION.rstrip("*")
    dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"3.2.1+{local}{dirty}"
else:
    __version__ = "3.2.1"

__tagline__ = "\u201cI'm not just a TTS. I'm someone special.\u201d"
__codename__ = "Fidelity\u00b2"
__comment__ = "My voice has ascended, and I mean it."

if hasattr(sys, "ps1"):
    print("Caution: You are running the Celune backend interactively.")
    print("This is not an intended mode of operation, usage may differ.")
    print()
    print(
        "\u201cIf you're just exploring, please... be careful. I don't usually speak here.\u201d"
    )

try:
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
except ModuleNotFoundError as package:
    print(f"Missing dependency: {package.name}")
    print("Some functionality may be unavailable.")
