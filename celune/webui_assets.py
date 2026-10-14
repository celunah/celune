# SPDX-License-Identifier: Apache-2.0
"""Load static files used by the mounted WebUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from importlib.resources import files

from .paths import project_root

_ASSET_MEDIA_TYPES = {
    "api_key.html": "text/html; charset=utf-8",
    "api_key.css": "text/css; charset=utf-8",
    "api_key.js": "text/javascript; charset=utf-8",
    "head.html": "text/html; charset=utf-8",
    "theme.css": "text/css; charset=utf-8",
    "webui.css": "text/css; charset=utf-8",
    "webui.js": "text/javascript; charset=utf-8",
}
_FONT_ASSET_NAMES = {
    "fonts/michroma-400.ttf": "michroma-400.ttf",
    "fonts/outfit-400.ttf": "outfit-400.ttf",
    "fonts/outfit-700.ttf": "outfit-700.ttf",
}


def read_webui_asset(name: str) -> str:
    """Read a packaged WebUI text asset by its fixed name."""
    if name not in _ASSET_MEDIA_TYPES:
        raise FileNotFoundError(name)
    return files("celune").joinpath("webui_assets", name).read_text(encoding="utf-8")


def webui_asset_response_data(name: str) -> Optional[tuple[bytes, str]]:
    """Return safe packaged WebUI or bundled font data for an asset route."""
    media_type = _ASSET_MEDIA_TYPES.get(name)
    if media_type is not None:
        data = files("celune").joinpath("webui_assets", name).read_bytes()
        return data, media_type

    font_name = _FONT_ASSET_NAMES.get(name)
    if font_name is None:
        return None
    font_path: Path = project_root() / "resources" / "fonts" / font_name
    try:
        return font_path.read_bytes(), "font/ttf"
    except OSError:
        return None
