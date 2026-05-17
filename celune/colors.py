# SPDX-License-Identifier: MIT
"""Color utility helpers."""

from __future__ import annotations

import colorsys
import random
from typing import Final

from textual.theme import Theme

DEFAULT_BACKGROUND: Final[str] = "#1d1826"
DEFAULT_ACCENT: Final[str] = "#cebaff"
RGB = tuple[int, int, int]


def random_hex() -> str:
    """Return a random six-digit hex color.

    Returns:
        str: The random hex color.
    """
    hex_numbers = []
    for _ in range(6):
        hex_numbers.append(random.choice("123456789abcdef"))

    return f"#{''.join(hex_numbers)}"


def _rgb(color: str) -> RGB:
    return (
        int(color[1:3], 16),
        int(color[3:5], 16),
        int(color[5:7], 16),
    )


def _hex(rgb: RGB) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _blend(color: str, destination: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    source_rgb = _rgb(color)
    destination_rgb = _rgb(destination)
    return _hex(
        (
            round(source_rgb[0] + ((destination_rgb[0] - source_rgb[0]) * amount)),
            round(source_rgb[1] + ((destination_rgb[1] - source_rgb[1]) * amount)),
            round(source_rgb[2] + ((destination_rgb[2] - source_rgb[2]) * amount)),
        )
    )


def _hls_color(hue: float, lightness: float, saturation: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return _hex(
        (
            round(red * 255),
            round(green * 255),
            round(blue * 255),
        )
    )


def _relative_luminance(color: str) -> float:
    def channel(value: int) -> float:
        normalized = value / 255
        if normalized <= 0.04045:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(value) for value in _rgb(color))
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _contrast_ratio(first: str, second: str) -> float:
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _ensure_contrast(color: str, background: str, minimum: float) -> str:
    """Keep blending towards white or black until the specified accent and background color is readable against the UI.

    Args:
        color: The accent color to blend.
        background: The background color to blend.
        minimum: The minimum contrast threshold that has to be met.

    Returns:
        str: The color meeting the contrast threshold requirements.
    """
    if _contrast_ratio(color, background) >= minimum:
        return color

    white_candidate = color
    black_candidate = color
    for step in range(1, 101):
        amount = step / 100
        white_candidate = _blend(color, "#ffffff", amount)
        black_candidate = _blend(color, "#000000", amount)
        white_ok = _contrast_ratio(white_candidate, background) >= minimum
        black_ok = _contrast_ratio(black_candidate, background) >= minimum
        if white_ok and black_ok:
            white_change = _contrast_ratio(color, white_candidate)
            black_change = _contrast_ratio(color, black_candidate)
            return white_candidate if white_change <= black_change else black_candidate
        if white_ok:
            return white_candidate
        if black_ok:
            return black_candidate

    return (
        white_candidate
        if _contrast_ratio(white_candidate, background)
        >= _contrast_ratio(black_candidate, background)
        else black_candidate
    )


def _default_dark_palette() -> dict[str, str]:
    return {
        "primary": "#cebaff",
        "secondary": "#a595cc",
        "accent": "#7c7099",
        "foreground": "#e2ceff",
        "background": "#1d1826",
        "surface": "#1d1826",
        "warning": "#f0e68c",
        "error": "#f07178",
    }


def _default_light_palette() -> dict[str, str]:
    return {
        "primary": "#33293f",
        "secondary": "#281732",
        "accent": "#1e1126",
        "foreground": "#473d53",
        "background": "#ece8ff",
        "surface": "#ece8ff",
        "warning": "#6b5e00",
        "error": "#7a1f24",
    }


def _derive_dark_palette(background: str, accent: str) -> dict[str, str]:
    if background == DEFAULT_BACKGROUND and accent == DEFAULT_ACCENT:
        return _default_dark_palette()

    primary = _ensure_contrast(accent, background, 4.5)
    foreground = _ensure_contrast(_blend(primary, "#ffffff", 0.18), background, 7.0)
    return {
        "primary": primary,
        "secondary": _blend(primary, background, 0.22),
        "accent": _blend(primary, background, 0.45),
        "foreground": foreground,
        "background": background,
        "surface": background,
        "warning": _ensure_contrast(_hls_color(0.15, 0.7, 0.8), background, 4.5),
        "error": _ensure_contrast(_hls_color(0.99, 0.66, 0.82), background, 4.5),
    }


def _derive_light_palette(background: str, accent: str) -> dict[str, str]:
    if background == DEFAULT_BACKGROUND and accent == DEFAULT_ACCENT:
        return _default_light_palette()

    light_background = (
        background
        if _relative_luminance(background) >= 0.8
        else _blend(background, "#ffffff", 0.9)
    )
    primary = _ensure_contrast(accent, light_background, 4.5)
    foreground = _ensure_contrast(_blend(primary, "#000000", 0.18), light_background, 7)
    return {
        "primary": primary,
        "secondary": _blend(primary, light_background, 0.18),
        "accent": _blend(primary, light_background, 0.35),
        "foreground": foreground,
        "background": light_background,
        "surface": light_background,
        "warning": _ensure_contrast(
            _hls_color(0.15, 0.28, 0.85), light_background, 4.5
        ),
        "error": _ensure_contrast(_hls_color(0.99, 0.32, 0.72), light_background, 4.5),
    }


def _theme(name: str, palette: dict[str, str], *, dark: bool) -> Theme:
    return Theme(
        name=name,
        primary=palette["primary"],
        secondary=palette["secondary"],
        accent=palette["accent"],
        foreground=palette["foreground"],
        background=palette["background"],
        surface=palette["surface"],
        warning=palette["warning"],
        error=palette["error"],
        dark=dark,
    )


def configure_theme(
    background: str = DEFAULT_BACKGROUND,
    accent: str = DEFAULT_ACCENT,
) -> None:
    """Rebuild Celune's theme family from two bundle-provided seed colors.

    Args:
        background: The background color provided by a CEVOICE pack.
        accent: The accent color provided by a CEVOICE pack.

    Returns:
        None: This function builds Celune's theme colors automatically.
    """
    global THEME, THEME_LIGHT, SEVERITY_COLORS

    dark_palette = _derive_dark_palette(background, accent)
    light_palette = _derive_light_palette(background, accent)
    THEME = _theme("celune", dark_palette, dark=True)
    THEME_LIGHT = _theme("celune_light", light_palette, dark=False)
    SEVERITY_COLORS = {
        "celune": {
            "info": dark_palette["primary"],
            "warning": dark_palette["warning"],
            "error": dark_palette["error"],
        },
        "celune_light": {
            "info": light_palette["primary"],
            "warning": light_palette["warning"],
            "error": light_palette["error"],
        },
        "celune_april_fools": {
            "info": random_hex(),
            "warning": random_hex(),
            "error": random_hex(),
        },
    }


THEME_APRIL_FOOLS = Theme(
    name="celune_april_fools",
    primary=random_hex(),
    secondary=random_hex(),
    accent=random_hex(),
    foreground=random_hex(),
    background=random_hex(),
    surface=random_hex(),
    warning=random_hex(),
    error=random_hex(),
    dark=False,
)

THEME: Theme
THEME_LIGHT: Theme
SEVERITY_COLORS: dict[str, dict[str, str]]
configure_theme()
