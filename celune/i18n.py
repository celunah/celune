# SPDX-License-Identifier: MIT
"""Celune internationalization helpers backed by JSON language files."""

import os
import json
import ctypes
import contextlib
import locale as _locale  # else it gets shadowed
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

DEFAULT_LOCALE = "en"
_LANG_DIR = Path(__file__).resolve().parent / "lang"

STRINGS: dict[str, dict[str, str]] = {}


def _normalize_locale_name(locale_name: Optional[str]) -> str:
    """Normalize a locale label into a lookup-friendly language code."""
    if not locale_name:
        return DEFAULT_LOCALE

    return locale_name.replace("_", "-").split(".", maxsplit=1)[0].lower()


def _locale_candidates(locale_name: Optional[str]) -> list[str]:
    """Return locale lookup candidates from most to least specific."""
    normalized = _normalize_locale_name(locale_name)
    candidates = [normalized]

    if "-" in normalized:
        candidates.append(normalized.split("-", maxsplit=1)[0])

    if DEFAULT_LOCALE not in candidates:
        candidates.append(DEFAULT_LOCALE)

    return candidates


def _lang_file(locale_name: str) -> Path:
    """Return the JSON file path for one normalized language code."""
    return _LANG_DIR / f"{locale_name}.json"


def _load_locale_strings(locale_name: str) -> dict[str, str]:
    """Load one locale file from disk, returning an empty table when absent."""
    normalized = _normalize_locale_name(locale_name)
    path = _lang_file(normalized)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    resolved: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            resolved[key] = value
    return resolved


def _ensure_locale_loaded(locale_name: Optional[str]) -> None:
    """Load one locale table into the in-memory cache when needed."""
    normalized = _normalize_locale_name(locale_name)
    if normalized in STRINGS:
        return

    STRINGS[normalized] = _load_locale_strings(normalized)


def get_system_locale() -> str:
    """Get the current system locale, falling back to English if unavailable.

    Returns:
        str: The detected locale code, or ``"en"`` when no locale can be found.
    """
    lang, _ = _locale.getlocale()
    if lang:
        return lang

    if os.name == "nt":
        with contextlib.suppress(Exception):
            windll = getattr(ctypes, "windll", SimpleNamespace()).kernel32
            lang_code = windll.GetUserDefaultUILanguage()
            return _locale.windows_locale.get(lang_code, DEFAULT_LOCALE)

    lang = os.environ.get("LANG")
    if lang:
        return lang.split(".")[0]

    return DEFAULT_LOCALE


_ensure_locale_loaded(DEFAULT_LOCALE)
_current_locale = get_system_locale()


def set_locale(locale: str) -> None:
    """Set Celune's active locale.

    Args:
        locale: The locale code to store as the current language selection.
    """
    global _current_locale
    _current_locale = locale


def get_locale() -> str:
    """Get Celune's current locale setting.

    Returns:
        str: The currently configured locale code.
    """
    return _current_locale


def string(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Get an internationalized string for the selected language.

    Args:
        key: The translation key to look up.
        locale: An optional locale override. When omitted, the current locale is used.
        kwargs: Optional format values interpolated into the resolved string.

    Returns:
        str: The translated string, or the key itself when no translation exists.
    """
    text = None
    for lang in _locale_candidates(locale or _current_locale):
        _ensure_locale_loaded(lang)
        text = STRINGS.get(lang, {}).get(key)
        if text is not None:
            break

    if text is None:
        text = key

    if kwargs:
        return text.format(**kwargs)

    return text
