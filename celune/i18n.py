# pylint: disable=W0603
"""Celune internationalization stubs."""

from __future__ import annotations

import os
import locale as _locale  # else it gets shadowed
import ctypes
import contextlib

DEFAULT_LOCALE = "en"

STRINGS: dict[str, dict[str, str]] = {
    "en": {},
}


def get_system_locale() -> str:
    """Get the current system locale, fall back to English if not found."""
    lang, _ = _locale.getlocale()
    if lang:
        return lang

    if os.name == "nt":
        with contextlib.suppress(Exception):
            windll = ctypes.windll.kernel32
            lang_code = windll.GetUserDefaultUILanguage()
            return _locale.windows_locale.get(lang_code, "en")

    lang = os.environ.get("LANG")
    if lang:
        return lang.split(".")[0]

    return "en"


_current_locale = get_system_locale()


def set_locale(locale: str) -> None:
    """Set Celune's locale settings."""
    global _current_locale
    _current_locale = locale


def get_locale() -> str:
    """Get Celune's current locale settings."""
    return _current_locale


def string(key: str, locale: str | None = None, **kwargs) -> str:
    """Get internationalization string in the selected language by key."""
    lang = locale or _current_locale
    text = STRINGS.get(lang, {}).get(key)

    if text is None:
        text = STRINGS.get(DEFAULT_LOCALE, {}).get(key, key)

    if kwargs:
        return text.format(**kwargs)

    return text
