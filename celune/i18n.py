"""Celune internationalization stubs."""

from __future__ import annotations

import os
import locale as _locale  # else it gets shadowed
import ctypes
import contextlib
from types import SimpleNamespace
from typing import Optional

DEFAULT_LOCALE = "en"

STRINGS: dict[str, dict[str, str]] = {
    "en": {},
}


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
            return _locale.windows_locale.get(lang_code, "en")

    lang = os.environ.get("LANG")
    if lang:
        return lang.split(".")[0]

    return "en"


_current_locale = get_system_locale()


def set_locale(locale: str) -> None:
    """Set Celune's active locale.

    Args:
        locale: The locale code to store as the current language selection.

    Returns:
        None: This method updates global locale state in place.
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
        locale: An optional locale override. When omitted, the current locale is
            used.
        **kwargs: Optional format values interpolated into the resolved string.

    Returns:
        str: The translated string, or the key itself when no translation exists.
    """
    lang = locale or _current_locale
    text = STRINGS.get(lang, {}).get(key)

    if text is None:
        text = STRINGS.get(DEFAULT_LOCALE, {}).get(key, key)

    if kwargs:
        return text.format(**kwargs)

    return text
