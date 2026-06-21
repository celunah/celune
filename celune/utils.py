# SPDX-License-Identifier: MIT
"""Celune common utility functions."""

import os
import re
import math
import time
import random
import inspect
import datetime
import textwrap
import traceback
import subprocess
import multiprocessing
from pathlib import Path
from collections.abc import Iterator
from typing import Union, Callable, Optional, Literal, Any, overload

import psutil
import langdetect

from .paths import traceback_path
from .constants import REFERENCE_NEW_MOON
from .typing.utils import CallerInfo, LanguageResult
from .terminal import supports_ansi as terminal_supports_ansi


def get_revision() -> str:
    """Get the current Git repository revision.

    Returns:
        str: The short commit hash, suffixed with ``*`` when the worktree is dirty, or an empty string when Git metadata
        is unavailable.
    """
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = "*" if status else ""
        return f"{rev}{dirty}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def format_number(num: float, precision: int = 0, fallback: str = "N/A") -> str:
    """Format a number without trailing zeroes.

    Args:
        num: The numeric value to format.
        precision: The number of decimal places to preserve before trimming.
        fallback: The fallback value to return if the number is not representable.

    Returns:
        str: The formatted numeric string.

    Raises:
        ValueError: ``precision`` is negative.
    """
    if precision < 0:
        raise ValueError("precision must be >= 0")

    if not math.isfinite(num):
        return fallback

    digits = precision if precision > 0 else 12
    text = f"{num:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a hexadecimal color code to an RGB tuple.

    Args:
        color: A 3-digit or 6-digit hexadecimal color string, optionally prefixed with ``#`` or ``0x``.

    Returns:
        tuple[int, ...]: The parsed ``(red, green, blue)`` color components.

    Raises:
        ValueError: ``color`` is not a valid 3- or 6-character hex code.
    """
    color = color.strip()

    if color.startswith("#"):
        color = color[1:]
    elif color.lower().startswith("0x"):
        color = color[2:]

    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6 or any(c.lower() not in "0123456789abcdef" for c in color):
        raise ValueError(f"expected a 3 or 6-character hex code, found {color}")

    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
    )


def lunar_info(dt: datetime.datetime) -> tuple[float, float, float]:
    """Get lunar state from the given date and time.

    Args:
        dt: The date and time to use.

    Returns:
        tuple[float, float, float]: The lunar phase, illumination level and days until a full moon.
    """
    frac_dt = dt.astimezone(datetime.timezone.utc)
    since_ref = (frac_dt - REFERENCE_NEW_MOON).total_seconds() / 86400
    cycle_days = 29.530588
    phase = (since_ref / cycle_days) % 1.0
    illumination = 0.5 * (1 - math.cos(2 * math.pi * phase))
    days_until_full = ((0.5 - phase) % 1.0) * cycle_days

    return phase, illumination, days_until_full


def lunar_phase(phase: float) -> str:
    """Convert a phase float to a phase name.

    Args:
        phase: The floating point phase.

    Returns:
        str: The corresponding phase name.
    """
    if phase < 0.03 or phase >= 0.97:
        return "new moon"
    if phase < 0.22:
        return "waxing crescent"
    if phase < 0.28:
        return "first quarter"
    if phase < 0.47:
        return "waxing gibbous"
    if phase < 0.53:
        return "full moon"
    if phase < 0.72:
        return "waning gibbous"
    if phase < 0.78:
        return "last quarter"

    return "waning crescent"


def celune_day_status(now: datetime.datetime) -> str:
    """Return a formatted Celune Day status message.

    Args:
        now: The current date and time.

    Returns:
        str: The formatted Celune Day status message.
    """
    celune_day_this_year = datetime.datetime(now.year, 6, 2)

    if now.date() == celune_day_this_year.date():
        return f"Today is Celune Day {now.year}"

    if now > celune_day_this_year:
        next_celune_day = datetime.datetime(now.year + 1, 6, 2)
    else:
        next_celune_day = celune_day_this_year

    days_until = (next_celune_day.date() - now.date()).days
    suffix = "s" if days_until != 1 else ""
    return f"{days_until} day{suffix} until Celune Day {next_celune_day.year}"


def range_interpolated(
    value: float, lo: Union[int, float], hi: Union[int, float], power: float = 3.0
) -> Union[int, float]:
    """Get interpolated number within a specified range.

    Args:
        value: The number (0-1) to convert to interpolated value.
        lo: The lower bound of the interpolated range.
        hi: The upper bound of the interpolated range.
        power: How strongly to interpolate the number.

    Returns:
        Union[int, float]: The interpolated number.
    """
    clamped = max(0.0, min(1.0, value))
    value = clamped**power
    return lo + value * (hi - lo)


def cuda_architecture(capability: tuple[int, int]) -> str:
    """Convert a CUDA capability tuple to an architecture name.

    Args:
        capability: CUDA capability formatted as tuple.

    Returns:
        str: The architecture name.

    Raises:
        NotImplementedError: The CUDA capability is below Celune's supported minimum.
        ValueError: The CUDA capability is not recognized.
    """

    major, minor = capability

    if major in [10, 11, 12] and minor == 0:  # recommended family
        return "Blackwell"
    if major == 9 and minor == 0:
        return "Hopper"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 8 and minor in [0, 6, 7]:  # CELINE INVADED THE CUDA ZONE!
        return "Ampere"
    if major < 8:  # too old
        raise NotImplementedError("capability not supported")

    raise ValueError(
        "invalid capability"
    )  # non-CUDA GPU reported a capability not known to Celune


def run_async(
    func: Callable, *args, daemon: bool = True, **kwargs
) -> multiprocessing.Process:
    """Run a function asynchronously. The function must not return a value or affect Celune directly, because it will
    run detached from Celune.

    Args:
        func: The function to call.
        daemon: Whether to use a daemon process. Defaults to ``True``.
        args: The arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        multiprocessing.Process: The process object.
    """
    proc = multiprocessing.Process(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
    )
    proc.start()
    return proc


def supports_ansi() -> bool:
    """Does the terminal support ANSI color codes?

    Returns:
        bool: Whether the terminal supports ANSI color codes.
    """
    return terminal_supports_ansi()


def format_error(e: Exception, dev: bool) -> str:
    """Format an error message.

    Args:
        e: The exception to format.
        dev: Whether developer mode is enabled.

    Returns:
        str: Either the full traceback or the exception text.
    """
    if dev:
        trace = traceback.format_exc()
        with open(traceback_path(create_parent=True), "w", encoding="utf-8") as f:
            f.write(trace)

    details = str(e) or "no error description"
    return traceback.format_exc() if dev else details


def indent(text: str, spaces: int, direction: Literal["left", "right"] = "left") -> str:
    """Indent a string from left or the right.

    Args:
        text: The text to indent.
        spaces: How many spaces to indent with.
        direction: The direction to indent from, must be horizontal.

    Returns:
        str: The indented string.

    Raises:
        ValueError: Invalid indenting direction.
    """
    if direction == "left":
        return " " * spaces + text
    if direction == "right":
        return text + " " * spaces

    raise ValueError("can't indent from this direction")


def get_caller() -> Optional[CallerInfo]:
    """Get information on the caller importing a package.

    Returns:
        dict: The caller's information.
    """
    for frame in inspect.stack():
        filename = frame.filename
        current = Path(__file__).resolve().parts[-2]

        if "importlib" in filename or filename.startswith("<frozen"):
            continue

        if current in filename:
            continue

        # need to explicitly annotate this else PyCharm does not recognize the type of the return value
        info: CallerInfo = {
            "function": frame.function,
            "filename": frame.filename,
            "line": frame.lineno,
        }

        return info

    return None


def caller_is_repl() -> bool:
    """Is the caller importing Celune the Python REPL?

    Returns:
        bool: If the caller is the Python REPL.
    """
    caller = get_caller()

    if caller is not None:
        return caller["filename"].startswith("<python-input-")

    return False


def detected_ide() -> Optional[str]:
    """Return a known IDE name from common environment markers.

    Returns:
        Optional[str]: The recognized IDE name, or ``None`` when no supported marker is present.
    """
    if os.environ.get("PYCHARM_HOSTED"):
        return "PyCharm"
    if os.environ.get("TERM_PROGRAM", "").casefold() == "vscode":
        return "VS Code"
    return None


def title_case(text: str) -> str:
    """Return a title-cased version of the input string.

    Args:
        text: The text to title-case.

    Returns:
        str: The title-cased string.
    """

    return text[0].upper() + text[1:]


def ipa_to_english(ipa: str) -> tuple[str, int]:
    """Return an English approximation of the input IPA. The output may be inaccurate with non-English IPA inputs.

    Args:
        ipa: The IPA to approximate.

    Returns:
        tuple[str, int]: The English approximation of the input IPA, and the amount of unmatched IPA characters.
    """

    ipa_map = {
        # consonants
        "p": "p",
        "b": "b",
        "t": "t",
        "d": "d",
        "k": "k",
        "g": "g",
        "m": "m",
        "n": "n",
        "ŋ": "ng",
        "tʃ": "ch",
        "dʒ": "j",
        "f": "f",
        "v": "v",
        "θ": "th",
        "ð": "dh",
        "s": "s",
        "z": "z",
        "ʃ": "sh",
        "ʒ": "zh",
        "h": "h",
        "w": "w",
        "j": "y",
        "r": "r",
        "l": "l",
        "ɾ": "d",
        "ɲ": "ny",
        "ç": "hy",
        "ʎ": "ly",
        "ʁ": "r",
        "χ": "h",
        # vowels
        "i": "ee",
        "ɪ": "ih",
        "e": "eh",
        "eɪ": "ay",
        "ɛ": "eh",
        "æ": "a",
        "ʌ": "uh",
        "ə": "uh",  # schwa is ambiguous, not guaranteed to be correct in all cases
        "u": "oo",
        "ʊ": "u",
        "oʊ": "oh",
        "ɔ": "aw",
        "ɑ": "ah",
        "aɪ": "ai",
        "aʊ": "ow",
        "ɔɪ": "oi",
        "y": "ee",
        # uncommon
        "x": "h",
        "ʔ": "-",
        "ɚ": "er",
        "ɝ": "er",
        "ɹ": "r",
        # marks
        "ˈ": "",
        "ˌ": "",
        "ː": "",
        ".": "",
    }

    ipa = ipa.strip("/[]")
    result = []
    i = 0
    unmatched = 0

    keys = sorted(ipa_map, key=len, reverse=True)
    while i < len(ipa):
        for key in keys:
            if ipa.startswith(key, i):
                result.append(ipa_map[key])
                i += len(key)
                break
        else:
            ch = ipa[i]
            if ch == " ":
                result.append("-")
            else:
                result.append(ch)
                unmatched += 1

            i += 1

    return "".join(result), unmatched


def replace_ipa(text: str, strict: bool = True) -> tuple[str, int]:
    """Return an English approximation of the input IPA.

    Args:
        text: The IPA to approximate.
        strict: Whether the input text must be delimited by IPA brackets (slashes or square brackets) to be treated as
            IPA or not.

    Returns:
        tuple[str, int]: The English approximation of the input IPA, and the amount of unmatched IPA characters.
    """
    total_unmatched = 0

    def repl(rmatch: re.Match[str]) -> str:
        nonlocal total_unmatched

        ipa = rmatch.group(1) or rmatch.group(2) or ""

        # PyCharm loves its "TYPO" warnings, but this is an IPA dictionary, not a word!
        ipa_markers = set("ŋʃʒθðɾɲçʎʁχɪɛæʌəʊɔɑɚʔɝɹˈˌː.")

        if strict and not ipa_markers.intersection(ipa):
            return rmatch.group(0)

        converted, unmatched = ipa_to_english(rmatch.group(0))
        total_unmatched += unmatched

        return converted

    result = re.sub(r"/([^/\[\]]+)/|\[([^/\[\]]+)]", repl, text)
    return result, total_unmatched


def custom_assert(condition: bool, exception: Optional[Exception]) -> None:
    """Assert a condition and raise a given exception if not met.

    Args:
        condition: The condition to assert against.
        exception: The exception to raise if the condition was not met.

    Raises:
        exception: An exception object was raised directly.
        AssertionError: An exception class was not specified, while assertion failed.
        TypeError: An object was specified to be raised that was not an instance of Exception.
        Exception: A specified exception class was raised because assertion failed.
    """
    if not condition:
        if isinstance(exception, Exception):
            raise exception
        if isinstance(exception, type) and issubclass(exception, BaseException):
            raise exception()
        if exception is None:
            raise AssertionError
        raise TypeError(
            f"expected an instance of Exception or None, got {type(exception).__name__}"
        )


def typing_delay(char: str) -> float:
    """Return the typing animation delay for one character.

    Args:
        char: The character that should influence the delay.

    Returns:
        float: Delay in seconds before the character appears.
    """
    if char in ".!?":
        rand_delay = random.uniform(0.25, 0.45)
    elif char in ",;:":
        rand_delay = random.uniform(0.12, 0.22)
    elif char == " ":
        rand_delay = random.uniform(0.02, 0.05)
    else:
        rand_delay = 0.06 + random.uniform(0.0, 0.1)

    return 0.08 + rand_delay


def typing_animation(text: str) -> Iterator[str]:
    """Iterate over an input string in a typing-like way.

    Args:
        text: The text to iterate over.

    Returns:
        Iterator[str]: An iterator that yields characters in a typing-like way.
    """

    for char in text:
        time.sleep(typing_delay(char))
        yield char


def detect_language(text: str, supported: list[str]) -> LanguageResult:
    """Detect possible languages in input text and report if it is in the supported language list.

    Args:
        text: The text to detect language of.
        supported: A list of supported languages to check against.

    Returns:
        LanguageResult: The language detection result metadata object.
    """

    try:
        main_lang = langdetect.detect(text)
        possible_langs = langdetect.detect_langs(text)
        probabilities = {}

        for lang in possible_langs:
            probabilities[lang.lang] = lang.prob

        result: LanguageResult = {
            "language": main_lang,
            "languages": list(probabilities.keys()),
            "supported": main_lang in supported,
            "probabilities": probabilities,
        }

        return result
    except langdetect.LangDetectException:
        result: LanguageResult = {
            "language": "en",
            "languages": ["en"],
            "supported": "en" in supported,
            "probabilities": {"en": 1.0},
        }

        return result


def is_april_fools() -> bool:
    """Is today April Fools?

    Returns:
        bool: Whether today is April Fools.
    """
    now = datetime.datetime.now()
    return now.month == 4 and now.day == 1


def is_celune_day() -> bool:
    """Is today Celune Day? (June 2nd).

    Returns:
        bool: Whether today is Celune Day.
    """
    now = datetime.datetime.now()
    return now.month == 6 and now.day == 2


def rng_replace(
    text: str, targets: list[str], replacements: list[str], rate: float = 0.5
) -> str:
    """Replace text with given replacements according to a set probability.

    Args:
        text: The input string.
        targets: A list of words/phrases to search for.
        replacements: A list of potential replacement strings.
        rate: The random probability (0.0 to 1.0) at which text will be replaced.

    Returns:
        str: The string with randomized, case-preserved replacements.
    """
    if not targets or not replacements:
        return text

    pattern_str = r"\b(" + "|".join(re.escape(t) for t in targets) + r")\b"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    def repl(rmatch: re.Match) -> str:
        source = rmatch.group(0)

        if random.random() >= rate:
            if isinstance(source, bytes):
                return source.decode("utf-8")
            return source

        target = random.choice(replacements)

        if source.isupper():
            return target.upper()

        if source[:1].isupper():
            return title_case(target)

        return target.lower()

    return pattern.sub(repl, text)


# the below functions are the only functions in Celune that use the 'Any' type
# 'Any' is okay for the type annotation of celune.utils.discard()


@overload
def discard(val: Any) -> None:
    """Overload #1 for the implementation of celune.utils.discard().

    Args:
        val: A discardable value.
    """


@overload
def discard(val: Any, attr: str) -> None:
    """Overload #2 for the implementation of celune.utils.discard().

    Args:
        val: A discardable value.
        attr: A discardable attribute on an object.
    """


def discard(val: Any, attr: Optional[str] = None) -> None:
    """Discard a value or clear an explicitly named attribute.

    Args:
        val: Value to discard, or the attribute owner.
        attr: Optional attribute name to clear on ``val``.
    """
    if attr is not None:
        setattr(val, attr, None)
        return

    _ = val
    del _


def is_port_usable(port: int) -> bool:
    """Check if a port can be bound.

    Args:
        port: The port to check.

    Returns:
        bool: Whether the port can be bound.
    """
    try:
        for conn in psutil.net_connections():
            laddr = conn.laddr

            if not isinstance(laddr, tuple) and laddr.port == port:
                return False

        return True
    except (psutil.Error, OSError):
        return False


def make_persona_card(
    name: str,
    age: str,
    gender: str,
    persona: str,
    traits: dict[str, str],
    context: str,
    voice: str,
) -> str:
    """Return a persona card for the current character.

    Args:
        name: The character name.
        age: The character's age.
        gender: The character's gender or LGBT type.
        persona: The character's personality description.
        traits: The character's trait values.
        context: Additional context information for the character.
        voice: The character's selected voice type.

    Returns:
        str: The formatted persona card.

    Raises:
        ValueError: The Persona card has missing or invalid traits.
    """
    required_traits = ("warmth", "directness", "humor", "detail")
    missing_traits = [trait for trait in required_traits if trait not in traits]
    if missing_traits:
        raise ValueError(f"persona card is missing traits: {', '.join(missing_traits)}")

    unknown_traits = [trait for trait in traits if trait not in required_traits]
    if unknown_traits:
        raise ValueError(
            f"undefined trait for persona card: {', '.join(unknown_traits)}"
        )

    base_card = """
    # Speaker Profile

    Name: {name}
    Age: {age}
    Gender: {gender}

    ## Personality
    {persona}

    ## Response Style
    - Warmth: {warmth}
    - Directness: {directness}
    - Humor: {humor}
    - Detail: {detail}

    ## Context
    {context}

    ## Voice
    {voice}
    """

    custom_assert(bool(name.strip()), ValueError("persona card name cannot be empty"))
    assert bool(name.strip())

    return (
        textwrap.dedent(base_card)
        .strip()
        .format(
            name=name,
            age=age,
            gender=gender,
            persona=persona,
            warmth=traits["warmth"],
            directness=traits["directness"],
            humor=traits["humor"],
            detail=traits["detail"],
            context=context,
            voice=voice,
        )
    )


def raise_test() -> None:
    """Raise a testing exception. This is used only in development.

    Raises:
        RuntimeError: A testing exception.
    """

    raise RuntimeError("testing exception")


def normalize_special_characters(text: str) -> str:
    """Normalize special characters in input string for TTS.

    Args:
        text: The text to normalize.

    Returns:
        str: The normalized text.
    """

    special_char_mappings = str.maketrans(
        {
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u201e": '"',  # double low quote
            "\u2018": "'",  # left single quote
            "\u2019": "'",  # right single quote
            "\u2013": " - ",  # en dash
            "\u2014": " - ",  # em dash
            "\u2026": "...",  # ellipsis
        }
    )

    return text.translate(special_char_mappings).replace("  ", " ")
