# SPDX-License-Identifier: Apache-2.0
"""Convert X-SAMPA transcription symbols to their IPA equivalents."""

from typing import Optional

_SYMBOLS = {
    '"': "ˈ",
    "%": "ˌ",
    "!": "ꜜ",
    "^": "ꜛ",
    "&": "ɶ",
    "1": "ɨ",
    "2": "ø",
    "3": "ɜ",
    "3\\": "ɞ",
    "4": "ɾ",
    "5": "ɫ",
    "6": "ɐ",
    "7": "ɤ",
    "8": "ɵ",
    "9": "œ",
    "@": "ə",
    "@\\": "ɘ",
    "A": "ɑ",
    "B": "β",
    "B\\": "ʙ",
    "C": "ç",
    "D": "ð",
    "E": "ɛ",
    "F": "ɱ",
    "G": "ɣ",
    "G\\": "ɢ",
    "H": "ɥ",
    "I": "ɪ",
    "J": "ɲ",
    "J\\": "ɟ",
    "K": "ɬ",
    "K\\": "ɮ",
    "L": "ʎ",
    "L\\": "ʟ",
    "M": "ɯ",
    "M\\": "ɰ",
    "N": "ŋ",
    "N\\": "ɴ",
    "O": "ɔ",
    "O\\": "ʘ",
    "P": "ʋ",
    "Q": "ɒ",
    "R": "ʁ",
    "R\\": "ʀ",
    "S": "ʃ",
    "T": "θ",
    "U": "ʊ",
    "V": "ʌ",
    "W": "ʍ",
    "X": "χ",
    "X\\": "ħ",
    "Y": "ʏ",
    "Z": "ʒ",
    "?": "ʔ",
    "?\\": "ʕ",
    "||\\": "ǁ",
    "!\\": "ǃ",
    "=\\": "ǂ",
    "|\\": "ǀ",
    "<\\": "ʢ",
    "H\\": "ʜ",
    ">\\": "ʡ",
    "{": "æ",
    "|": "|",
    "||": "‖",
    "}": "ʉ",
    "~": "̃",
    ":": "ː",
    ":\\": "ˑ",
    "::": "ːː",
    ".": ".",
    "-": " ",
    "_": "͡",
    "_+": "̟",
    "_-": "̠",
    '_"': "̈",
    "_0": "̥",
    "_A": "̘",
    "_B": "˩",
    "_F": "˥˩",
    "_G": "ˠ",
    "_H": "˦",
    "_L": "˨",
    "_M": "˧",
    "_O": "̹",
    "_R": "˩˥",
    "_T": "˥",
    "_a": "̺",
    "_c": "̜",
    "_d": "̪",
    "_e": "ˠ",
    "_h": "ʰ",
    "_j": "ʲ",
    "_k": "̰",
    "_l": "ˡ",
    "_m": "̻",
    "_n": "ⁿ",
    "_o": "̞",
    "_q": "̙",
    "_r": "̝",
    "_t": "̤",
    "_v": "̬",
    "_w": "ʷ",
    "_^": "̯",
    "_~": "̃",
    "_}": "̚",
    "_>": "ʼ",
    "_`": "˞",
    "_\\": "ˤ",
    "_X": "̆",
    "_=": "̩",
    "_?\\": "ˤ",
    "_R_F": "˧˦˧",
    "-\\": "‿",
    "<R>": "↗",
    "<F>": "↘",
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
    "'": "ʲ",
    "aI": "aɪ",
    "aU": "aʊ",
    "eI": "eɪ",
    "oU": "oʊ",
    "OI": "ɔɪ",
    "@U": "əʊ",
    "tS": "tʃ",
    "dZ": "dʒ",
    "n`": "ɳ",
    "t`": "ʈ",
    "d`": "ɖ",
    "r`": "ɽ",
    "s`": "ʂ",
    "z`": "ʐ",
    "l`": "ɭ",
    "r\\`": "ɻ",
    "r\\": "ɹ",
    "s\\": "ɕ",
    "z\\": "ʑ",
    "j\\": "ʝ",
    "p\\": "ɸ",
    "h\\": "ɦ",
    "v\\": "ʋ",
    "l\\": "ɺ",
    "d_<": "ɗ",
    "b_<": "ɓ",
    "J\\_<": "ʄ",
    "G\\_<": "ʛ",
    "g_<": "ɠ",
}

_SYMBOLS_BY_LENGTH = tuple(sorted(_SYMBOLS, key=len, reverse=True))
_IPA_VOWELS = frozenset(
    [
        "i",
        "y",
        "ɨ",
        "ʉ",
        "ɯ",
        "u",
        "ɪ",
        "ʏ",
        "ʊ",
        "e",
        "ø",
        "ɘ",
        "ɵ",
        "ɤ",
        "o",
        "ə",
        "ɛ",
        "œ",
        "ɜ",
        "ɞ",
        "ʌ",
        "ɔ",
        "æ",
        "ɐ",
        "a",
        "ɶ",
        "ɑ",
        "ɒ",
    ]
)
_VOWEL_SYMBOLS = frozenset(
    symbol for symbol, ipa in _SYMBOLS.items() if ipa and ipa[0] in _IPA_VOWELS
)
_NUCLEUS_MODIFIERS = frozenset(
    symbol
    for symbol in _SYMBOLS
    if symbol.startswith("_") or symbol in {"~", ":", ":\\", "::"}
)


def _symbol_at(value: str, offset: int) -> Optional[str]:
    return next(
        (
            candidate
            for candidate in _SYMBOLS_BY_LENGTH
            if value.startswith(candidate, offset)
        ),
        None,
    )


def to_ipa(value: str) -> str:
    """Convert standard X-SAMPA symbols to Unicode IPA.

    Args:
        value: ASCII X-SAMPA transcription.

    Returns:
        str: IPA transcription with syllable separators preserved.

    Raises:
        ValueError: If the input is not ASCII or contains an unknown symbol.
    """
    if not value.isascii():
        raise ValueError("X-SAMPA transcription must contain ASCII characters only")

    converted: list[str] = []
    offset = 0
    while offset < len(value):
        if value[offset].isspace():
            converted.append(value[offset])
            offset += 1
            continue
        symbol = _symbol_at(value, offset)
        if symbol is None:
            raise ValueError(
                f"unsupported X-SAMPA symbol at character {offset + 1}: "
                f"{value[offset]!r}"
            )
        converted.append(_SYMBOLS[symbol])
        offset += len(symbol)

    return "".join(converted)


def vowel_nucleus_start(value: str) -> int:
    """Return the X-SAMPA offset of one final vowel nucleus.

    Args:
        value: One X-SAMPA syllable with an optional consonant onset.

    Returns:
        int: Character offset where the vowel nucleus begins.

    Raises:
        ValueError: If the syllable has multiple nuclei, a coda, or invalid text.
    """
    if (
        not value
        or not value.isascii()
        or "." in value
        or any(ch.isspace() for ch in value)
    ):
        raise ValueError("slide pronunciation must be one ASCII X-SAMPA syllable")

    symbols: list[tuple[str, int]] = []
    offset = 0
    while offset < len(value):
        symbol = _symbol_at(value, offset)
        if symbol is None:
            raise ValueError(
                f"unsupported X-SAMPA symbol at character {offset + 1}: "
                f"{value[offset]!r}"
            )
        symbols.append((symbol, offset))
        offset += len(symbol)

    vowel_indices = tuple(
        index
        for index, (symbol, _start) in enumerate(symbols)
        if symbol in _VOWEL_SYMBOLS
    )
    if len(vowel_indices) != 1:
        raise ValueError("slide pronunciation must contain one vowel nucleus")

    vowel_index = vowel_indices[0]
    if any(
        symbol not in _NUCLEUS_MODIFIERS
        for symbol, _start in symbols[vowel_index + 1 :]
    ):
        raise ValueError("slide pronunciation must end with its vowel nucleus")
    return symbols[vowel_index][1]
