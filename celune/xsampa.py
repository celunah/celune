# SPDX-License-Identifier: Apache-2.0
"""Convert X-SAMPA transcription symbols to their IPA equivalents."""

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
        symbol = next(
            (
                candidate
                for candidate in _SYMBOLS_BY_LENGTH
                if value.startswith(candidate, offset)
            ),
            None,
        )
        if symbol is None:
            raise ValueError(
                f"unsupported X-SAMPA symbol at character {offset + 1}: "
                f"{value[offset]!r}"
            )
        converted.append(_SYMBOLS[symbol])
        offset += len(symbol)

    return "".join(converted)
