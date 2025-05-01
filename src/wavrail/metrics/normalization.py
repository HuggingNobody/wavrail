"""Unicode-safe text normalization helpers.

Three small utilities are exposed:

- :func:`fullwidth_to_halfwidth` and :func:`halfwidth_to_fullwidth`
  for the common Chinese full-width / ASCII half-width digit and
  letter conversion.
- :func:`normalize_digits` for converting between Arabic digits
  and Chinese numerals in either direction.

A :class:`TextNormalizer` composes these and adds punctuation
stripping, case folding, and whitespace squashing so the same
reference and hypothesis strings are produced regardless of
the punctuation or capitalisation convention used in the
underlying text.
"""

from __future__ import annotations

from collections.abc import Iterable

_FULLWIDTH_OFFSET = 0xFEE0
_ASCII_DIGIT_START = 0x30
_ASCII_DIGIT_END = 0x39
_ASCII_UPPER_START = 0x41
_ASCII_UPPER_END = 0x5A
_ASCII_LOWER_START = 0x61
_ASCII_LOWER_END = 0x7A
_ASCII_SPACE = 0x20

_CHINESE_DIGITS = "零一二三四五六七八九"


def fullwidth_to_halfwidth(text: str) -> str:
    """Map full-width ASCII forms to their ASCII equivalents."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
            continue
        delta = code - _FULLWIDTH_OFFSET
        is_digit = _ASCII_DIGIT_START <= delta <= _ASCII_DIGIT_END
        is_upper = _ASCII_UPPER_START <= delta <= _ASCII_UPPER_END
        is_lower = _ASCII_LOWER_START <= delta <= _ASCII_LOWER_END
        if is_digit or is_upper or is_lower:
            out.append(chr(delta))
        else:
            out.append(ch)
    return "".join(out)


def halfwidth_to_fullwidth(text: str) -> str:
    """Map ASCII letters, digits, and punctuation to full-width forms."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == " ":
            out.append("　")
            continue
        if (
            _ASCII_DIGIT_START <= code <= _ASCII_DIGIT_END
            or _ASCII_UPPER_START <= code <= _ASCII_UPPER_END
            or _ASCII_LOWER_START <= code <= _ASCII_LOWER_END
        ):
            out.append(chr(code + _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def normalize_digits(text: str, *, to_chinese: bool = False) -> str:
    """Convert digits between Arabic and Chinese numeral forms.

    When ``to_chinese`` is true, Arabic digits ``0``-``9`` are
    mapped to Chinese numerals ``零``-``九``. When false, Chinese
    numerals are mapped back to Arabic digits.
    """
    if to_chinese:
        arabic_to_chinese = str.maketrans("0123456789", _CHINESE_DIGITS)
        return text.translate(arabic_to_chinese)
    chinese_to_arabic = str.maketrans(_CHINESE_DIGITS, "0123456789")
    return text.translate(chinese_to_arabic)


_ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_CJK_PUNCTUATION = (
    "\uff0c\uff01\uff1f\uff1b\uff1a\u300c\u300d\u300e\u300f\uff08\uff09\u3010\u3011\u300a\u300b"
)
_PUNCTUATION = set(_ASCII_PUNCTUATION + _CJK_PUNCTUATION)


class TextNormalizer:
    """Compose a fixed sequence of normalisations."""

    def __init__(
        self,
        *,
        strip_punctuation: bool = True,
        case_fold: bool = True,
        fullwidth_normalization: bool = True,
        digits_to_chinese: bool = False,
    ) -> None:
        self.strip_punctuation = strip_punctuation
        self.case_fold = case_fold
        self.fullwidth_normalization = fullwidth_normalization
        self.digits_to_chinese = digits_to_chinese

    def __call__(self, text: str) -> str:
        if text is None:
            raise TypeError("text must be a string")
        if self.fullwidth_normalization:
            text = fullwidth_to_halfwidth(text)
        if self.case_fold:
            text = text.casefold()
        if self.strip_punctuation:
            text = "".join(ch for ch in text if ch not in _PUNCTUATION)
        if self.digits_to_chinese:
            text = normalize_digits(text, to_chinese=True)
        return " ".join(text.split())

    def normalize_many(self, texts: Iterable[str]) -> list[str]:
        return [self(text) for text in texts]


__all__ = [
    "TextNormalizer",
    "fullwidth_to_halfwidth",
    "halfwidth_to_fullwidth",
    "normalize_digits",
]
