"""Per-prompt feature extractors: answer text to a dict of named observations."""

from __future__ import annotations

import re
from collections import Counter

from melib import first_letter, first_number, parse_numbered

WORD = re.compile(r"[a-z']+")


def char_ngrams(text: str, n: int = 3) -> dict[str, float]:
    clean = re.sub(r"\s+", " ", text.lower())
    grams = Counter(clean[i : i + n] for i in range(max(0, len(clean) - n + 1)))
    total = sum(grams.values()) or 1
    return {f"c:{gram}": count / total for gram, count in grams.items()}


def words(text: str) -> dict[str, float]:
    tokens = WORD.findall(text.lower())
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {f"w:{token}": count / total for token, count in counts.items()}


def items_categorical(text: str, count: int, normalize) -> dict[str, str]:
    values = parse_numbered(text, count)
    out: dict[str, str] = {}
    for index, value in enumerate(values):
        normalized = normalize(value)
        if normalized is not None:
            out[f"i{index + 1}"] = normalized
    return out


def norm_letter(value: str | None) -> str | None:
    return first_letter(value, "ABC")


def norm_number(value: str | None) -> str | None:
    number = first_number(value)
    return None if number is None else str(int(number)) if number == int(number) else str(number)


def norm_word(value: str | None) -> str | None:
    if value is None:
        return None
    tokens = WORD.findall(value.lower())
    return tokens[0] if tokens else None


def digit_sequence(text: str, alphabet: str) -> list[str]:
    return [ch for ch in text if ch in alphabet]


def sequence_ngrams(symbols: list[str], n: int) -> dict[str, float]:
    grams = Counter("".join(symbols[i : i + n]) for i in range(max(0, len(symbols) - n + 1)))
    total = sum(grams.values()) or 1
    return {f"s{n}:{gram}": count / total for gram, count in grams.items()}
