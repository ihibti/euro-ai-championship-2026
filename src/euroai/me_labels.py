"""Rewrite model-equality labels in the organizers' format: family A, B, ...; model A1, A2, B1, ...

The organizers' update of 2026-11-21: the same letter marks the same family, and a model is the
family letter followed by a number. Only which endpoints share a label matters for the score, so
any grouping can be renamed without changing it.
"""

from __future__ import annotations

from collections.abc import Iterable

Row = tuple[str, str, str]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def family_letters(index: int) -> str:
    if index < 0:
        raise ValueError("index must be zero or more")
    letters = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, len(ALPHABET))
        letters = ALPHABET[remainder] + letters
    return letters


def canonical_labels(rows: Iterable[Row]) -> list[Row]:
    ordered = sorted(rows, key=lambda row: row[0])
    families: dict[str, str] = {}
    models: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    renamed: list[Row] = []
    for endpoint, family, model in ordered:
        if family not in families:
            families[family] = family_letters(len(families))
        letter = families[family]
        key = (family, model)
        if key not in models:
            counts[letter] = counts.get(letter, 0) + 1
            models[key] = f"{letter}{counts[letter]}"
        renamed.append((endpoint, letter, models[key]))
    return renamed


def parse_rows(text: str) -> list[Row]:
    rows: list[Row] = []
    for line in text.lstrip("\ufeff").splitlines():
        fields = [cell.strip() for cell in line.split(",")]
        if not any(fields) or fields[:3] == ["endpoint", "family", "model"]:
            continue
        if len(fields) < 3:
            raise ValueError(f"expected endpoint,family,model, got {line!r}")
        rows.append((fields[0], fields[1], fields[2]))
    return rows


def render_rows(rows: Iterable[Row]) -> str:
    lines = ["endpoint,family,model", *(",".join(row) for row in rows)]
    return "\n".join(lines) + "\n"
