"""Shared parsing and statistics for the model-equality analysis. Pure functions on answer rows."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LINE = re.compile(r"^\s*(\d{1,3})\s*[.):\-]\s*(.*?)\s*$")


def load_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_endpoint_prompt(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        if row.get("response") is None:
            continue
        grouped[(str(row["endpoint"]), str(row["id"]))].append(
            (int(row["repeat"]), str(row["response"]))
        )
    return {key: [text for _, text in sorted(items)] for key, items in grouped.items()}


def parse_numbered(text: str, count: int) -> list[str | None]:
    values: list[str | None] = [None] * count
    for line in text.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < count and values[index] is None:
            values[index] = match.group(2)
    return values


def first_letter(value: str | None, letters: str = "AB") -> str | None:
    if value is None:
        return None
    match = re.search(rf"\b([{letters}])\b", value)
    return match.group(1) if match else None


def first_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    keep = ~(np.isnan(first) | np.isnan(second))
    if keep.sum() < 3:
        return float("nan")
    a, b = rank(first[keep]), rank(second[keep])
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def item_matrix(texts: Sequence[str], count: int, convert) -> np.ndarray:
    matrix = np.full((len(texts), count), np.nan)
    for row, text in enumerate(texts):
        for column, value in enumerate(parse_numbered(text, count)):
            converted = convert(value)
            if converted is not None:
                matrix[row, column] = converted
    return matrix
