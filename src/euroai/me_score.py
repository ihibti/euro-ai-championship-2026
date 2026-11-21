"""Local model-equality score: the organizers' practice MCC over endpoint pairs."""

from __future__ import annotations

import csv
import io
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

HEADER = ["endpoint", "family", "model"]
Labels = Mapping[str, tuple[str, str]]


@dataclass(frozen=True)
class PairScore:
    mcc: float
    tp: int
    fp: int
    fn: int
    tn: int
    wrong: list[tuple[str, str]] = field(default_factory=list)


def parse_labels(text: str, prefix: str | None = None) -> dict[str, tuple[str, str]]:
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if rows and rows[0][:3] == HEADER:
        rows = rows[1:]
    labels: dict[str, tuple[str, str]] = {}
    for number, row in enumerate(rows, start=1):
        if len(row) < 3:
            raise ValueError(f"row {number} has {len(row)} fields, expected endpoint,family,model")
        endpoint, family, model = row[0], row[1], row[2]
        if prefix is not None and not endpoint.startswith(prefix):
            continue
        if endpoint in labels:
            raise ValueError(f"endpoint {endpoint} appears twice")
        labels[endpoint] = (family, model)
    return labels


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0


def same_model_mcc(truth: Labels, guess: Labels) -> PairScore:
    missing = sorted(set(truth) - set(guess))
    if missing:
        raise ValueError(f"no answer for: {', '.join(missing)}")
    tp = fp = fn = tn = 0
    wrong: list[tuple[str, str]] = []
    for first, second in itertools.combinations(sorted(truth), 2):
        actual = truth[first] == truth[second]
        predicted = guess[first] == guess[second]
        tp += actual and predicted
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not actual and not predicted
        if actual != predicted:
            wrong.append((first, second))
    return PairScore(mcc(tp, fp, fn, tn), tp, fp, fn, tn, wrong)
