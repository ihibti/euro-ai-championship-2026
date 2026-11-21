"""Validated prediction CSVs.

The column names and value range default to the format shown in organizer
slides from an earlier event (`id,label`, label = probability in [0, 1]).
This event's actual submission format is TBD; adjust the arguments once known.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path


def validate_predictions(
    ids: Sequence[object],
    values: Sequence[float],
    value_range: tuple[float, float] | None = (0.0, 1.0),
) -> None:
    if len(ids) != len(values):
        raise ValueError(f"{len(ids)} ids but {len(values)} values")
    if len(ids) == 0:
        raise ValueError("no predictions")
    if len({str(i) for i in ids}) != len(ids):
        raise ValueError("ids must be unique")
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite value {value!r}")
        if value_range is not None and not value_range[0] <= number <= value_range[1]:
            raise ValueError(f"value {number} outside {value_range}")


def write_predictions_csv(
    path: str | Path,
    ids: Sequence[object],
    values: Sequence[float],
    *,
    id_column: str = "id",
    value_column: str = "label",
    value_range: tuple[float, float] | None = (0.0, 1.0),
) -> Path:
    validate_predictions(ids, values, value_range)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, value_column])
        writer.writerows((i, repr(float(v))) for i, v in zip(ids, values, strict=True))
    return path


def read_predictions_csv(
    path: str | Path, *, id_column: str = "id", value_column: str = "label"
) -> tuple[list[str], list[float]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row[id_column] for row in rows], [float(row[value_column]) for row in rows]
