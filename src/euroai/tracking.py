"""Offline JSON-lines metric logging (works on nodes without internet access)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._handle = self.path.open("a")

    def log(self, step: int | None = None, **metrics: Any) -> dict[str, Any]:
        record = {"time": self._clock(), "step": step, **metrics}
        self._handle.write(json.dumps(record, default=float) + "\n")
        self._handle.flush()
        return record

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
