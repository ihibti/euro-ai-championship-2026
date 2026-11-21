"""Exact-match and common-prefix statistics within and between endpoints, per prompt.

uv run python tasks/model_equality/analysis/collisions.py tasks/model_equality/answers/<f>.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import by_endpoint_prompt, load_rows  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prefix_len(first: str, second: str) -> int:
    return len(os.path.commonprefix([first, second]))


def within(texts: Sequence[str]) -> tuple[float, float]:
    pairs = list(itertools.combinations([normalize(t) for t in texts], 2))
    if not pairs:
        return float("nan"), float("nan")
    same = np.mean([a == b for a, b in pairs])
    prefix = np.mean([prefix_len(a, b) / max(1, min(len(a), len(b))) for a, b in pairs])
    return float(same), float(prefix)


def between(first: Sequence[str], second: Sequence[str]) -> tuple[float, float]:
    pairs = [(normalize(a), normalize(b)) for a in first for b in second]
    same = np.mean([a == b for a, b in pairs])
    prefix = np.mean([prefix_len(a, b) / max(1, min(len(a), len(b))) for a, b in pairs])
    return float(same), float(prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    parser.add_argument("--ids", default="")
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    grouped = by_endpoint_prompt(load_rows(args.answers))
    wanted = set(filter(None, args.ids.split(",")))
    prompts = sorted({p for _, p in grouped if not wanted or p in wanted})
    endpoints = sorted({e for e, _ in grouped})
    print("within-endpoint identical-pair rate per prompt (n = answers per endpoint)")
    print("prompt       " + " ".join(e[-2:].rjust(5) for e in endpoints))
    for prompt in prompts:
        cells = []
        for endpoint in endpoints:
            texts = grouped.get((endpoint, prompt), [])
            same, _ = within(texts)
            cells.append("    -" if np.isnan(same) else f"{same:5.2f}")
        n = min(len(grouped.get((e, prompt), [])) for e in endpoints)
        print(f"{prompt[:10]:10s} {n:2d}" + " ".join(cells))
    if args.matrix:
        print("\nbetween-endpoint identical rate, summed over prompts (diagonal: within)")
        print("      " + " ".join(e[-2:].rjust(5) for e in endpoints))
        for x in endpoints:
            cells = []
            for y in endpoints:
                values = []
                for prompt in prompts:
                    a = grouped.get((x, prompt), [])
                    b = grouped.get((y, prompt), [])
                    if not a or not b:
                        continue
                    values.append(within(a)[0] if x == y else between(a, b)[0])
                values = [v for v in values if not np.isnan(v)]
                cells.append(f"{np.mean(values):5.2f}" if values else "    -")
            print(f"{x[-2:]:>5s} " + " ".join(cells))


if __name__ == "__main__":
    main()
