"""Answer length and cap-hit rate per prompt and endpoint, each read against the pooled
same-model copies (E12).

uv run python tasks/model_equality/analysis/lengths.py --answers \
    tasks/model_equality/answers/round1.jsonl tasks/model_equality/answers/night-a.jsonl \
    tasks/model_equality/answers/night-b.jsonl

For a value v (median characters, or cap-hit rate), the gap of an endpoint is its own v minus
the v of the pooled answers of the same-model copies (for a copy itself: of the other copies).
The spread is the largest absolute gap among the copies; another model passes on a prompt when
its absolute gap exceeds the spread.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows  # noqa: E402
from stop import cap_hit  # noqa: E402

SAME = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
OTHERS = (("ref-02", "fp8"), ("ref-09", "L0.3"), ("ref-04", "L0.7"), ("ref-07", "L1.5"))


@dataclass(frozen=True)
class Stats:
    median_chars: float
    cap_rate: float
    count: int


def answers(
    rows: Iterable[Mapping[str, object]], prompt_id: str, prefix: str
) -> dict[str, list[str]]:
    cells: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        endpoint = str(r["endpoint"])
        if r["id"] != prompt_id or not endpoint.startswith(prefix) or r.get("response") is None:
            continue
        cells[endpoint].append((int(r["repeat"]), str(r["response"])))
    return {e: [t for _, t in sorted(v)] for e, v in sorted(cells.items())}


def endpoint_stats(texts: Mapping[str, Sequence[str]]) -> dict[str, Stats]:
    return {
        e: Stats(
            median_chars=float(np.median([len(t) for t in v])),
            cap_rate=float(np.mean([cap_hit(t) for t in v])),
            count=len(v),
        )
        for e, v in texts.items()
        if v
    }


def pooled_median(groups: Sequence[Sequence[float]]) -> float:
    return float(np.median([x for g in groups for x in g]))


def pooled_rate(groups: Sequence[Sequence[bool]]) -> float:
    return float(np.mean([x for g in groups for x in g]))


def gaps(
    values: Mapping[str, Sequence],
    same: Sequence[str],
    pool: Callable[[Sequence[Sequence]], float],
) -> tuple[dict[str, float], float]:
    members = [s for s in same if s in values]
    gap: dict[str, float] = {}
    for e, own in values.items():
        rest = [values[s] for s in members if s != e]
        gap[e] = pool([own]) - pool(rest)
    spread = max(abs(gap[s]) for s in members)
    return gap, spread


def cell(gap: Mapping[str, float], spread: float, endpoint: str, digits: int) -> str:
    if endpoint not in gap:
        return "-"
    mark = " *" if abs(gap[endpoint]) > spread else ""
    return f"{gap[endpoint]:+.{digits}f}{mark}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    args = parser.parse_args(argv)
    rows = load_rows(args.answers)
    prompt_ids = list(dict.fromkeys(str(r["id"]) for r in rows))
    names = " | ".join(n for _, n in OTHERS)
    print("Median characters per answer: plain; gap of each model from the pooled copies; spread")
    print(f"| prompt | n | plain | {names} | copies spread |")
    print("|---|---|---|---|---|---|---|---|")
    rates: list[str] = []
    for prompt_id in prompt_ids:
        texts = answers(rows, prompt_id, args.prefix)
        stats = endpoint_stats(texts)
        if not all(s in stats for s in SAME):
            continue
        n = min(s.count for s in stats.values())
        chars = {e: [float(len(t)) for t in v] for e, v in texts.items()}
        gap, spread = gaps(chars, SAME, pooled_median)
        cols = " | ".join(cell(gap, spread, e, 0) for e, _ in OTHERS)
        plain = f"{stats['ref-01'].median_chars:.0f}"
        print(f"| {prompt_id} | {n} | {plain} | {cols} | {spread:.0f} |")
        hits = {e: [cap_hit(t) for t in v] for e, v in texts.items()}
        rgap, rspread = gaps(hits, SAME, pooled_rate)
        own = " ".join(
            str(round(stats[e].cap_rate * stats[e].count)) for e in (*SAME, *(o for o, _ in OTHERS))
        )
        rcols = " | ".join(cell(rgap, rspread, e, 2) for e, _ in OTHERS)
        rates.append(f"| {prompt_id} | {n} | {own} | {rcols} | {rspread:.2f} |")
    print("\nCap hits per endpoint (01 03 05 06 08 02 09 04 07); rate gap from the pooled copies")
    print(f"| prompt | n | cap hits | {names} | copies spread |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rates))
    print("\n* = absolute gap above the spread among the five same-model copies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
