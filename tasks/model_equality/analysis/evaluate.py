"""Score group.py on the reference endpoints at a fixed answer count, and resample that count.

uv run python tasks/model_equality/analysis/evaluate.py --per-prompt 4 --draws 200 \
    --answers tasks/model_equality/answers/round1.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import group  # noqa: E402
from melib import load_rows  # noqa: E402

from euroai.me_score import parse_labels, same_model_mcc  # noqa: E402

TRUTH = ROOT / "tasks" / "model_equality" / "upstream" / "practice_answers.csv"


def truth_labels() -> dict[str, tuple[str, str]]:
    return parse_labels(TRUTH.read_text(encoding="utf-8"))


def subsample(rows: list[dict], per_prompt: int, rng: random.Random | None) -> list[dict]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["endpoint"], row["id"])].append(row)
    chosen: list[dict] = []
    for items in cells.values():
        items = sorted(items, key=lambda r: int(r["repeat"]))
        if rng is None:
            chosen.extend(items[:per_prompt])
        else:
            chosen.extend(rng.sample(items, min(per_prompt, len(items))))
    return chosen


def score_rows(rows: list[dict], **options) -> tuple[float, list[tuple[str, str, str]]]:
    labels = group.decide(rows, prefix="ref-", **options)
    guess = {endpoint: (family, model) for endpoint, family, model in labels}
    return same_model_mcc(truth_labels(), guess).mcc, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--per-prompt", type=int, default=4)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--kinds", default=",".join(group.MODEL_KINDS))
    parser.add_argument("--mode", default="pooled")
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    wanted = set(filter(None, args.ids.split(",")))
    rows = [
        r
        for r in load_rows(args.answers)
        if str(r["endpoint"]).startswith("ref-") and (not wanted or r["id"] in wanted)
    ]
    options = {"factor": args.factor, "kinds": tuple(args.kinds.split(",")), "mode": args.mode}
    first, labels = score_rows(subsample(rows, args.per_prompt, None), **options)
    print(f"first {args.per_prompt} repeats: MCC {first:.3f}")
    for endpoint, family, model in labels:
        print(f"  {endpoint} {family} {model}")
    if args.draws <= 0:
        return
    rng = random.Random(args.seed)
    values = [
        score_rows(subsample(rows, args.per_prompt, rng), **options)[0] for _ in range(args.draws)
    ]
    values_array = np.array(values)
    quantiles = np.quantile(values_array, [0.1, 0.5, 0.9])
    print(
        f"{args.draws} random draws of {args.per_prompt} repeats per prompt: mean "
        f"{values_array.mean():.3f}, sd {values_array.std():.3f}, "
        f"10/50/90% {quantiles[0]:.3f}/{quantiles[1]:.3f}/{quantiles[2]:.3f}, "
        f"min {values_array.min():.3f}, max {values_array.max():.3f}"
    )
    counts = defaultdict(int)
    for value in values:
        counts[round(value, 3)] += 1
    print("distribution:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
