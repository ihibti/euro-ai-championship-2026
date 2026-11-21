"""Separation experiment on saved reference answers.

For every prompt and feature map, the distance between two endpoints is one minus the
normalised kernel similarity of their answers, K_xy / sqrt(K_xx K_yy), where K is the mean
cosine between answer feature vectors and K_xx uses distinct pairs only. Everything is computed
between half-samples (even against odd repeats), so the within-endpoint noise (one endpoint's
even half against its odd half) and the between-endpoint distances have the same sample size.

uv run python tasks/model_equality/analysis/explore.py tasks/model_equality/answers/round1.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import group  # noqa: E402
from melib import load_rows  # noqa: E402

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
QUANT = "ref-02"
LORA = {"L.3": "ref-09", "L.7": "ref-04", "L1.5": "ref-07"}


def tail_only(text: str) -> group.Vector:
    clean = group.squash(text)
    return group.unit({f"tail={clean[-3:]}": 1.0})


def head_only(text: str) -> group.Vector:
    clean = group.squash(text)
    return group.unit({f"head={clean[:12]}": 1.0})


def full_only(text: str) -> group.Vector:
    return group.unit({f"full={group.squash(text)}": 1.0})


def char3(text: str) -> group.Vector:
    clean = group.squash(text)
    counts: dict[str, float] = {}
    for i in range(max(0, len(clean) - 2)):
        counts[clean[i : i + 3]] = counts.get(clean[i : i + 3], 0.0) + 1.0
    return group.unit(counts)


FEATURE_MAPS: dict[str, Callable[[str], group.Vector]] = {
    "seq": group.seq_features,
    "full": full_only,
    "head": head_only,
    "tail": tail_only,
    "bat": group.bat_features,
    "words": group.txt_features,
    "char3": char3,
}


def kernel(first: Sequence[group.Vector], second: Sequence[group.Vector], same: bool) -> float:
    if same:
        return group.self_mean(first)
    return group.cross_mean(first, second)


def norm_sim(a: Sequence[group.Vector], b: Sequence[group.Vector]) -> float:
    kaa, kbb, kab = kernel(a, a, True), kernel(b, b, True), kernel(a, b, False)
    if not (kaa > 0 and kbb > 0) or math.isnan(kab):
        return float("nan")
    return min(1.0, kab / math.sqrt(kaa * kbb))


def split(vectors: Sequence[group.Vector]) -> tuple[list[group.Vector], list[group.Vector]]:
    return list(vectors[0::2]), list(vectors[1::2])


def half_distance(cells: Mapping[str, Sequence[group.Vector]], x: str, y: str) -> float:
    xa, xb = split(cells[x])
    if x == y:
        return 1 - norm_sim(xa, xb)
    ya, yb = split(cells[y])
    return 1 - float(np.nanmean([norm_sim(xa, yb), norm_sim(xb, ya)]))


def rows_for(raw: list[dict], prompt: str, extract, repeats: set[int] | None):
    cells: dict[str, list[tuple[int, group.Vector]]] = defaultdict(list)
    for row in raw:
        if row["id"] != prompt or row.get("response") is None:
            continue
        if repeats is not None and int(row["repeat"]) not in repeats:
            continue
        cells[row["endpoint"]].append((int(row["repeat"]), extract(row["response"])))
    return {e: [v for _, v in sorted(items, key=lambda t: t[0])] for e, items in cells.items()}


def summary(cells: Mapping[str, Sequence[group.Vector]]) -> dict[str, float]:
    def mean(pairs):
        values = [half_distance(cells, x, y) for x, y in pairs if x in cells and y in cells]
        return float(np.nanmean(values)) if values else float("nan")

    out = {
        "noise": mean([(e, e) for e in cells]),
        "a:S-S": mean(list(itertools.combinations(S, 2))),
        "b:Q-S": mean([(QUANT, s) for s in S]),
        "b:Q-01": mean([(QUANT, "ref-01")]),
    }
    for name, endpoint in LORA.items():
        out[f"c:{name}-S"] = mean([(endpoint, s) for s in S])
    out["c:L-L"] = mean(list(itertools.combinations(LORA.values(), 2)))
    out["01-03"] = mean([("ref-01", "ref-03")])
    out["01-05"] = mean([("ref-01", "ref-05")])
    out["01-06"] = mean([("ref-01", "ref-06")])
    out["01-08"] = mean([("ref-01", "ref-08")])
    return out


def choice_rates(texts: Sequence[str], count: int = 30) -> np.ndarray:
    from melib import first_letter, parse_numbered

    hits = np.zeros(count)
    seen = np.zeros(count)
    for text in texts:
        for index, value in enumerate(parse_numbered(text, count)):
            letter = first_letter(value, "AB")
            if letter is not None:
                seen[index] += 1
                hits[index] += letter == "A"
    with np.errstate(invalid="ignore"):
        return np.where(seen > 0, (hits + 0.5) / (seen + 1.0), np.nan)


def choice_rank_table(raw: list[dict], prompt: str = "bat_choice") -> None:
    from melib import spearman

    texts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in raw:
        if row["id"] == prompt and row.get("response") is not None:
            texts[row["endpoint"]].append((int(row["repeat"]), row["response"]))
    ordered = {e: [t for _, t in sorted(v)] for e, v in texts.items()}
    endpoints = sorted(ordered)
    halves = {e: (choice_rates(v[0::2]), choice_rates(v[1::2])) for e, v in ordered.items()}
    full = {e: choice_rates(v) for e, v in ordered.items()}
    print(f"\n{prompt}: Spearman of per-item P(A); diagonal = split-half")
    print("      " + " ".join(e[-2:].rjust(5) for e in endpoints))
    for x in endpoints:
        cells = []
        for y in endpoints:
            rho = spearman(*halves[x]) if x == y else spearman(full[x], full[y])
            cells.append(f"{rho:5.2f}")
        print(f"{x[-2:]:>5s} " + " ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    parser.add_argument("--repeats", default="")
    parser.add_argument("--choice", action="store_true")
    args = parser.parse_args()
    if args.choice:
        choice_rank_table(load_rows(args.answers))
        return
    raw = load_rows(args.answers)
    repeats = None
    if args.repeats:
        low, high = (int(x) for x in args.repeats.split("-"))
        repeats = set(range(low, high + 1))
    prompts = sorted({row["id"] for row in raw})
    keys = None
    for prompt in prompts:
        for fname, extract in FEATURE_MAPS.items():
            cells = rows_for(raw, prompt, extract, repeats)
            row = summary(cells)
            if keys is None:
                keys = list(row)
                print("| prompt | feature | " + " | ".join(keys) + " |")
                print("|---" * (len(keys) + 2) + "|")
            print(f"| {prompt} | {fname} | " + " | ".join(f"{row[k]:.2f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
