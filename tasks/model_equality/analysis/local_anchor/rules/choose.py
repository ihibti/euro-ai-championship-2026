"""Choose a model-equality grouping by expected score over stated probabilities.

Each family has labelled endpoints with a probability of being the plain model in disguise,
and a pool of base-like endpoints that holds the plain copy. Every changed endpoint is its own model.
uv run python choose.py spec.json
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys


def pair_counts(truth_same: dict[str, bool], group: frozenset[str]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth_same), 2):
        same_truth = truth_same[a] and truth_same[b]
        same_pred = a in group and b in group
        if same_truth and same_pred:
            tp += 1
        elif same_pred:
            fp += 1
        elif same_truth:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def sample_truth(family: dict, rng: random.Random) -> dict[str, bool]:
    min_changed = family.get("min_changed", 0)
    pool = family["pool"]
    names = list(pool)
    weights = [pool[name] for name in names]
    while True:
        truth = {name: rng.random() < p for name, p in family["labelled"].items()}
        plain = rng.choices(names, weights=weights)[0]
        for name in names:
            truth[name] = name == plain or rng.random() < family.get("hidden_same", 0.0)
        if sum(1 for same in truth.values() if not same) >= min_changed:
            return truth


def expected_scores(spec: dict, groups: dict[str, list[frozenset[str]]], draws: int, seed: int) -> dict:
    rng = random.Random(seed)
    totals: dict[tuple, float] = {}
    families = list(spec)
    per_family_counts = {name: [] for name in families}
    for _ in range(draws):
        for name in families:
            truth = sample_truth(spec[name], rng)
            per_family_counts[name].append([pair_counts(truth, group) for group in groups[name]])
    for combo in itertools.product(*(range(len(groups[name])) for name in families)):
        total = 0.0
        for draw in range(draws):
            tp = fp = fn = tn = 0
            for name, index in zip(families, combo):
                a, b, c, d = per_family_counts[name][draw][index]
                tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
            total += mcc(tp, fp, fn, tn)
        totals[combo] = total / draws
    return totals


def candidate_groups(family: dict) -> list[frozenset[str]]:
    sure = [name for name, p in family["labelled"].items() if p >= 0.85]
    maybe = [name for name, p in family["labelled"].items() if p < 0.85]
    maybe += family.get("pool_candidates", [])
    out = []
    for size in range(len(maybe) + 1):
        for extra in itertools.combinations(maybe, size):
            out.append(frozenset(sure) | frozenset(extra))
    return out


def main(argv: list[str]) -> int:
    spec = json.load(open(argv[1]))
    draws = int(argv[2]) if len(argv) > 2 else 1500
    groups = {name: candidate_groups(family) for name, family in spec.items()}
    totals = expected_scores(spec, groups, draws, seed=0)
    families = list(spec)
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    for combo, value in ranked[:8]:
        parts = []
        for name, index in zip(families, combo):
            group = groups[name][index]
            sure = {n for n, p in spec[name]["labelled"].items() if p >= 0.85}
            parts.append(f"{name}: sure + {sorted(group - sure)}")
        print(f"model MCC {value:.3f} | score {0.1 + 0.9 * value:.3f} | " + " ; ".join(parts))
    base = tuple(0 for _ in families)
    print(f"sure labels only: model MCC {totals[base]:.3f} | score {0.1 + 0.9 * totals[base]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
