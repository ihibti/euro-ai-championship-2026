"""Simulator: calibration and power of three tests for one 30-item binary battery.

Two endpoints answer the battery. The scenarios set how the second endpoint's item logits relate
to the first's: equal, a temperature change, a uniform shift (same model), or item-specific
shifts of growing size (different models). Each answer gives one observation per item, with 5%
of items unparsed.

Tests: "affine/bootstrap" (deviance of a shared item-logit fit with an affine map per endpoint,
parametric bootstrap), "affine/permutation" (the same deviance, answers permuted between the two
endpoints), "delta/permutation" (per-item agreement kernel, exact permutation), which the method
uses.

Usage: uv run python tasks/model_equality/analysis/second/simulate.py [--reps N]
Writes simulate.md next to this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import secondlib as s

HERE = Path(__file__).resolve().parent
ITEMS = 30
MISSING = 0.05

SCENARIOS: dict[str, tuple[str, float]] = {
    "same model: identical": ("same", 0.0),
    "same model: temperature x1.6": ("temperature", 1.6),
    "same model: uniform shift 0.7": ("shift", 0.7),
    "different: item shifts sd 0.3": ("items", 0.3),
    "different: item shifts sd 0.7": ("items", 0.7),
    "different: item shifts sd 1.5": ("items", 1.5),
    "different: item shifts sd 3": ("items", 3.0),
}


def second_logits(theta: np.ndarray, kind: str, size: float, rng: np.random.Generator):
    if kind == "same":
        return theta
    if kind == "temperature":
        return theta / size
    if kind == "shift":
        return theta + size
    return theta + size * rng.standard_normal(theta.shape)


def answers(logits: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    values = (rng.random((n, logits.size)) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    values[rng.random(values.shape) < MISSING] = np.nan
    return values


def p_values(x: np.ndarray, y: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    observed, null = s.bootstrap_deviance(s.counts(x), s.counts(y), draws, rng)
    p_boot = (1.0 + float(np.sum(null >= observed))) / (draws + 1.0)
    perm = s.permutation_deviances(x, y, 5000, rng)
    p_affine = float(np.mean(perm >= perm[0] - 1e-9))
    masks = s.split_masks(len(x), len(y), 5000, rng)
    delta = s.mmd_over_splits(s.battery_gram(np.vstack([x, y])), masks)
    p_delta = float(np.mean(delta >= delta[0] - 1e-9))
    return [p_boot, p_affine, p_delta]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--draws", type=int, default=99)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--base-sd", type=float, nargs="+", default=[1.5, 3.0])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    tests = ["affine/bootstrap", "affine/permutation", "delta/permutation"]
    columns = [(sd, n) for sd in args.base_sd for n in args.sizes]
    lines = [
        "# Simulator: share of p < 0.05, one 30-item binary battery",
        "",
        f"{args.reps} replicates per cell; {args.draws} bootstrap draws; permutations exact."
        " Base item logits are drawn from N(0, sd^2) anew in each replicate; 5% of items are"
        " unparsed. A same-model row should read about 0.05 or less. Columns: base sd and"
        " answers per endpoint. Standard error about 0.015 at 0.05 with 200 replicates.",
        "",
        "| scenario | test | " + " | ".join(f"sd {sd}, n={n}" for sd, n in columns) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    for label, (kind, size) in SCENARIOS.items():
        table = {t: [] for t in tests}
        for sd, n in columns:
            hits = np.zeros(len(tests))
            for _ in range(args.reps):
                theta = sd * rng.standard_normal(ITEMS)
                x = answers(theta, n, rng)
                y = answers(second_logits(theta, kind, size, rng), n, rng)
                hits += np.array(p_values(x, y, args.draws, rng)) < 0.05
            for t, h in zip(tests, hits / args.reps, strict=True):
                table[t].append(h)
        for t in tests:
            lines.append(f"| {label} | {t} | " + " | ".join(f"{v:.3f}" for v in table[t]) + " |")
            print(lines[-1], flush=True)
    (HERE / "simulate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote simulate.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
