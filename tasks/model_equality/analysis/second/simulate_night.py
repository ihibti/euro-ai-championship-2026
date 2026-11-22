"""Simulator for the night-probe statistic: calibration and power of "mode" against "mmd".

Two endpoints answer a 20-item battery with 4 options per item, 10 answers each (the night
probes' repeat count). Scenarios set the second endpoint's item logits: identical, flatter or
sharper (temperature), a uniform bias towards one option (same model), or a new mode on some
items, or small noise on all items (different models).

Usage: uv run python tasks/model_equality/analysis/second/simulate_night.py [--reps N]
Writes simulate_night.md next to this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nightlib as n
import numpy as np
import secondlib as s

HERE = Path(__file__).resolve().parent
ITEMS, OPTIONS = 20, 4


def scenario(theta: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "same model: identical":
        return theta
    if name == "same model: temperature x1.6 (flatter)":
        return theta / 1.6
    if name == "same model: temperature /1.6 (sharper)":
        return theta * 1.6
    if name == "same model: bias +1 on option 0":
        out = theta.copy()
        out[:, 0] += 1.0
        return out
    if name.startswith("different: new mode on"):
        k = int(name.split()[4])
        out = theta.copy()
        for item in rng.choice(ITEMS, k, replace=False):
            out[item] = rng.permutation(out[item])
        return out
    if name.startswith("different: noise sd"):
        return theta + float(name.split()[3]) * rng.standard_normal(theta.shape)
    raise ValueError(name)


SCENARIOS = [
    "same model: identical",
    "same model: temperature x1.6 (flatter)",
    "same model: temperature /1.6 (sharper)",
    "same model: bias +1 on option 0",
    "different: new mode on 2 items",
    "different: new mode on 5 items",
    "different: noise sd 0.5 on all items",
    "different: noise sd 1.0 on all items",
]


def draw(theta: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    p = np.exp(theta - theta.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    cumulative = p.cumsum(axis=1)
    u = rng.random((size, ITEMS, 1))
    return (u > cumulative[None, :, :]).sum(axis=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=150)
    parser.add_argument("--answers", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--base-sd", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    lines = [
        "# Night-probe simulator: share of p < 0.05",
        "",
        f"{ITEMS} items x {OPTIONS} options, base logits N(0, {args.base_sd}^2), {args.reps}"
        " replicates per cell, 1,000 sampled permutations per test. Same-model rows should read"
        " 0.05 or less.",
        "",
        "| scenario | " + " | ".join(f"mode, n={k} | mmd, n={k}" for k in args.answers) + " |",
        "|---|" + "---|---|" * len(args.answers),
    ]
    for name in SCENARIOS:
        cells = []
        for size in args.answers:
            hits = {"mode": 0, "mmd": 0}
            for _ in range(args.reps):
                theta = args.base_sd * rng.standard_normal((ITEMS, OPTIONS))
                x = draw(theta, size, rng)
                y = draw(scenario(theta, name, rng), size, rng)
                onehots = n.informative_views(np.vstack([x, y]))
                masks = s.split_masks(size, size, 1000, rng)
                for kind in hits:
                    dist = n.split_statistic(onehots, masks, kind) if onehots else np.zeros(1)
                    hits[kind] += float(np.mean(dist >= dist[0] - 1e-9)) < 0.05
            cells += [f"{hits['mode'] / args.reps:.3f}", f"{hits['mmd'] / args.reps:.3f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    (HERE / "simulate_night.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
