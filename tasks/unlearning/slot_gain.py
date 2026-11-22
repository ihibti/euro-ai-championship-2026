"""Expected gain over the best score for single lines, pairs, and the last slot given the first.

Reads the drawn scores that `score_fit.py --draws-out` saves. Each draw is anchored on the best
observed line's score, so a line's anchored score is best * line / best line under that draw. Only
the best submission counts, so a plan is worth the mean of max(0, its best anchored score - best).

With `--shared ledger_inputs.json`, each line is anchored instead on the scored lines that share runs
with it: its drawn value times the geometric mean of their observed-to-drawn ratios, weighted by the
weight they share. A line that shares no run with a scored line keeps the best line's anchor.

    cd <team folder> && python <repo>/tasks/unlearning/slot_gain.py draws.npz \
        --first m15_s1s2ep40x2 --lines k9_e35t2_lr1e4x1.95,k14_t1_lr9e5x1.45 \
        [--shared <repo>/tasks/unlearning/ledger_inputs.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import score_fit as sf  # noqa: E402

BANDS = (0.0, 0.2812, 0.2866, 0.295, 1.0)
FACTOR_WINDOW = (1.8, 2.1)


def anchored(values: np.ndarray, reference: int, best: float) -> np.ndarray:
    return best * values / values[:, [reference]]


def overlap(a: dict[str, float], b: dict[str, float]) -> float:
    """Weight two blends share: the sum over runs of the smaller of their normalised weights."""
    return sum(min(w, b.get(run, 0.0)) for run, w in a.items())


def weights_of(base: str) -> dict[str, float]:
    parts = sf.members(base)
    total = sum(w for _, w in parts)
    return {run: w / total for run, w in parts}


def shared_anchor(
    model: np.ndarray, anchors: np.ndarray, scores: np.ndarray, shares: np.ndarray
) -> np.ndarray:
    """Per draw, `model` times the share-weighted geometric mean of observed / drawn over anchors."""
    ratios = np.log(scores[None, :] / anchors)
    return model * np.exp(ratios @ (shares / shares.sum()))


def shared_scores(
    values: np.ndarray, names: list[str], inputs: dict, reference: int, best: float
) -> np.ndarray:
    """Every column re-anchored on the scored lines it shares runs with, inside FACTOR_WINDOW."""
    known = set(sf.BLENDS) | set(inputs["candidates"])
    column = {n: i for i, n in enumerate(names)}
    anchors = []
    for name, score in inputs["observed"].items():
        parsed = sf.parse_line(name, known)
        if parsed and name in column and FACTOR_WINDOW[0] <= parsed[1] <= FACTOR_WINDOW[1]:
            anchors.append((column[name], weights_of(parsed[0]), score))
    out = anchored(values, reference, best)
    for k, name in enumerate(names):
        parsed = sf.parse_line(name, known)
        if parsed is None or name in inputs["observed"]:
            continue
        mine = weights_of(parsed[0])
        shares = np.array([overlap(mine, w) for _, w, _ in anchors])
        if shares.sum() > 0:
            cols = [c for c, _, _ in anchors]
            scored = np.array([s for _, _, s in anchors])
            out[:, k] = shared_anchor(values[:, k], values[:, cols], scored, shares)
    return out


def joint_gain(scores: np.ndarray, floor: np.ndarray | float) -> float:
    """Mean gain of the best of several lines (columns of `scores`) over `floor`, per draw."""
    return float((np.maximum(scores.max(1), floor) - floor).mean())


def conditional_gains(
    scores: np.ndarray, first: np.ndarray, best: float, bands: tuple[float, ...] = BANDS
) -> list[tuple[float, float, float, np.ndarray, np.ndarray]]:
    """Per band of the first line's score: share of draws, then each column's P and gain over it."""
    rows = []
    for low, high in itertools.pairwise(bands):
        mask = (first >= low) & (first < high)
        if not mask.any():
            continue
        floor = np.maximum(first[mask], best)
        sub = scores[mask]
        beats = (sub > floor[:, None]).mean(0)
        gains = np.maximum(sub - floor[:, None], 0.0).mean(0)
        rows.append((low, high, float(mask.mean()), beats, gains))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("draws", type=Path)
    parser.add_argument("--first", required=True)
    parser.add_argument("--lines", required=True, help="comma-separated line names")
    parser.add_argument("--shared", type=Path, default=None, help="ledger_inputs.json")
    args = parser.parse_args(argv)
    with np.load(args.draws) as data:
        values = data["values"].astype(np.float64)
        names = [str(n) for n in data["names"]]
        reference, best = int(data["best"][0]), float(data["best"][1])
    if args.shared is None:
        scores = anchored(values, reference, best)
    else:
        inputs = json.loads(args.shared.read_text())
        scores = shared_scores(values, names, inputs, reference, best)
    column = {n: i for i, n in enumerate(names)}
    lines = args.lines.split(",")
    first = scores[:, column[args.first]]
    how = "the scored lines sharing runs, else " if args.shared else ""
    print(f"anchored on {how}{names[reference]} at {best:.4f}; {len(scores)} draws")
    for name in [args.first, *lines]:
        x = scores[:, column[name]]
        gain = joint_gain(x[:, None], best)
        print(f"  {name:30s} mean {x.mean():.4f} P {np.mean(x > best):.2f} gain {gain:.4f}")
    print(f"with {args.first}:")
    for name in lines:
        pair = scores[:, [column[args.first], column[name]]]
        print(f"  + {name:28s} joint gain {joint_gain(pair, best):.4f}")
    others = scores[:, [column[n] for n in lines]]
    for low, high, share, beats, gains in conditional_gains(others, first, best):
        print(f"{args.first} in [{low:.4f}, {high:.4f}): {share:.2f} of the draws")
        for name, p, g in zip(lines, beats, gains, strict=True):
            print(f"  {name:30s} P {p:.2f} gain {g:.4f}")


if __name__ == "__main__":
    main()
