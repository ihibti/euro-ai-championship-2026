"""Expected gain over the best score for single lines, pairs, and the last slot given the first.

Reads the drawn scores that `score_fit.py --draws-out` saves. Each draw is anchored on the best
observed line's score, so a line's anchored score is best * line / best line under that draw. Only
the best submission counts, so a plan is worth the mean of max(0, its best anchored score - best).

    cd <team folder> && python <repo>/tasks/unlearning/slot_gain.py draws.npz \
        --first m15_s1s2ep40x2 --lines k9_e35t2_lr1e4x1.95,k14_t1_lr9e5x1.45
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

BANDS = (0.0, 0.2812, 0.2866, 0.295, 1.0)


def anchored(values: np.ndarray, reference: int, best: float) -> np.ndarray:
    return best * values / values[:, [reference]]


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
    args = parser.parse_args(argv)
    with np.load(args.draws) as data:
        values = data["values"].astype(np.float64)
        names = [str(n) for n in data["names"]]
        reference, best = int(data["best"][0]), float(data["best"][1])
    scores = anchored(values, reference, best)
    column = {n: i for i, n in enumerate(names)}
    lines = args.lines.split(",")
    first = scores[:, column[args.first]]
    print(f"anchored on {names[reference]} at {best:.4f}; {len(scores)} draws")
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
