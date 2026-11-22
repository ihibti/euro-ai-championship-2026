"""Do some endpoints of a family share one deviation pattern on the battery items?

No side is taken as the core. For two endpoints a and b of the family, the other endpoints are
split at random into two disjoint halves; a's per-item deviation is taken from the mean of one
half and b's from the mean of the other, so the two deviations share no sampling noise, and
their correlation over the items is averaged over --splits random halvings. Copies of one
adapter at several strengths deviate on the same items in the same direction and correlate;
independent copies do not. For a candidate group, the statistic is the mean correlation over
its pairs; the p-value comes from random groups of the same size drawn from the same endpoints.

Usage, from this directory:
  uv run python shared_shift.py --answers ../../answers/round1.jsonl --family B \\
      --exclude ep-08 ep-20 --group ep-03 ep-06 ep-28
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import secondlib as s

HERE = Path(__file__).resolve().parent


def item_vectors(corpus: s.Corpus, endpoints: list[str], batteries: tuple[str, ...]) -> dict:
    matrices = s.item_matrices(corpus)
    return {
        e: np.concatenate([np.nanmean(matrices[e][b], axis=0) for b in batteries])
        for e in endpoints
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or x[ok].std() == 0 or y[ok].std() == 0:
        return 0.0
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def pair_matrix(vectors: dict, splits: int, rng: np.random.Generator) -> dict:
    names = sorted(vectors)
    matrix = {}
    for a, b in itertools.combinations(names, 2):
        others = [e for e in names if e not in (a, b)]
        values = []
        for _ in range(splits):
            order = rng.permutation(len(others))
            half = len(others) // 2
            first = [others[i] for i in order[:half]]
            second = [others[i] for i in order[half:]]
            da = vectors[a] - np.nanmean([vectors[e] for e in first], axis=0)
            db = vectors[b] - np.nanmean([vectors[e] for e in second], axis=0)
            values.append(correlation(da, db))
        matrix[(a, b)] = matrix[(b, a)] = float(np.mean(values))
    return matrix


def group_statistic(matrix: dict, group: list[str]) -> float:
    return float(np.mean([matrix[(a, b)] for a, b in itertools.combinations(group, 2)]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--families", type=Path, default=HERE / "recon_ep.csv")
    parser.add_argument("--family", required=True)
    parser.add_argument("--prefix", default="ep-")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--group", nargs="+", action="append", required=True)
    parser.add_argument("--batteries", nargs="+", default=["bat_choice", "bat_trivia"])
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    with args.families.open(encoding="utf-8") as handle:
        family = {row["endpoint"]: row["family"] for row in csv.DictReader(handle)}
    rows = [r for r in s.read_jsonl(args.answers) if r["endpoint"].startswith(args.prefix)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    members = sorted(
        e for e in corpus.answers if family[e] == args.family and e not in args.exclude
    )
    rng = np.random.default_rng(args.seed)
    vectors = item_vectors(corpus, members, tuple(args.batteries))
    matrix = pair_matrix(vectors, args.splits, rng)
    print(f"family {args.family}: {' '.join(members)}; batteries {', '.join(args.batteries)}")
    print("| a | " + " | ".join(e[-2:] for e in members) + " |")
    print("|---|" + "---|" * len(members))
    for a in members:
        cells = ["-" if a == b else f"{matrix[(a, b)]:+.2f}" for b in members]
        print(f"| {a[-2:]} | " + " | ".join(cells) + " |")
    for group in args.group:
        observed = group_statistic(matrix, group)
        null = [
            group_statistic(matrix, list(g)) for g in itertools.combinations(members, len(group))
        ]
        p = float(np.mean([v >= observed - 1e-12 for v in null]))
        print(
            f"group {' '.join(group)}: mean r {observed:+.3f}; share of the {len(null)} groups of"
            f" size {len(group)} at or above it {p:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
