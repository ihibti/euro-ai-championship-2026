"""bat_choice deviations of each endpoint from its family's core, against a reference endpoint's.

For every endpoint: the mean shift of its per-item A share from the core's (the core without the
endpoint itself), and the correlation of its per-item deviations with those of --reference
(deviation from the reference family's core), with an item-permutation p-value. A flatter copy
of a model whose core leans to one letter shifts towards 0.5 on the near-tie items; so does a
watermark.

Usage, from this directory:
  uv run python choice_shift.py --answers ../../answers/round1.jsonl --reference ep-04 \\
      --core A=ep-01,ep-12,ep-14,ep-17,ep-18,ep-24,ep-25,ep-33 B=ep-03,ep-07,ep-21,ep-26,ep-30,ep-32
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import secondlib as s

HERE = Path(__file__).resolve().parent


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--families", type=Path, default=HERE / "recon_ep.csv")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--core", nargs="+", required=True, help="FAMILY=ep-01,ep-02,...")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    with args.families.open(encoding="utf-8") as handle:
        family = {row["endpoint"]: row["family"] for row in csv.DictReader(handle)}
    core = {item.split("=")[0]: item.split("=")[1].split(",") for item in args.core}
    rows = [r for r in s.read_jsonl(args.answers) if r["endpoint"].startswith("ep-")]
    corpus = s.build_corpus(rows, prefix="ep-")
    batteries = s.item_matrices(corpus)
    share = {e: np.nanmean(batteries[e]["bat_choice"], axis=0) for e in corpus.answers}

    def deviation(endpoint: str) -> np.ndarray:
        members = [e for e in core[family[endpoint]] if e != endpoint]
        return share[endpoint] - np.mean([share[e] for e in members], axis=0)

    reference = deviation(args.reference)
    rng = np.random.default_rng(args.seed)
    print(f"| family | endpoint | mean shift | r with {args.reference} | item-permutation p |")
    print("|---|---|---|---|---|")
    for endpoint in sorted(corpus.answers, key=lambda e: (family[e], e)):
        if family[endpoint] not in core or endpoint == args.reference:
            continue
        mine = deviation(endpoint)
        r = correlation(mine, reference)
        null = np.array([correlation(mine, rng.permutation(reference)) for _ in range(args.draws)])
        print(
            f"| {family[endpoint]} | {endpoint} | {np.nanmean(mine):+.3f} | {r:+.2f} "
            f"| {float(np.mean(null >= r)):.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
