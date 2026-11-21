"""Permutation test of "same answer distribution" between endpoint pairs, on group.py's distance.

uv run python tasks/model_equality/analysis/permtest.py \
    --answers tasks/model_equality/answers/round1.jsonl --prefix ref- --below 4

For each pair, the answers of the two endpoints to each prompt are pooled and split again at
random with the original sizes; the p-value is the share of splits at least as distant as the
observed one. The test is exact for endpoints without a disguise (a quantized or LoRA copy
against the plain model). A system prompt, persona or temperature changes the distribution too,
so a small p-value there says "different answers", not "different model".
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import group  # noqa: E402
from melib import load_rows  # noqa: E402

Cells = Mapping[str, Sequence[group.Vector]]


def pair_distance(first: Cells, second: Cells, kinds: Sequence[str], mode: str) -> float:
    _, sim = group.similarity_matrix({"x": first, "y": second}, kinds, mode)
    return float(1 - sim[0, 1])


def permutation_pvalue(
    first: Cells,
    second: Cells,
    kinds: Sequence[str],
    mode: str,
    draws: int,
    rng: random.Random,
) -> tuple[float, float]:
    observed = pair_distance(first, second, kinds, mode)
    prompts = sorted(set(first) & set(second))
    hits = 0
    for _ in range(draws):
        left: dict[str, list[group.Vector]] = {}
        right: dict[str, list[group.Vector]] = {}
        for prompt in prompts:
            pool = list(first[prompt]) + list(second[prompt])
            rng.shuffle(pool)
            size = len(first[prompt])
            left[prompt], right[prompt] = pool[:size], pool[size:]
        hits += pair_distance(left, right, kinds, mode) >= observed
    return observed, (hits + 1) / (draws + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--below", type=int, default=None)
    parser.add_argument("--kinds", default=",".join(group.MODEL_KINDS))
    parser.add_argument("--mode", choices=("pooled", "prompt"), default="pooled")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", default="", help="comma list of a:b pairs; default all")
    args = parser.parse_args(argv)
    rows = load_rows(args.answers)
    if args.below is not None:
        rows = [row for row in rows if int(row["repeat"]) < args.below]
    cells = group.collect(rows, args.prefix)
    kinds = tuple(filter(None, args.kinds.split(",")))
    if args.pairs:
        pairs = [tuple(item.split(":")) for item in args.pairs.split(",")]
    else:
        pairs = list(itertools.combinations(sorted(cells), 2))
    rng = random.Random(args.seed)
    print(f"mode {args.mode}, kinds {','.join(kinds)}, {args.draws} permutations per pair")
    print("pair distance p-value")
    for x, y in pairs:
        distance, p = permutation_pvalue(cells[x], cells[y], kinds, args.mode, args.draws, rng)
        print(f"{x} {y} {distance:.3f} {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
