"""A pairwise battery matrix inside each family, and the largest group with no rejected pair,
a reading that does not assume the plain model is the majority.

uv run python tasks/model_equality/analysis/pairwise.py \
    --answers tasks/model_equality/answers/round1.jsonl --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv

Frozen on 2026-11-22 at 03:03 Paris, before any fresh battery answer was read.
- Items and modes as in modal.py (bat_trivia and bat_choice, 30 items each).
- Pair statistic: the items on which the two endpoints' mode sets share no token
  (modal.flips), over the 60 items. Temperature keeps the order of the probabilities, so it
  keeps the modes in expectation.
- Null: the answers of the two endpoints are pooled per battery and split at random into the
  two sizes, DRAWS times (seed 0); one-sided p = (1 + draws with at least the observed
  flips) / (DRAWS + 1).
- A pair is rejected at p < alpha (0.01), with no correction for the number of pairs.
- Largest group: the largest set of endpoints with no rejected pair inside, found by exhaustive
  search. Ties go to the set whose smallest pairwise p is largest, then to the first in sorted
  order. Under "changed copies are singletons", that group is the plain model and whatever
  cannot be told from it.
- --join FAMILY=e1,e2 adds endpoints of another source (such as local- copies) to a family.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

Pair = tuple[str, str]


def pair_null(
    first: modal.Codes, second: modal.Codes, draws: int = modal.DRAWS, seed: int = modal.SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros(draws, dtype=np.int64)
    for battery in sorted(set(first) & set(second)):
        a, b = first[battery], second[battery]
        if not len(a) or not len(b):
            continue
        pool = np.concatenate([a, b])
        size = modal.size_of(pool)
        order = np.argsort(rng.random((draws, len(pool))), axis=1)
        left = modal.item_counts(pool[order[:, : len(a)]], size)
        right = modal.item_counts(pool[order[:, len(a) :]], size)
        total += modal.flips(left, right).sum(axis=-1)
    return total


def pair_test(
    first: modal.Codes, second: modal.Codes, draws: int = modal.DRAWS
) -> tuple[int, float]:
    seen = modal.observed_flips(first, {b: second[b] for b in first if b in second})
    null = pair_null(first, second, draws)
    return seen, float((1 + (null >= seen).sum()) / (draws + 1))


def matrix(
    values: Mapping[str, modal.Codes], members: Sequence[str], draws: int = modal.DRAWS
) -> dict[Pair, tuple[int, float]]:
    return {
        (a, b): pair_test(values[a], values[b], draws)
        for a, b in itertools.combinations(members, 2)
    }


def largest_group(
    members: Sequence[str], tests: Mapping[Pair, tuple[int, float]], alpha: float = stop.ALPHA
) -> list[str]:
    def p_of(a: str, b: str) -> float:
        return tests.get((a, b), tests.get((b, a), (0, 1.0)))[1]

    ordered = sorted(members)
    for size in range(len(ordered), 0, -1):
        best: tuple[float, list[str]] | None = None
        for group in itertools.combinations(ordered, size):
            pairs = [p_of(a, b) for a, b in itertools.combinations(group, 2)]
            if any(p < alpha for p in pairs):
                continue
            floor = min(pairs, default=1.0)
            if best is None or floor > best[0]:
                best = (floor, list(group))
        if best is not None:
            return best[1]
    return []


def parse_joins(items: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items:
        family, _, names = item.partition("=")
        out.setdefault(family, []).extend(n.strip() for n in names.split(",") if n.strip())
    return out


def family_members(
    path: Path | None, endpoints: Sequence[str], joins: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    families = modal.read_families(path, endpoints)
    for family, extra in joins.items():
        families.setdefault(family, [])
        families[family] = sorted(set(families[family]) | set(extra))
    return families


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--families", type=Path, default=None)
    parser.add_argument("--join", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=stop.ALPHA)
    parser.add_argument("--draws", type=int, default=modal.DRAWS)
    args = parser.parse_args(argv)
    joins = parse_joins(args.join)
    joined = {e for names in joins.values() for e in names}
    rows = [
        r
        for r in load_rows(args.answers)
        if str(r["endpoint"]).startswith(args.prefix) or r["endpoint"] in joined
    ]
    own = sorted({str(r["endpoint"]) for r in rows if r["id"] in modal.BATTERIES} - joined)
    for family, members in family_members(args.families, own, joins).items():
        values = modal.encode(modal.battery_tokens(rows, members))
        present = [m for m in members if m in values]
        tests = matrix(values, present, args.draws)
        print(f"\n== family {family}: pairwise p (flips), alpha {args.alpha}")
        print("| | " + " | ".join(present) + " |")
        print("|---" * (len(present) + 1) + "|")
        for a in present:
            cells = []
            for b in present:
                if a == b:
                    cells.append("-")
                    continue
                flips, p = tests.get((a, b), tests.get((b, a)))
                mark = "*" if p < args.alpha else ""
                cells.append(f"{p:.3f}{mark} ({flips})")
            print(f"| {a} | " + " | ".join(cells) + " |")
        group = largest_group(present, tests, args.alpha)
        rest = [m for m in present if m not in group]
        print(f"- largest group with no rejected pair ({len(group)}): {' '.join(group)}")
        print(f"- outside it: {' '.join(rest) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
