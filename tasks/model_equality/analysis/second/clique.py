"""Largest groups of endpoints with no rejected pair inside, from the mode statistic.

Frozen before the scored battery rows were read (PR #27, after the desk's 03:01 comment). Inside
each family, every pair of endpoints gets the mode statistic of `nightlib` on the chosen probes:
answers are permuted between the two endpoints, and the statistic only moves when a modal answer
changes, so a hotter copy of the same model is not rejected. A pair is rejected when its p-value
is below --alpha (0.01). The groups reported are the largest sets with no rejected pair inside
(maximum cliques of the graph of pairs not rejected), with every rejected pair listed. An
endpoint takes part only when it has at least --min answers on every chosen probe. No majority
is assumed: a family of distinct changed copies gives a small largest group.

Usage, from this directory:
  uv run python clique.py --answers FILE ... --prefix ep- --families recon_ep.csv
  uv run python clique.py --answers FILE ... --prefix ref- --probes p3_repeat p4_nextword
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Sequence
from pathlib import Path

import nightlib as n
import secondlib as s

from euroai.me_score import parse_labels

PROBES = ("bat_choice", "bat_trivia")


def max_cliques(nodes: Sequence[str], allowed: set[frozenset]) -> list[tuple[str, ...]]:
    ordered = sorted(nodes)
    for size in range(len(ordered), 0, -1):
        found = [
            group
            for group in itertools.combinations(ordered, size)
            if all(frozenset(pair) in allowed for pair in itertools.combinations(group, 2))
        ]
        if found:
            return found
    return []


def family_cliques(
    corpus: s.Corpus, members: Iterable[str], probes: Sequence[str], alpha: float, seed: int
) -> tuple[list[tuple[str, ...]], dict]:
    members = sorted(members)
    sub = s.Corpus(
        {q: corpus.prompts[q] for q in probes if q in corpus.prompts},
        {e: {q: corpus.answers[e].get(q, []) for q in probes} for e in members},
    )
    tables = n.build_tables(sub, {q: n.NIGHT_PROMPTS[q] for q in probes})
    tests = n.all_pairs(tables, members, seed)
    allowed = {frozenset(pair) for pair, test in tests.items() if test.p >= alpha}
    return max_cliques(members, allowed), tests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", nargs="+", required=True)
    parser.add_argument("--families", type=Path)
    parser.add_argument(
        "--extra", nargs="*", default=[], help="endpoint=family for endpoints not in the CSV"
    )
    parser.add_argument("--probes", nargs="+", default=list(PROBES))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    rows = [
        row
        for path in args.answers
        for row in s.read_jsonl(path)
        if row["endpoint"].startswith(tuple(args.prefix)) and row["id"] in args.probes
    ]
    corpus = s.build_corpus(rows)
    usable = [
        e
        for e in sorted(corpus.answers)
        if all(len(corpus.answers[e].get(q, [])) >= args.min for q in args.probes)
    ]
    family = {}
    if args.families:
        family = {e: f for e, (f, _) in parse_labels(args.families.read_text("utf-8")).items()}
    family.update(dict(item.split("=", 1) for item in args.extra))
    groups: dict[str, list[str]] = {}
    for e in usable:
        groups.setdefault(family.get(e, "all"), []).append(e)
    left_out = sorted(set(corpus.answers) - set(usable))
    print(
        f"probes {', '.join(args.probes)}; alpha {args.alpha}; left out (too few answers): "
        f"{' '.join(left_out) or 'none'}"
    )
    for name, members in sorted(groups.items()):
        cliques, tests = family_cliques(corpus, members, args.probes, args.alpha, args.seed)
        rejected = sorted((t.p, a, b, t.z) for (a, b), t in tests.items() if t.p < args.alpha)
        size = len(cliques[0]) if cliques else 0
        print(
            f"## family {name}: {len(members)} endpoints, largest group {size}, "
            f"{len(cliques)} such group(s), {len(rejected)} of {len(tests)} pairs rejected"
        )
        for group in cliques[:6]:
            outside = [e for e in members if e not in group]
            print(f"  group: {' '.join(group)} | outside: {' '.join(outside) or '-'}")
        for p, a, b, z in rejected:
            print(f"  rejected {a} {b}: p {p:.4f}, z {z:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
