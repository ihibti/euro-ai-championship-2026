"""How many unlabelled endpoints are indistinguishable from the labelled copies?

Repeats `plain_closeness.py` on random subsets of the answers (--keep of each endpoint's answers
per probe, without replacement) and, in each draw, sorts the candidates by their mean statistic
against the labelled copies and cuts at the largest gap between neighbours. The count before the
cut is the number of candidates that sit with the labelled copies. Reported: the distribution of
that count over the draws, and how often each candidate falls before the cut.

Usage, from this directory:
  uv run python closeness_draws.py --answers ... --labelled ... --candidates ... --probes ...
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import plain_closeness as pc
import secondlib as s


def subsample(corpus: s.Corpus, keep: int, rng: np.random.Generator) -> s.Corpus:
    answers = {
        e: {
            q: [r[i] for i in sorted(rng.choice(len(r), min(keep, len(r)), replace=False))]
            for q, r in per.items()
        }
        for e, per in corpus.answers.items()
    }
    return s.Corpus(corpus.prompts, answers)


def gap_count(scores: dict) -> tuple[int, list[str]]:
    ordered = sorted(scores, key=scores.get)
    values = np.array([scores[c] for c in ordered])
    cut = int(np.argmax(np.diff(values))) + 1
    return cut, ordered[:cut]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--labelled", nargs="+", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--min", type=int, default=10)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    wanted = set(args.labelled) | set(args.candidates)
    rows = [
        r
        for path in args.answers
        for r in s.read_jsonl(path)
        if r["endpoint"] in wanted and r["id"] in args.probes
    ]
    corpus = s.build_corpus(rows)
    probes = [
        q
        for q in args.probes
        if q in corpus.prompts
        and min(len(corpus.answers[e].get(q, [])) for e in wanted) >= args.min
    ]
    rng = np.random.default_rng(args.seed)
    counts: Counter = Counter()
    inside: Counter = Counter()
    full = {
        c: float(
            np.mean(
                [
                    pc.closeness(corpus, args.labelled, args.labelled, args.candidates, q, 1, 7)[c][
                        0
                    ]
                    for q in probes
                ]
            )
        )
        for c in args.candidates
    }
    for draw in range(args.draws):
        sub = subsample(corpus, args.keep, rng)
        per = [
            pc.closeness(sub, args.labelled, args.labelled, args.candidates, q, 1, draw)
            for q in probes
        ]
        scores = {c: float(np.mean([p[c][0] for p in per])) for c in args.candidates}
        cut, members = gap_count(scores)
        counts[cut] += 1
        inside.update(members)
    cut, members = gap_count(full)
    print(f"probes {', '.join(probes)}; {args.draws} draws keeping {args.keep} answers")
    print(
        f"all answers: count {cut} ({' '.join(members)}); scores "
        + ", ".join(f"{c} {full[c]:+.2f}" for c in sorted(full, key=full.get))
    )
    print("count over draws: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(
        "share of draws before the cut: "
        + ", ".join(
            f"{c} {inside[c] / args.draws:.2f}"
            for c in sorted(args.candidates, key=lambda c: -inside[c])
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
