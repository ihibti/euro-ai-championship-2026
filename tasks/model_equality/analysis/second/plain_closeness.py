"""Which unlabelled endpoint is closest to a family's labelled same-model copies?

For each probe, every candidate gets the mode statistic of `nightlib` against each labelled copy
(answers permuted between the two endpoints). The statistic moves only when modal answers differ,
so a temperature or top-p copy is a clean reference for the plain copy's modes; a prompt or
watermark copy shifts a few modes, but by the same amount against every candidate, so the order
of the candidates is kept. Reported per probe: the candidate's mean standardised statistic against
all labelled copies, and against the decoding copies alone (--decoding). Lowest is closest.

Usage, from this directory:
  uv run python plain_closeness.py \\
      --answers ../../answers/round1.jsonl ../../answers/night-e.jsonl \\
      --labelled ep-02 ep-23 ep-33 ep-14 ep-04 --decoding ep-33 ep-14 \\
      --candidates ep-01 ep-12 ep-17 ep-18 ep-24 ep-25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nightlib as n
import numpy as np
import secondlib as s

PROBES = (
    "bat_choice",
    "bat_trivia",
    "idn_identity",
    "txt_story",
    "e1_leak_repeat",
    "e2_leak_quote",
)


def closeness(corpus, labelled, decoding, candidates, probe, minimum, seed):
    endpoints = sorted(set(labelled) | set(candidates))
    if probe not in corpus.prompts:
        return None
    if min(len(corpus.answers[e].get(probe, [])) for e in endpoints) < minimum:
        return None
    sub = s.Corpus(
        {probe: corpus.prompts[probe]}, {e: {probe: corpus.answers[e][probe]} for e in endpoints}
    )
    tests = n.all_pairs(n.build_tables(sub, {probe: n.NIGHT_PROMPTS[probe]}), endpoints, seed)

    def z(a: str, b: str) -> float:
        return s.standardized(tests[tuple(sorted((a, b)))])

    return {
        c: (
            float(np.mean([z(c, r) for r in labelled])),
            float(np.mean([z(c, r) for r in decoding])),
        )
        for c in candidates
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--labelled", nargs="+", required=True)
    parser.add_argument("--decoding", nargs="+", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--probes", nargs="+", default=list(PROBES))
    parser.add_argument("--min", type=int, default=10)
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
    used, table = [], {}
    for probe in args.probes:
        found = closeness(
            corpus, args.labelled, args.decoding, args.candidates, probe, args.min, args.seed
        )
        if found is not None:
            used.append(probe)
            table[probe] = found
    print(f"labelled {' '.join(args.labelled)}; decoding {' '.join(args.decoding)}")
    print("Cells: mean standardised mode statistic against all labelled / against decoding copies.")
    print()
    print("| candidate | " + " | ".join(used) + " | mean, all | mean, decoding |")
    print("|---|" + "---|" * (len(used) + 2))
    ranked = sorted(args.candidates, key=lambda c: float(np.mean([table[p][c][0] for p in used])))
    for c in ranked:
        cells = [f"{table[p][c][0]:+.2f} / {table[p][c][1]:+.2f}" for p in used]
        all_mean = float(np.mean([table[p][c][0] for p in used]))
        dec_mean = float(np.mean([table[p][c][1] for p in used]))
        print(f"| {c} | " + " | ".join(cells) + f" | {all_mean:+.2f} | {dec_mean:+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
