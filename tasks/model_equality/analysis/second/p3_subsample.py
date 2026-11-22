"""How often the p3 rule gives the right split on the references at n answers per endpoint.

Draws n of the pooled answers per endpoint without replacement, runs the rule (`p3rule.py`,
either feature) and scores it against the practice answers.

Usage, from this directory (pooled file from `pool.py`):
  uv run python p3_subsample.py --answers /tmp/p3_pooled_refs.jsonl --sizes 10 15 20 --draws 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import p3rule as r
import secondlib as s

from euroai.me_score import parse_labels, same_model_mcc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--probe", default="p3_repeat")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    corpus = s.build_corpus(s.read_jsonl(args.answers), prefix="ref-")
    answers = {e: per.get(args.probe, []) for e, per in corpus.answers.items()}
    truth = parse_labels(r.TRUTH.read_text(encoding="utf-8"), prefix="ref-")
    rng = np.random.default_rng(args.seed)
    print("| feature | answers per endpoint | mean MCC | share at 0.73 or more | share at 0 |")
    print("|---|---|---|---|---|")
    for name, feature in sorted(r.FEATURES.items()):
        flags = {e: np.array([feature(t) for t in rs]) for e, rs in answers.items()}
        for size in args.sizes:
            scores = []
            for _ in range(args.draws):
                picks = {
                    e: rng.choice(len(f), size=min(size, len(f)), replace=False)
                    for e, f in flags.items()
                }
                counts = {e: int(flags[e][p].sum()) for e, p in picks.items()}
                totals = {e: len(p) for e, p in picks.items()}
                call = r.call_family(sorted(answers), counts, totals, rng, args.bootstrap)
                labels = {e: ("F", "base") for e in call.base} | {e: ("F", e) for e in call.lora}
                scores.append(same_model_mcc(truth, labels).mcc)
            scores = np.array(scores)
            print(
                f"| {name} | {size} | {scores.mean():.3f} | {np.mean(scores >= 0.73):.2f} "
                f"| {np.mean(scores <= 0.0):.2f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
