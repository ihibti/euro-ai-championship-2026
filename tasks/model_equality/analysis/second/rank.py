"""Rank the endpoints of each family by how changed they look, from the second method's features.

Frozen features, each turned into a per-endpoint score inside the family (endpoints listed with
--exclude, the prompted ones, are left out of the family before anything is computed):
- stop: the share of `capped` answers on the stop probes (pooled over --stop ids); the references
  give its direction, since their LoRA copies run long more. Used when every endpoint has at least
  --min-stop answers.
- one score per mode probe (--modes ids): the endpoint's mean standardised mode statistic against
  the other endpoints of its family (nightlib), so an endpoint whose modes differ from the rest
  scores high. Used when every endpoint has at least --min-mode answers.
Each score is standardised across the family; the combined score is their mean. Highest first.

Usage, from this directory:
  uv run python rank.py --answers ../../answers/night-b.jsonl ../../answers/night-d.jsonl \\
      ../../answers/round1.jsonl ../../answers/night-e.jsonl --prefix ref-
  uv run python rank.py --answers ... --prefix ep- --families recon_ep.csv \\
      --exclude ep-02 ep-23 ep-08 ep-20 ep-16 ep-29
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nightlib as n
import numpy as np
import p3rule as r
import secondlib as s

from euroai.me_score import parse_labels

STOP = ("p3_repeat", "d1_repeat150")
MODES = ("bat_choice", "bat_trivia", "idn_identity", "e1_leak_repeat", "e2_leak_quote")


def stop_scores(corpus: s.Corpus, members: list[str], probes, minimum: int) -> dict | None:
    answers = {e: [t for q in probes for t in corpus.answers[e].get(q, [])] for e in members}
    if min(len(v) for v in answers.values()) < minimum:
        return None
    return {e: float(np.mean([r.is_capped(t) for t in v])) for e, v in answers.items()}


def mode_scores(corpus: s.Corpus, members: list[str], probe: str, minimum: int, seed: int):
    if probe not in corpus.prompts:
        return None
    counts = [len(corpus.answers[e].get(probe, [])) for e in members]
    if min(counts) < minimum:
        return None
    sub = s.Corpus(
        {probe: corpus.prompts[probe]},
        {e: {probe: corpus.answers[e][probe]} for e in members},
    )
    tests = n.all_pairs(n.build_tables(sub, {probe: n.NIGHT_PROMPTS[probe]}), members, seed)
    return {
        e: float(np.mean([s.standardized(t) for (a, b), t in tests.items() if e in (a, b)]))
        for e in members
    }


def standardise(values: dict) -> dict | None:
    array = np.array(list(values.values()))
    spread = float(array.std())
    if spread == 0.0:
        return None
    return {e: (v - float(array.mean())) / spread for e, v in values.items()}


def rank_family(corpus, members, args) -> tuple[list[tuple[str, float]], dict]:
    features: dict[str, dict] = {}
    stop = stop_scores(corpus, members, args.stop, args.min_stop)
    if stop is not None:
        features["stop"] = stop
    for probe in args.modes:
        found = mode_scores(corpus, members, probe, args.min_mode, args.seed)
        if found is not None:
            features[probe] = found
    scaled = {k: v for k, v in ((k, standardise(v)) for k, v in features.items()) if v}
    combined = {
        e: float(np.mean([v[e] for v in scaled.values()])) if scaled else 0.0 for e in members
    }
    return sorted(combined.items(), key=lambda kv: -kv[1]), features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--families", type=Path)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--stop", nargs="*", default=list(STOP))
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--min-stop", type=int, default=10)
    parser.add_argument("--min-mode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    rows = [row for path in args.answers for row in s.read_jsonl(path)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    endpoints = [e for e in sorted(corpus.answers) if e not in args.exclude]
    if args.families:
        labels = parse_labels(args.families.read_text(encoding="utf-8"))
        groups: dict[str, list[str]] = {}
        for e in endpoints:
            groups.setdefault(labels[e][0], []).append(e)
    else:
        groups = {"all": endpoints}
    for family, members in groups.items():
        ranking, features = rank_family(corpus, members, args)
        names = sorted(features)
        print(f"## family {family}: features used {', '.join(names) or 'none'}")
        print("| rank | endpoint | combined | " + " | ".join(names) + " |")
        print("|---|---|---|" + "---|" * len(names))
        for place, (e, score) in enumerate(ranking, start=1):
            cells = " | ".join(f"{features[k][e]:.2f}" for k in names)
            print(f"| {place} | {e} | {score:.2f} | {cells} |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
