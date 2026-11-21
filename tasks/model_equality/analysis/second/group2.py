"""Group endpoints with the second method and write endpoint,family,model.

Usage, from the repository root:
  uv run python tasks/model_equality/analysis/second/group2.py --prefix ref- --out FILE.csv
  uv run python tasks/model_equality/analysis/second/group2.py --prefix ep- \\
      --answers tasks/model_equality/answers/round1-ep.jsonl --out FILE.csv

Each pair of endpoints gets an exact permutation test per prompt (kernel two-sample statistic,
answers permuted between the two endpoints), standardised and summed over prompts. Groups are
joined, closest first, only while every pair between them has p above alpha divided by the
number of such pairs. On the references it also prints the MCC, and with --subsample N how far
the MCC moves when each endpoint keeps a random subset of its answers per prompt.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import secondlib as s

from euroai.me_score import parse_labels, same_model_mcc

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
TRUTH = TASK / "upstream" / "practice_answers.csv"
PROMPT_SETS = {
    "all": dict(s.MODEL_PROMPTS),
    "batteries": {k: v for k, v in s.MODEL_PROMPTS.items() if k.startswith("bat_")},
    "sequences": {k: v for k, v in s.MODEL_PROMPTS.items() if k.startswith("seq_")},
}


def group(
    corpus: s.Corpus, prompts: dict[str, str], alpha: float, seed: int, linkage: str = "average"
) -> tuple[list[list[str]], dict]:
    grams = s.corpus_grams(corpus, prompts)
    endpoints = sorted(corpus.answers)
    tests = s.all_pairs(grams, endpoints, seed)
    if linkage == "complete":
        groups = s.complete_linkage(endpoints, {k: t.p for k, t in tests.items()}, alpha)
    else:
        groups = s.average_linkage(endpoints, tests)
    return groups, tests


def subsample(corpus: s.Corpus, size: int, rng: np.random.Generator) -> s.Corpus:
    answers = {
        endpoint: {
            qid: [
                reps[i] for i in sorted(rng.choice(len(reps), min(size, len(reps)), replace=False))
            ]
            for qid, reps in per.items()
        }
        for endpoint, per in corpus.answers.items()
    }
    return s.Corpus(corpus.prompts, answers)


def as_labels(groups: list[list[str]]) -> dict[str, tuple[str, str]]:
    return {e: ("F1", f"M{i}") for i, g in enumerate(groups, start=1) for e in g}


def undecided(
    labels: dict[str, tuple[str, str]], tests: dict, threshold: float = 1.645
) -> list[tuple[str, str, float, bool]]:
    low, high = threshold / np.sqrt(2.0), threshold * np.sqrt(2.0)
    rows = []
    for (a, b), test in sorted(tests.items()):
        if labels[a][0] != labels[b][0]:
            continue
        z = s.standardized(test)
        if low <= z <= high:
            rows.append((a, b, z, labels[a] == labels[b]))
    return rows


def report(labels: dict[str, tuple[str, str]], tests: dict, prompts: dict[str, str]) -> str:
    rows = undecided(labels, tests)
    per_endpoint: dict[str, int] = {}
    for a, b, _, _ in rows:
        per_endpoint[a] = per_endpoint.get(a, 0) + 1
        per_endpoint[b] = per_endpoint.get(b, 0) + 1
    lines = [
        "# Undecided pairs: where more answers settle the most",
        "",
        "A pair inside one family is undecided when its standardised z lies between"
        " 1.645/sqrt(2) and 1.645*sqrt(2): doubling the answers moves a real difference of that"
        " size past the threshold, and leaves a pure-noise pair below it most of the time.",
        "",
        f"Undecided pairs: {len(rows)} of"
        f" {sum(labels[a][0] == labels[b][0] for a, b in tests)} pairs inside families.",
        "",
        "| endpoint | undecided pairs |",
        "|---|---|",
    ]
    for endpoint, count in sorted(per_endpoint.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {endpoint} | {count} |")
    lines += ["", "| a | b | z standardised | grouped now |", "|---|---|---|---|"]
    for a, b, z, same in rows:
        lines.append(f"| {a} | {b} | {z:.2f} | {'same' if same else 'different'} |")
    decided = [t for (a, b), t in tests.items() if s.standardized(t) > 1.645 * np.sqrt(2.0)]
    lines += ["", "Mean prompt z over the clearly different pairs (which prompts separate):", ""]
    for qid in sorted(prompts):
        values = [t.per_battery[qid] for t in decided if qid in t.per_battery]
        if values:
            lines.append(f"- `{qid}`: {np.mean(values):.2f} over {len(values)} pairs")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", default=[TASK / "answers/round1.jsonl"])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", choices=sorted(PROMPT_SETS), default="batteries")
    parser.add_argument("--family-threshold", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--linkage", choices=["average", "complete"], default="average")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--subsample-size", type=int, default=3)
    args = parser.parse_args(argv)

    rows = [row for path in args.answers for row in s.read_jsonl(path)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    prompts = PROMPT_SETS[args.prompts]
    groups, tests = group(corpus, prompts, args.alpha, args.seed, args.linkage)
    family_tests = s.all_pairs(
        s.corpus_grams(corpus, PROMPT_SETS["all"]), sorted(corpus.answers), args.seed + 7
    )
    families = s.family_groups(groups, family_tests, args.family_threshold)
    labels = s.nested_labels(families, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(s.labels_to_csv(labels), encoding="utf-8")
    with args.out.with_name(args.out.stem + "_pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "z", "p", *sorted(prompts)])
        for (a, b), test in sorted(tests.items()):
            writer.writerow(
                [a, b, f"{test.z:.3f}", f"{test.p:.5f}"]
                + [
                    f"{test.per_battery[q]:.3f}" if q in test.per_battery else ""
                    for q in sorted(prompts)
                ]
            )
    counts = {
        e: {q: len(r) for q, r in per.items() if q in prompts} for e, per in corpus.answers.items()
    }
    print(f"answers per endpoint and prompt: min {min(min(c.values()) for c in counts.values())}")
    for g in groups:
        print("group:", " ".join(g))
    for f in families:
        print("family:", " ".join(f))
    text = report(labels, tests, prompts)
    args.out.with_name(args.out.stem + "_report.md").write_text(text, encoding="utf-8")
    with args.out.with_name(args.out.stem + "_family_pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "z_standardized", "p"])
        for (a, b), test in sorted(family_tests.items()):
            writer.writerow([a, b, f"{s.standardized(test):.3f}", f"{test.p:.5f}"])

    if args.prefix == "ref-":
        truth = parse_labels(TRUTH.read_text(encoding="utf-8"), prefix="ref-")
        result = same_model_mcc(truth, as_labels(groups))
        print(f"mcc={result.mcc:.4f} tp={result.tp} fp={result.fp} fn={result.fn} tn={result.tn}")
        if args.subsample:
            rng = np.random.default_rng(args.seed + 1)
            scores = []
            for draw in range(args.subsample):
                sub = subsample(corpus, args.subsample_size, rng)
                sub_groups, _ = group(sub, prompts, args.alpha, args.seed + 2 + draw, args.linkage)
                scores.append(same_model_mcc(truth, as_labels(sub_groups)).mcc)
            ordered = sorted(scores)
            print(
                f"subsample {args.subsample_size} answers per prompt, {args.subsample} draws: "
                f"mean {statistics.mean(scores):.3f}, min {ordered[0]:.3f}, "
                f"median {statistics.median(scores):.3f}, max {ordered[-1]:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
