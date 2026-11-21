"""Step 2 of the brief: each statistic against split-half noise on the reference endpoints.

Usage: uv run python tasks/model_equality/analysis/second/noise.py [--answers FILE]
Writes noise_<stem>.md next to this file.

The zero point is the test between two halves of one endpoint's answers. Pairs of endpoints are
tested at full size, and also at the size of a half (each half of one endpoint against each half
of the other), so that the pair z and the zero-point z are read at the same sample size.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path

import nightlib as n
import numpy as np
import secondlib as s
from group2 import PROMPT_SETS

HERE = Path(__file__).resolve().parent
ANSWERS = HERE.parents[1] / "answers"

KIND = {
    "ref-01": "plain",
    "ref-02": "fp8",
    "ref-03": "system prompt",
    "ref-04": "LoRA 0.7",
    "ref-05": "persona",
    "ref-06": "watermark",
    "ref-07": "LoRA 1.5",
    "ref-08": "temperature 1.0",
    "ref-09": "LoRA 0.3",
}
SAME = {"ref-01", "ref-03", "ref-05", "ref-06", "ref-08"}


def pair_type(a: str, b: str) -> str:
    def side(e: str) -> str:
        return "S" if e in SAME else "fp8" if e == "ref-02" else "LoRA"

    return {
        ("S", "S"): "1 same model, disguised",
        ("S", "fp8"): "2 fp8 copy against S",
        ("LoRA", "S"): "3 LoRA copy against S",
        ("LoRA", "LoRA"): "4 LoRA against LoRA",
        ("LoRA", "fp8"): "5 LoRA against fp8",
    }[tuple(sorted((side(a), side(b))))]


def halves(count: int) -> list[tuple[list[int], list[int]]]:
    size = count // 2
    if size < 2:
        return []
    splits = []
    for first in itertools.combinations(range(2 * size), size):
        if 0 in first:
            splits.append((list(first), [r for r in range(2 * size) if r not in first]))
    return splits


def fmt(values: list[float], digits: int = 2) -> str:
    return f"{statistics.median(values):.{digits}f}" if values else "n/a"


def share(values: list[float]) -> str:
    return f"{sum(v < 0.05 for v in values) / len(values):.2f}" if values else "n/a"


def statistics_for(corpus: s.Corpus, probes: str):
    if probes == "night":
        corpus = n.pool_ids(corpus)
        tables = n.build_tables(corpus)
        stats = [(f"night {q}", {q: tables[q]}) for q in sorted(tables)]
        stats.append(("night, all probes", tables))
        return corpus, stats, "night"
    stats = [
        (name, s.corpus_grams(corpus, PROMPT_SETS[name]))
        for name in ("batteries", "sequences", "all")
    ]
    return corpus, stats, "round1"


def run_test(kind: str, data, a_sel, b_sel, rng):
    if kind == "night":
        return n.view_test(data, a_sel, b_sel, rng, "mode")
    return s.kernel_test(data, a_sel, b_sel, rng)


def select(kind: str, data, endpoint: str, rows=None):
    return (
        n.selection(data, endpoint, rows) if kind == "night" else s.selection(data, endpoint, rows)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, default=ANSWERS / "round1.jsonl")
    parser.add_argument("--probes", choices=["round1", "night"], default="round1")
    parser.add_argument("--halves", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)

    corpus, stats, kind = statistics_for(
        s.build_corpus(s.read_jsonl(args.answers), prefix="ref-"), args.probes
    )
    rng = np.random.default_rng(args.seed)
    endpoints = sorted(corpus.answers)
    lines = [f"# Noise floor on the reference endpoints, `{args.answers.name}`", ""]
    for name, data in stats:
        qids = list(data)
        reps = {e: min(len(corpus.answers[e].get(q, [])) for q in qids) for e in endpoints}
        within: dict[str, list[s.PermutationTest]] = {}
        for e in endpoints:
            for first, second in halves(reps[e])[: args.halves]:
                test = run_test(
                    kind, data, select(kind, data, e, first), select(kind, data, e, second), rng
                )
                within.setdefault(e, []).append(test)
        rows = []
        for a, b in itertools.combinations(endpoints, 2):
            full = run_test(kind, data, select(kind, data, a), select(kind, data, b), rng)
            size = min(reps[a], reps[b]) // 2
            matched = []
            if size >= 2:
                parts = [list(range(size)), list(range(size, 2 * size))]
                for pa, pb in itertools.product(parts, parts):
                    test = run_test(
                        kind, data, select(kind, data, a, pa), select(kind, data, b, pb), rng
                    )
                    matched.append(test.z)
            rows.append((a, b, pair_type(a, b), full, matched))
        lines += [
            f"## Statistic: {name} ({', '.join(sorted(qids))})",
            "",
            "Answers per endpoint (minimum over these prompts): "
            + ", ".join(f"{e} {reps[e]}" for e in endpoints)
            + ".",
            "",
            "| rows | count | median z | share p<0.05 | median z at half size |",
            "|---|---|---|---|---|",
        ]
        zero = [t for tests in within.values() for t in tests]
        lines.append(
            f"| 0 zero point: two halves of one endpoint | {len(zero)} "
            f"| {fmt([t.z for t in zero])} | {share([t.p for t in zero])} "
            f"| {fmt([t.z for t in zero])} |"
        )
        for kind in sorted({r[2] for r in rows}):
            chosen = [r for r in rows if r[2] == kind]
            lines.append(
                f"| {kind} | {len(chosen)} | {fmt([r[3].z for r in chosen])} "
                f"| {share([r[3].p for r in chosen])} "
                f"| {fmt([statistics.mean(r[4]) for r in chosen if r[4]])} |"
            )
        lines += ["", "| a | b | type | z | p | z at half size |", "|---|---|---|---|---|---|"]
        for a, b, kind, full, matched in rows:
            half = f"{statistics.mean(matched):.2f}" if matched else "n/a"
            lines.append(f"| {a} | {b} | {kind} | {full.z:.2f} | {full.p:.4f} | {half} |")
        lines.append("")
    out = HERE / f"noise_{args.answers.stem}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
