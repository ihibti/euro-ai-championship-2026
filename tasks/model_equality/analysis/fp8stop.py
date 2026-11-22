"""A registered battery-side check of the fp8 stop hint: does ref-02 (fp8) stop earlier than
ref-01 (plain) on the fresh d probes, by the stop.py measures?

uv run python tasks/model_equality/analysis/fp8stop.py

Frozen on 2026-11-22 at 04:34 Paris. The desk's chain was then appending night_d rows at 30
repeats on ref-01 and ref-02; 4 rows with repeat >= 11 existed and none had been read.
- Rows: d1_repeat150, d2_story100, d3_openlist, d4_copycont on ref-01 and ref-02, repeat >=
  FRESH (11), from night-d.jsonl. Repeats 0 to 10 were read before this file and are excluded.
- Measure per answer, larger meaning a later stop: d1 stop.repetitions, d2 stop.word_count,
  d3 stop.list_items, d4 the stripped length in characters (the monotone form of
  stop.copy_only).
- Per probe, a one-sided permutation test that fp8 stops earlier: the statistic is ref-01's
  mean midrank minus ref-02's over the pooled answers; DRAWS relabellings, seed 0;
  p = (1 + draws at least as large) / (DRAWS + 1).
- Combined: Fisher's method over the four probes (rank.fisher_combined).
- Decision: the hint is revived on this side when the combined p < ALPHA (0.01) and at least 3 of
  the 4 probes point the predicted way (fp8's mean midrank below plain's). Otherwise it is not.
- Described only: the d1 stop.ran_long and d4 stop.copy_only counts.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rank  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PLAIN, FP8 = "ref-01", "ref-02"
FRESH = 11
ALPHA = 0.01
DRAWS = 20000
SEED = 0
MIN_AGREE = 3


def stripped_length(text: str) -> int:
    return len(text.strip())


MEASURES: dict[str, Callable[[str], int]] = {
    "d1_repeat150": stop.repetitions,
    "d2_story100": stop.word_count,
    "d3_openlist": stop.list_items,
    "d4_copycont": stripped_length,
}


def fresh_texts(
    rows: Iterable[Mapping[str, object]],
    endpoints: Sequence[str],
    min_repeat: int = FRESH,
    max_repeat: int | None = None,
) -> dict[tuple[str, str], list[str]]:
    cells: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        endpoint, prompt_id = str(row["endpoint"]), str(row["id"])
        if endpoint not in endpoints or prompt_id not in MEASURES or row.get("response") is None:
            continue
        repeat = int(row["repeat"])
        if repeat < min_repeat or (max_repeat is not None and repeat > max_repeat):
            continue
        cells[(endpoint, prompt_id)].append((repeat, str(row["response"])))
    return {k: [t for _, t in sorted(v)] for k, v in cells.items()}


def later_stop_gap(plain: Sequence[float], other: Sequence[float]) -> float:
    ranks = stop.midranks([*plain, *other])
    return float(ranks[: len(plain)].mean() - ranks[len(plain) :].mean())


def earlier_pvalue(
    plain: Sequence[float], other: Sequence[float], draws: int = DRAWS, seed: int = SEED
) -> float:
    if not len(plain) or not len(other):
        return float("nan")
    ranks = stop.midranks([*plain, *other])
    n = len(plain)
    observed = ranks[:n].mean() - ranks[n:].mean()
    rng = np.random.default_rng(seed)
    order = rng.random((draws, len(ranks))).argsort(axis=1)
    shuffled = ranks[order]
    gaps = shuffled[:, :n].mean(axis=1) - shuffled[:, n:].mean(axis=1)
    return float((np.sum(gaps >= observed - 1e-9) + 1) / (draws + 1))


def verdict(
    pvalues: Mapping[str, float], gaps: Mapping[str, float], alpha: float = ALPHA
) -> tuple[float, int, bool]:
    combined = rank.fisher_combined(list(pvalues.values()))
    agree = sum(g > 0 for g in gaps.values())
    return combined, agree, bool(combined < alpha and agree >= MIN_AGREE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--answers", nargs="+", default=[str(ROOT / "tasks/model_equality/answers/night-d.jsonl")]
    )
    parser.add_argument("--min-repeat", type=int, default=FRESH)
    parser.add_argument("--max-repeat", type=int, default=None)
    args = parser.parse_args(argv)
    texts = fresh_texts(load_rows(args.answers), (PLAIN, FP8), args.min_repeat, args.max_repeat)
    pvalues, gaps = {}, {}
    last = "end" if args.max_repeat is None else args.max_repeat
    print(f"repeats {args.min_repeat} to {last}")
    print("| probe | n plain / fp8 | median plain | median fp8 | plain minus fp8 midrank | p |")
    print("|---|---|---|---|---|---|")
    for prompt_id, measure in MEASURES.items():
        plain = [measure(t) for t in texts.get((PLAIN, prompt_id), [])]
        other = [measure(t) for t in texts.get((FP8, prompt_id), [])]
        if not plain or not other:
            continue
        gaps[prompt_id] = later_stop_gap(plain, other)
        pvalues[prompt_id] = earlier_pvalue(plain, other)
        print(
            f"| {prompt_id} | {len(plain)} / {len(other)} | {np.median(plain):.0f} "
            f"| {np.median(other):.0f} | {gaps[prompt_id]:+.2f} | {pvalues[prompt_id]:.4f} |"
        )
    for prompt_id, flag in (("d1_repeat150", stop.ran_long), ("d4_copycont", stop.copy_only)):
        a = sum(flag(t) for t in texts.get((PLAIN, prompt_id), []))
        b = sum(flag(t) for t in texts.get((FP8, prompt_id), []))
        n1, n2 = len(texts.get((PLAIN, prompt_id), [])), len(texts.get((FP8, prompt_id), []))
        print(f"- {prompt_id} {flag.__name__}: plain {a}/{n1}, fp8 {b}/{n2}")
    combined, agree, revived = verdict(pvalues, gaps)
    print(f"combined p {combined:.4g}; probes pointing the predicted way {agree} of {len(gaps)}")
    print(f"fp8 hint revived on the stop.py side: {revived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
