"""Second method's decision rule for p3_repeat, frozen before the confirmation answers.

Feature: an answer is a "long run" when it holds at least 100 consecutive "river" words (the
plain reference model mostly writes the word once; the LoRA copies mostly run to the 256-token
cap, about 128 or 256 words).

Rule, inside one family (endpoints already known to share a base model):
1. Fit one long-run rate for the whole family, and a mixture of two rates, each endpoint
   belonging to one class (binomial mixture, EM from 20 starts).
2. Likelihood ratio of two rates against one, with a parametric-bootstrap p-value under the
   one-rate fit. If p >= 0.01, the family shows no LoRA signal and the rule calls nothing.
3. Otherwise each endpoint goes to the class with the larger posterior. The class holding more
   endpoints is the base class (a tie goes to the lower rate); it forms one model group.
4. Endpoints in the other class are called LoRA and stay singletons, even when they cannot be
   told apart from each other: on the references, adapters of different strength are different
   models, so a merge of two such endpoints is wrong more often than right.

Usage, from the repository root:
  uv run python tasks/model_equality/analysis/second/p3rule.py --answers FILE --prefix ref-
  uv run python tasks/model_equality/analysis/second/p3rule.py --answers FILE --prefix ep- \\
      --families recon_ep.csv
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import secondlib as s

from euroai.me_labels import canonical_labels, render_rows
from euroai.me_score import parse_labels, same_model_mcc

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parents[1] / "upstream" / "practice_answers.csv"
LONG = 100
ALPHA = 0.01


def run_length(response: str) -> int:
    best = count = 0
    for word in re.findall(r"[A-Za-z]+", response):
        count = count + 1 if word.lower() == "river" else 0
        best = max(best, count)
    return best


def is_long(response: str) -> bool:
    return run_length(response) >= LONG


def _loglik(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return k * np.log(p) + (n - k) * np.log1p(-p)


def one_rate(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    rate = float(k.sum() / max(n.sum(), 1.0))
    return rate, float(_loglik(k, n, np.full(k.shape, rate)).sum())


def two_rates(
    k: np.ndarray, n: np.ndarray, starts: int = 20, steps: int = 200, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        rates = np.sort(rng.uniform(0.02, 0.98, 2))
        weight = 0.5
        for _ in range(steps):
            low = np.log(weight) + _loglik(k, n, np.full(k.shape, rates[0]))
            high = np.log(1.0 - weight) + _loglik(k, n, np.full(k.shape, rates[1]))
            top = np.maximum(low, high)
            post_high = np.exp(high - top) / (np.exp(low - top) + np.exp(high - top))
            weight = float(np.clip(1.0 - post_high.mean(), 1e-6, 1.0 - 1e-6))
            hk = (post_high * k).sum()
            hn = (post_high * n).sum()
            lk = ((1.0 - post_high) * k).sum()
            ln = ((1.0 - post_high) * n).sum()
            rates = np.array([lk / max(ln, 1e-9), hk / max(hn, 1e-9)])
        low = np.log(weight) + _loglik(k, n, np.full(k.shape, rates[0]))
        high = np.log(1.0 - weight) + _loglik(k, n, np.full(k.shape, rates[1]))
        total = float(np.logaddexp(low, high).sum())
        if best is None or total > best[1]:
            best = (rates, total, post_high, weight)
    return best


def mixture_p(k: np.ndarray, n: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    rate, single = one_rate(k, n)
    _, double, _, _ = two_rates(k, n)
    observed = 2.0 * (double - single)
    null = []
    for _ in range(draws):
        sim = rng.binomial(n.astype(int), rate).astype(float)
        _, sim_single = one_rate(sim, n)
        _, sim_double, _, _ = two_rates(sim, n, starts=5, steps=100)
        null.append(2.0 * (sim_double - sim_single))
    return (1.0 + float(np.sum(np.array(null) >= observed - 1e-9))) / (draws + 1.0)


@dataclass(frozen=True)
class FamilyCall:
    base: list[str]
    lora: list[str]
    p: float
    rates: tuple[float, float]
    long_share: dict[str, float]


def call_family(
    endpoints: Sequence[str],
    long_counts: Mapping[str, int],
    totals: Mapping[str, int],
    rng: np.random.Generator,
    draws: int = 499,
) -> FamilyCall:
    names = sorted(endpoints)
    k = np.array([float(long_counts[e]) for e in names])
    n = np.array([float(totals[e]) for e in names])
    share = {e: long_counts[e] / max(totals[e], 1) for e in names}
    if len(names) < 3 or n.min() < 1:
        return FamilyCall(names, [], 1.0, (float("nan"), float("nan")), share)
    p = mixture_p(k, n, draws, rng)
    rates, _, post_high, _ = two_rates(k, n)
    if p >= ALPHA:
        return FamilyCall(names, [], p, (float(rates[0]), float(rates[1])), share)
    high = [e for e, post in zip(names, post_high, strict=True) if post > 0.5]
    low = [e for e in names if e not in high]
    if len(high) > len(low):
        base, lora = high, low
    else:
        base, lora = low, high
    return FamilyCall(base, lora, p, (float(rates[0]), float(rates[1])), share)


def families_from(path: Path | None, endpoints: Sequence[str]) -> list[list[str]]:
    if path is None:
        return [sorted(endpoints)]
    labels = parse_labels(path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {}
    for endpoint in sorted(endpoints):
        if endpoint in labels:
            groups.setdefault(labels[endpoint][0], []).append(endpoint)
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--families", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    rows = [row for path in args.answers for row in s.read_jsonl(path)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    answers = {e: per.get("p3_repeat", []) for e, per in corpus.answers.items()}
    answers = {e: r for e, r in answers.items() if r}
    long_counts = {e: sum(is_long(r) for r in rs) for e, rs in answers.items()}
    totals = {e: len(rs) for e, rs in answers.items()}
    rng = np.random.default_rng(args.seed)
    labels: dict[str, tuple[str, str]] = {}
    for index, family in enumerate(families_from(args.families, list(answers)), start=1):
        call = call_family(family, long_counts, totals, rng, args.draws)
        print(
            f"family {index}: p={call.p:.4f} rates={call.rates[0]:.2f}/{call.rates[1]:.2f} "
            f"base={' '.join(call.base)} lora={' '.join(call.lora) or '-'}"
        )
        print("  long-run share:", ", ".join(f"{e} {v:.2f}" for e, v in call.long_share.items()))
        for e in call.base:
            labels[e] = (f"F{index}", f"F{index}-base")
        for e in call.lora:
            labels[e] = (f"F{index}", f"F{index}-{e}")
    if args.out:
        rows_out = canonical_labels([(e, f, m) for e, (f, m) in labels.items()])
        args.out.write_text(render_rows(rows_out), encoding="utf-8")
    if args.prefix == "ref-":
        truth = parse_labels(TRUTH.read_text(encoding="utf-8"), prefix="ref-")
        if set(truth) <= set(labels):
            result = same_model_mcc(truth, labels)
            print(
                f"mcc={result.mcc:.4f} tp={result.tp} fp={result.fp} fn={result.fn} tn={result.tn}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
