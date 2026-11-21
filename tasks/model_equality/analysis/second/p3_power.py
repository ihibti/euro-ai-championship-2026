"""How many p3_repeat answers per endpoint call a LoRA copy, with exact binomial tails.

Plain copies hit the long run at rate p0; a LoRA copy at rate p1. An endpoint is called LoRA when
its long-run count k reaches c, the smallest count whose tail under p0 is at most alpha. Two
readings of "at most one false call among 33 endpoints": the expected number of false calls is at
most 1 (alpha = 1/33), or two or more false calls happen with probability at most 0.10.

Usage: uv run python tasks/model_equality/analysis/second/p3_power.py
Writes p3_power.md next to this file.
"""

from __future__ import annotations

from pathlib import Path

from scipy.stats import binom

HERE = Path(__file__).resolve().parent
ENDPOINTS = 33


def alpha_two_or_more(limit: float = 0.10) -> float:
    low, high = 0.0, 0.2
    for _ in range(60):
        mid = (low + high) / 2.0
        if binom.sf(1, ENDPOINTS, mid) > limit:
            high = mid
        else:
            low = mid
    return low


def cutoff(n: int, p0: float, alpha: float) -> int:
    for c in range(n + 2):
        if binom.sf(c - 1, n, p0) <= alpha:
            return c
    return n + 1


def main() -> int:
    lines: list[str] = []
    for p0 in (0.1, 0.2):
        lines += table(p0)
    (HERE / "p3_power.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def table(p0: float) -> list[str]:
    readings = {
        "E[false calls] <= 1": 1.0 / ENDPOINTS,
        "P(2+ false calls) <= 0.10": alpha_two_or_more(),
    }
    lines = [
        f"# p3_repeat: answers per endpoint to call a LoRA copy, plain rate {p0}",
        "",
        f"Plain rate p0 = {p0}. Exact one-sided binomial test per endpoint with p0 known. Power is"
        " the chance that one LoRA endpoint is called; false calls are counted over 33 endpoints"
        " as if none were LoRA (the worst case).",
        "",
    ]
    for label, alpha in readings.items():
        lines += [
            f"## {label}: per-endpoint alpha {alpha:.4f}",
            "",
            "| answers n | call at k >= c | actual alpha | expected false calls in 33 "
            "| power, p1 = 0.6 | power, p1 = 0.35 |",
            "|---|---|---|---|---|---|",
        ]
        for n in (4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
            c = cutoff(n, p0, alpha)
            actual = float(binom.sf(c - 1, n, p0))
            lines.append(
                f"| {n} | {c} | {actual:.4f} | {ENDPOINTS * actual:.2f} "
                f"| {binom.sf(c - 1, n, 0.6):.3f} | {binom.sf(c - 1, n, 0.35):.3f} |"
            )
        lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
