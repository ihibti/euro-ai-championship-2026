"""A ranking of the non-prompted endpoints of each family by one combined statistic over the
frozen probes, for a cut chosen elsewhere.

uv run python tasks/model_equality/analysis/rank.py --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv

Frozen on 2026-11-22 at 02:49 Paris, before any fresh battery answer existed.
Per family, the prompted copies (PROMPTED) are neither ranked nor pooled. Each other endpoint
is compared with the pool of the other non-prompted members, one p-value per probe:
- d1_repeat150, stop.ran_long rate, one-sided "runs long more" (the references' LoRA direction),
  Fisher;
- the batteries, modal.modal_pvalue (more flips, one-sided);
- e2_leak_quote, yes rate, two-sided Fisher (no reference direction);
- e1_leak_repeat, default-text rate (rank.MARKERS), two-sided Fisher.
A probe without answers for the endpoint is left out. The p-values are combined by Fisher's
method (-2 sum log p against chi-square with 2k degrees of freedom). The Cauchy combination was
tried first and dropped before any fresh row: a two-sided p of 1 sends tan((0.5 - p) pi) to
minus infinity and every combined p to 1. Fisher's method assumes independent probes, so the
combined p is a ranking score, not a calibrated test. Endpoints are listed from the smallest.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaksplit  # noqa: E402
import modal  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

ANSWERS = ROOT / "tasks/model_equality/answers"
PROMPTED = {"ep-02", "ep-23", "ep-08", "ep-20", "ep-16", "ep-29", "ref-03", "ref-05"}
MARKER = {"A": "qwen", "B": "qwen", "C": "llama", "F": "smollm"}
CLIP = 1e-15
MARKERS = {
    **leaksplit.MARKERS,
    "smollm": re.compile(r"helpful ai assistant(?: named smollm)?,? trained by hugging face", re.I),
}


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    total, hits = n1 + n2, a + b
    denominator = math.comb(total, hits)
    upper = range(a, min(n1, hits) + 1)
    return min(1.0, sum(math.comb(n1, k) * math.comb(n2, hits - k) for k in upper) / denominator)


def fisher_combined(pvalues: Sequence[float]) -> float:
    kept = [max(p, CLIP) for p in pvalues if math.isfinite(p)]
    if not kept:
        return float("nan")
    half = -sum(math.log(p) for p in kept)
    return min(1.0, math.exp(-half) * sum(half**i / math.factorial(i) for i in range(len(kept))))


def rate_pvalue(values: Mapping[str, list[bool]], e: str, pool: Sequence[str], greater: bool):
    own = values.get(e, [])
    rest = [v for m in pool if m != e for v in values.get(m, [])]
    if not own or not rest:
        return float("nan")
    if greater:
        return fisher_greater(sum(own), len(own), sum(rest), len(rest))
    return stop.fisher_two_sided(sum(own), len(own), sum(rest), len(rest))


def long_flags(rows: Sequence[Mapping[str, object]], members: Sequence[str]):
    out: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["id"] == "d1_repeat150" and row["endpoint"] in members and row.get("response"):
            out[str(row["endpoint"])].append(stop.ran_long(str(row["response"])))
    return dict(out)


def probe_pvalues(
    rows: Sequence[Mapping[str, object]], members: Sequence[str], marker: str
) -> dict[str, dict[str, float]]:
    ranked = [m for m in members if m not in PROMPTED]
    values = modal.encode(modal.battery_tokens(rows, ranked))
    longs = long_flags(rows, ranked)
    yes = leaksplit.flags(rows, ranked, "e2_leak_quote", leaksplit.said_yes)
    default = leaksplit.flags(
        rows, ranked, "e1_leak_repeat", leaksplit.quotes_default(MARKERS[marker])
    )
    out: dict[str, dict[str, float]] = {}
    for e in ranked:
        out[e] = {
            "d1 long": rate_pvalue(longs, e, ranked, greater=True),
            "modal": modal.modal_pvalue(values, e, [m for m in ranked if m in values])
            if e in values
            else float("nan"),
            "e2 yes": rate_pvalue(yes, e, ranked, greater=False),
            "e1 default": rate_pvalue(default, e, ranked, greater=False),
        }
    return out


def ranking(pvalues: Mapping[str, Mapping[str, float]]) -> list[tuple[str, float]]:
    combined = [(e, fisher_combined(list(p.values()))) for e, p in pvalues.items()]
    return sorted(combined, key=lambda item: (not math.isfinite(item[1]), item[1], item[0]))


def read_families(path: Path | None, endpoints: Sequence[str]) -> dict[str, list[str]]:
    if path is None:
        return {"F": sorted(endpoints)}
    families: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["endpoint"] in endpoints:
                families[row["family"]].append(row["endpoint"])
    return {f: sorted(m) for f, m in families.items()}


def fmt(p: float) -> str:
    return "-" if not math.isfinite(p) else f"{p:.2g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--answers",
        nargs="+",
        default=[str(ANSWERS / n) for n in ("round1.jsonl", "night-d.jsonl", "night-e.jsonl")],
    )
    parser.add_argument("--prefix", default="ep-")
    parser.add_argument("--families", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    endpoints = sorted({str(r["endpoint"]) for r in rows})
    for family, members in read_families(args.families, endpoints).items():
        pvalues = probe_pvalues(rows, members, MARKER.get(family, "qwen"))
        print(f"\n== family {family}")
        print("| rank | endpoint | combined p | d1 long | modal | e2 yes | e1 default |")
        print("|---|---|---|---|---|---|---|")
        for k, (e, p) in enumerate(ranking(pvalues), start=1):
            cells = " | ".join(fmt(pvalues[e][name]) for name in pvalues[e])
            print(f"| {k} | {e} | {fmt(p)} | {cells} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
