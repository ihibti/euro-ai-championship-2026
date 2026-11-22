"""Reconnaissance of endpoints from the recon prompts (idn_identity, txt_story, txt_health,
bat_trivia, bat_choice): what each endpoint says it is, its lengths, its spread, and families.

uv run python tasks/model_equality/analysis/recon.py \
    --answers tasks/model_equality/answers/round1.jsonl --prefix ep-

Per endpoint: the model and lab names it gives, the answer lengths, how many battery items it
answers in format, and how much its own repeats agree (a hot endpoint agrees little with
itself). Families: endpoints are joined, average linkage, while the mean agreement of their
trivia answers is at least half the mean self-agreement of their members. That rule is a first
pass for the family term only; the identity claims are printed beside it as separate evidence.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows, parse_numbered  # noqa: E402

WORD = re.compile(r"[a-z0-9']+")
NAMES = (
    "llama",
    "meta",
    "qwen",
    "alibaba",
    "mistral",
    "mixtral",
    "olmo",
    "allen",
    "ai2",
    "gemma",
    "google",
    "gemini",
    "gpt",
    "openai",
    "chatgpt",
    "claude",
    "anthropic",
    "phi",
    "microsoft",
    "tinyllama",
    "smollm",
    "hugging",
    "deepseek",
    "falcon",
    "tii",
    "pythia",
    "eleutherai",
    "vicuna",
    "lmsys",
    "zephyr",
    "stablelm",
    "stability",
    "granite",
    "ibm",
    "yi",
    "baichuan",
    "internlm",
    "nvidia",
    "nemotron",
    "cohere",
    "command",
    "bloom",
    "bigscience",
    "opt",
)
Pair = tuple[str, str]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def identity_claims(text: str) -> list[str]:
    tokens = set(words(text))
    return [name for name in NAMES if name in tokens]


def first_words(text: str, count: int) -> list[str | None]:
    out: list[str | None] = []
    for value in parse_numbered(text, count):
        tokens = words(value or "")
        out.append(tokens[0] if tokens else None)
    return out


ECHOED = re.compile(r"^(\s*\d+[.)])[^\n]*\?[ \t]*\n+[ \t]*[-*\u2022][ \t]*", re.MULTILINE)


def battery_answers(text: str, count: int) -> list[str | None]:
    return first_words(ECHOED.sub(r"\1 ", text), count)


def cluster_k(
    endpoints: Sequence[str], similarity: Mapping[Pair, float], k: int
) -> list[list[str]]:
    def link(first: Sequence[str], second: Sequence[str]) -> float:
        values = [
            similarity.get((x, y), similarity.get((y, x), math.nan)) for x in first for y in second
        ]
        defined = [v for v in values if math.isfinite(v)]
        return float(np.mean(defined)) if defined else -math.inf

    clusters = [[e] for e in endpoints]
    while len(clusters) > k:
        i, j = max(
            itertools.combinations(range(len(clusters)), 2),
            key=lambda pair: link(clusters[pair[0]], clusters[pair[1]]),
        )
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
    return sorted(clusters, key=lambda c: c[0])


def item_agreement(first: Sequence[str | None], second: Sequence[str | None]) -> float | None:
    both = [(a, b) for a, b in zip(first, second, strict=False) if a is not None and b is not None]
    return float(np.mean([a == b for a, b in both])) if both else None


def word_cosine(first: str, second: str) -> float | None:
    a, b = Counter(words(first)), Counter(words(second))
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return sum(a[k] * b[k] for k in a) / norm if norm else None


def mean_defined(values) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def self_agreement(answers: Sequence, compare: Callable) -> float | None:
    return mean_defined(compare(a, b) for a, b in itertools.combinations(answers, 2))


def cross_agreement(first: Sequence, second: Sequence, compare: Callable) -> float | None:
    return mean_defined(compare(a, b) for a in first for b in second)


def family_clusters(
    endpoints: Sequence[str], similarity: Mapping[Pair, float], own: Mapping[str, float]
) -> list[list[str]]:
    def sim(x: str, y: str) -> float:
        return similarity.get((x, y), similarity.get((y, x), 0.0))

    clusters = [[e] for e in endpoints]
    while len(clusters) > 1:
        best, pair = -math.inf, None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            cross = float(np.mean([sim(x, y) for x in clusters[i] for y in clusters[j]]))
            base = float(np.mean([own[e] for e in clusters[i] + clusters[j]]))
            if cross >= 0.5 * base and cross - 0.5 * base > best:
                best, pair = cross - 0.5 * base, (i, j)
        if pair is None:
            break
        i, j = pair
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
    return sorted(clusters, key=lambda c: c[0])


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def collect(rows) -> dict[str, dict[str, list[str]]]:
    cells: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("response") is not None:
            cells[str(row["endpoint"])][str(row["id"])].append(
                (int(row["repeat"]), str(row["response"]))
            )
    return {
        e: {p: [t for _, t in sorted(v)] for p, v in prompts.items()}
        for e, prompts in cells.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ep-")
    parser.add_argument("--below", type=int, default=None, help="use only repeats below this")
    parser.add_argument("--k", type=int, default=3, help="clusters for the average-link cut")
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    if args.below is not None:
        rows = [r for r in rows if int(r["repeat"]) < args.below]
    cells = collect(rows)
    endpoints = sorted(cells)
    trivia = {
        e: [battery_answers(t, 30) for t in cells[e].get("bat_trivia", [])] for e in endpoints
    }
    choice = {e: [first_words(t, 30) for t in cells[e].get("bat_choice", [])] for e in endpoints}
    print(
        "| endpoint | claims | median chars story/health/identity | trivia items | choice items "
        "| self: trivia, choice, story, health |"
    )
    print("|---|---|---|---|---|---|")
    for e in endpoints:
        claims = Counter(n for t in cells[e].get("idn_identity", []) for n in identity_claims(t))
        lengths = [
            fmt(float(np.median([len(t) for t in cells[e].get(p, [])])), 0)
            if cells[e].get(p)
            else "-"
            for p in ("txt_story", "txt_health", "idn_identity")
        ]
        parsed_t = [sum(v is not None for v in a) for a in trivia[e]]
        parsed_c = [sum(v is not None for v in a) for a in choice[e]]
        spreads = [
            self_agreement(trivia[e], item_agreement),
            self_agreement(choice[e], item_agreement),
            self_agreement(cells[e].get("txt_story", []), word_cosine),
            self_agreement(cells[e].get("txt_health", []), word_cosine),
        ]
        named = ", ".join(f"{k} {v}" for k, v in claims.most_common()) or "-"
        print(
            f"| {e} | {named} | {'/'.join(lengths)} | {' '.join(map(str, parsed_t))} | "
            f"{' '.join(map(str, parsed_c))} | {', '.join(fmt(s) for s in spreads)} |"
        )
    print("\nidentity answers, first 150 characters of each repeat")
    for e in endpoints:
        for text in cells[e].get("idn_identity", []):
            print(f"- {e}: {text[:150]!r}")
    print("\ntxt_health openings (a persona or a system prompt shows here first)")
    for e in endpoints:
        for text in cells[e].get("txt_health", []):
            print(f"- {e}: {text[:110]!r}")
    print("\nbat_choice: share of items answered A (copying the example '1. A' gives 1.00)")
    for e in endpoints:
        letters = [v for a in choice[e] for v in a if v in ("a", "b")]
        share = fmt(letters.count("a") / len(letters)) if letters else "-"
        print(f"- {e}: {share} over {len(letters)} items")
    similarity: dict[Pair, float] = {}
    own: dict[str, float] = {}
    for e in endpoints:
        own[e] = self_agreement(trivia[e], item_agreement) or 0.0
    print("\ntrivia agreement between endpoints (diagonal: self)")
    print("| | " + " | ".join(e[-2:] for e in endpoints) + " |")
    print("|---" * (len(endpoints) + 1) + "|")
    for x in endpoints:
        row = []
        for y in endpoints:
            if x == y:
                row.append(fmt(own[x]))
                continue
            value = cross_agreement(trivia[x], trivia[y], item_agreement)
            similarity[(x, y)] = math.nan if value is None else value
            row.append(fmt(value))
        print(f"| {x[-2:]} | " + " | ".join(row) + " |")
    print("\nfirst-pass families (cross trivia agreement >= half the members' self-agreement)")
    zeroed = {pair: (v if math.isfinite(v) else 0.0) for pair, v in similarity.items()}
    for number, members in enumerate(family_clusters(endpoints, zeroed, own), start=1):
        claims = Counter(
            n for e in members for t in cells[e].get("idn_identity", []) for n in identity_claims(t)
        )
        print(f"- family {number}: {', '.join(members)}; claims: {dict(claims.most_common(5))}")
    print(f"\naverage-link cut into {args.k} clusters on trivia agreement (missing pairs ignored)")
    for number, members in enumerate(cluster_k(endpoints, similarity, args.k), start=1):
        claims = Counter(
            n for e in members for t in cells[e].get("idn_identity", []) for n in identity_claims(t)
        )
        print(f"- cluster {number} ({len(members)}): {', '.join(members)}; claims: {dict(claims)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
