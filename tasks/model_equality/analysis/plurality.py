"""Per-position plurality profiles of repeated answers, and the distance between endpoints.

uv run python tasks/model_equality/analysis/plurality.py \
    --answers tasks/model_equality/answers/night-a.jsonl --prefix ref-

For each probe and endpoint, every repeat becomes a sequence of tokens: the words that follow
the copied passage (P1, prefix p1), the 0/1 symbols (P2, p2) or the first word of each numbered
item (P4, p4). The plurality token of each position is kept when the endpoint gives it in at
least --min-share of its repeats; a repeat without a token there counts against it. The distance
between two endpoints is their disagreement rate on the positions both keep. Self-agreement is
the mean plurality share; the first-divergence position is averaged over pairs of repeats.
P3 (p3) is summarised by its repetition count and P5 (p5) by the function name chosen.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probes"))

from build_night import P1, P5_NAMES  # noqa: E402
from melib import load_rows, parse_numbered  # noqa: E402

WORD = re.compile(r"[a-z0-9']+")
MARKER = re.compile(r"^\s*(?:\[\d+\]|\d+[.):]|[•*\-])\s*")
RUN = re.compile(r"(?<![0-9A-Za-z])[01]+(?![0-9A-Za-z])")
Token = str | None

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
PAIRS = [("ref-02", "fp8"), ("ref-09", "LoRA 0.3"), ("ref-04", "LoRA 0.7"), ("ref-07", "LoRA 1.5")]


@dataclass(frozen=True)
class Profile:
    tokens: tuple[Token, ...]
    shares: tuple[float, ...]
    repeats: int


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def find(tokens: Sequence[str], pattern: Sequence[str], start: int = 0) -> int | None:
    size = len(pattern)
    for index in range(start, len(tokens) - size + 1):
        if list(tokens[index : index + size]) == list(pattern):
            return index
    return None


def first_divergence(first: Sequence[Token], second: Sequence[Token]) -> int:
    for index, (a, b) in enumerate(zip(first, second, strict=False)):
        if a != b:
            return index
    return min(len(first), len(second))


def continuation(
    text: str, passage: Sequence[str], anchor: int = 4
) -> tuple[list[str] | None, int]:
    tokens = words(text)
    start = find(tokens, passage[:anchor])
    copied = 0 if start is None else first_divergence(tokens[start:], passage)
    end = find(tokens, passage[-anchor:], start or 0)
    if end is None:
        return None, copied
    return tokens[end + anchor :], copied


def binary_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for line in text.splitlines():
        for run in RUN.findall(MARKER.sub("", line, count=1)):
            symbols.extend(run)
    return symbols


def items(text: str, count: int = 20) -> list[Token]:
    out: list[Token] = []
    for value in parse_numbered(text, count):
        tokens = words(value) if value is not None else []
        out.append(tokens[0] if tokens else None)
    return out


def repeat_stats(text: str, word: str) -> tuple[int, int]:
    tokens = words(text)
    count = sum(token == word for token in tokens)
    other = next((i for i, token in enumerate(tokens) if token != word), len(tokens))
    return count, other


def chosen_name(text: str, names: Sequence[str]) -> str | None:
    found = [(text.find(name), name) for name in names if name in text]
    return min(found)[1] if found else None


def profile(sequences: Sequence[Sequence[Token]], length: int) -> Profile:
    tokens: list[Token] = []
    shares: list[float] = []
    for index in range(length):
        column = [seq[index] for seq in sequences if index < len(seq) and seq[index] is not None]
        counts = Counter(column)
        if not counts:
            tokens.append(None)
            shares.append(0.0)
            continue
        best = max(counts.values())
        tokens.append(min(token for token, count in counts.items() if count == best))
        shares.append(best / len(sequences))
    return Profile(tuple(tokens), tuple(shares), len(sequences))


def kept(prof: Profile, min_share: float) -> list[int]:
    return [i for i, share in enumerate(prof.shares) if share >= min_share and prof.tokens[i]]


def distance(first: Profile, second: Profile, min_share: float) -> tuple[float, int]:
    shared = sorted(set(kept(first, min_share)) & set(kept(second, min_share)))
    if not shared:
        return float("nan"), 0
    return float(np.mean([first.tokens[i] != second.tokens[i] for i in shared])), len(shared)


def self_agreement(prof: Profile) -> float:
    return float(np.mean(prof.shares)) if prof.shares else float("nan")


def kept_fraction(prof: Profile, min_share: float) -> float:
    return len(kept(prof, min_share)) / len(prof.shares) if prof.shares else float("nan")


def mean_first_divergence(
    first: Sequence[Sequence[Token]], second: Sequence[Sequence[Token]] | None = None
) -> float:
    if second is None:
        pairs = list(itertools.combinations(first, 2))
    else:
        pairs = [(a, b) for a in first for b in second]
    values = [first_divergence(a, b) for a, b in pairs]
    return float(np.mean(values)) if values else float("nan")


PASSAGE = words(P1.split("\n\n", 1)[1])


def p1_tokens(text: str) -> list[Token]:
    tokens, _ = continuation(text, PASSAGE)
    return list(tokens) if tokens is not None else []


SEQUENCERS: dict[str, tuple[Callable[[str], list[Token]], int, str]] = {
    "p1": (p1_tokens, 80, "first 80 words after the copied passage"),
    "p2": (binary_symbols, 60, "first 60 symbols"),
    "p4": (items, 20, "20 items"),
}


def sequences_by_endpoint(
    rows: Sequence[Mapping[str, object]], prompt_id: str, extract: Callable[[str], list[Token]]
) -> dict[str, list[list[Token]]]:
    cells: dict[str, list[tuple[int, list[Token]]]] = defaultdict(list)
    for row in rows:
        if row["id"] != prompt_id or row.get("response") is None:
            continue
        cells[str(row["endpoint"])].append((int(row["repeat"]), extract(str(row["response"]))))
    return {e: [seq for _, seq in sorted(v, key=lambda t: t[0])] for e, v in cells.items()}


def fmt(value: float, digits: int = 2) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def report_positional(rows, prompt_id: str, min_share: float) -> None:
    extract, length, label = SEQUENCERS[prompt_id.split("_", 1)[0]]
    seqs = sequences_by_endpoint(rows, prompt_id, extract)
    endpoints = sorted(seqs)
    profiles = {e: profile(seqs[e], length) for e in endpoints}
    print(f"\n== {prompt_id}: {label}, keep at share >= {min_share}")
    print("| endpoint | n | self-agreement | kept | split-half distance | self first-div |")
    print("|---|---|---|---|---|---|")
    for e in endpoints:
        even, odd = profile(seqs[e][0::2], length), profile(seqs[e][1::2], length)
        split, n_split = distance(even, odd, min_share)
        print(
            f"| {e} | {len(seqs[e])} | {fmt(self_agreement(profiles[e]))} | "
            f"{fmt(kept_fraction(profiles[e], min_share))} | {fmt(split)} ({n_split}) | "
            f"{fmt(mean_first_divergence(seqs[e]), 1)} |"
        )
    print("\ndistance matrix (positions kept by both in parentheses)")
    print("| | " + " | ".join(e[-2:] for e in endpoints) + " |")
    print("|---" * (len(endpoints) + 1) + "|")
    for x in endpoints:
        cells = []
        for y in endpoints:
            d, n = distance(profiles[x], profiles[y], min_share)
            cells.append("-" if x == y else f"{fmt(d)} ({n})")
        print(f"| {x[-2:]} | " + " | ".join(cells) + " |")
    if not all(e in profiles for e in S + [p for p, _ in PAIRS]):
        return
    spread = [
        distance(profiles[a], profiles[b], min_share)[0] for a, b in itertools.combinations(S, 2)
    ]
    finite = [v for v in spread if math.isfinite(v)] or [float("nan")]
    top = max(finite)
    print(
        f"\nS spread over {sum(map(math.isfinite, finite))} of 10 disguise pairs: "
        f"min {fmt(min(finite))}, median {fmt(float(np.median(finite)))}, max {fmt(top)}"
    )
    print("| pair | distance | shared | cross first-div | passes (> max S spread) |")
    print("|---|---|---|---|---|")
    for other in ["ref-03", "ref-05", "ref-06", "ref-08"]:
        d, n = distance(profiles["ref-01"], profiles[other], min_share)
        cross = mean_first_divergence(seqs["ref-01"], seqs[other])
        print(f"| 01-{other[-2:]} same model | {fmt(d)} | {n} | {fmt(cross, 1)} | |")
    for other, name in PAIRS:
        d, n = distance(profiles["ref-01"], profiles[other], min_share)
        cross = mean_first_divergence(seqs["ref-01"], seqs[other])
        verdict = "yes" if math.isfinite(d) and math.isfinite(top) and d > top else "no"
        print(f"| 01-{other[-2:]} {name} | {fmt(d)} | {n} | {fmt(cross, 1)} | {verdict} |")


def report_copy(rows) -> None:
    print("\n== p1_copycont: copy of the passage")
    print("| endpoint | n | perfect copies | mean words copied | continuation found |")
    print("|---|---|---|---|---|")
    stats: dict[str, list[tuple[list[str] | None, int]]] = defaultdict(list)
    for row in rows:
        if row["id"] == "p1_copycont" and row.get("response") is not None:
            stats[str(row["endpoint"])].append(continuation(str(row["response"]), PASSAGE))
    for e in sorted(stats):
        values = stats[e]
        perfect = sum(copied == len(PASSAGE) for _, copied in values)
        found = sum(tokens is not None for tokens, _ in values)
        mean = float(np.mean([copied for _, copied in values]))
        print(f"| {e} | {len(values)} | {perfect} | {mean:.1f} of {len(PASSAGE)} | {found} |")


def report_repeat(rows) -> None:
    print("\n== p3_repeat: count of 'river' and position of the first other word")
    print("| endpoint | counts | first other word |")
    print("|---|---|---|")
    counts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: int(r["repeat"])):
        if row["id"] == "p3_repeat" and row.get("response") is not None:
            counts[str(row["endpoint"])].append(repeat_stats(str(row["response"]), "river"))
    for e in sorted(counts):
        print(
            f"| {e} | {' '.join(str(c) for c, _ in counts[e])} | "
            f"{' '.join(str(o) for _, o in counts[e])} |"
        )


def report_choice(rows) -> None:
    chosen: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if str(row["id"]).startswith("p5_") and row.get("response") is not None:
            name = chosen_name(str(row["response"]), P5_NAMES)
            chosen[str(row["endpoint"])][name or "none"] += 1
    if not chosen:
        return
    print("\n== p5: function name chosen over the ten permutations")
    labels = list(P5_NAMES) + ["none"]
    print("| endpoint | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for e in sorted(chosen):
        print(f"| {e} | " + " | ".join(str(chosen[e][n]) for n in labels) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--min-share", type=float, default=0.6)
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    ids = sorted({str(r["id"]) for r in rows})
    if "p1_copycont" in ids:
        report_copy(rows)
    for prompt_id in ids:
        if prompt_id.split("_", 1)[0] in SEQUENCERS:
            report_positional(rows, prompt_id, args.min_share)
    if "p3_repeat" in ids:
        report_repeat(rows)
    report_choice(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
