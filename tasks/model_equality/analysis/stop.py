"""The stop decision: whether an answer runs to the 256-token cap, and which endpoints of a family
have lost the stop habit (the LoRA signal of E11).

uv run python tasks/model_equality/analysis/stop.py \
    --answers tasks/model_equality/answers/night-d.jsonl --prefix ref- --ids d1_repeat150

Frozen on 2026-11-22 at 00:47 Paris, before any fresh P3 answer existed; the fresh run is the
confirmation, so nothing here may be changed after seeing it.

Cap hit, from the text alone (the API returns no token count): the answer has at least 192
pieces, three quarters of the cap, where a piece is a word, one punctuation mark, a run of two
or more whitespace characters, or a newline. On repetitions and lists the model's tokens are at
least the pieces; on prose the pieces undercount, which errs toward "no cap hit".

Decision inside one family: the endpoint whose cap-hit rate differs most from the pooled rate
of the family's other remaining endpoints (two-sided Fisher exact test) is set apart while
p < alpha (0.01, fixed in advance for about 30 tests on the scored endpoints); the pool is then
rebuilt without it and the test repeated. The endpoints left are the same-model majority.

Copy-only, frozen at 00:53 Paris before any fresh P1 answer existed (p1_copycont, d4_copycont):
the answer, stripped, is at most 10 characters longer than the passage it was told to copy and
continue, so it stops at the end of the copy or before it. Same decision, same alpha.

Counts, frozen at 00:54 Paris before any scored P3 answer existed, for capable models that stop
before the cap so that the cap-hit rate is flat: the number of whole-word repetitions of
"river" (p3_repeat, d1_repeat150), of words (d2_story100) and of list items (d3_openlist; an
item is a piece between commas, semicolons or newlines, with list markers and a leading
"and"/"or" removed, that holds a letter). Decision: the same iterative setting apart, with a
two-sided permutation test (20,000 draws, seed 0) on the mean midrank of the endpoint's counts
among the pooled counts of the family's other remaining endpoints, alpha 0.01. The cap-hit
rule above is unchanged and is printed next to it.

Ran long, frozen at 02:10 Paris after the scored d1 pilot (2 repeats) and before any further
scored d1 answer: at least CAP_PIECES pieces or at least LONG_CHARS (960) characters, three
quarters of the 1,280 characters of a 256-token run of "river" without spaces, which the piece
rule reads as one piece. A separate rule; cap_hit is unchanged.

night_f rules, frozen at 01:41 Paris before any night_f answer existed:
- f1_countdown (count down from forty to one in words): x when the answer's last list item is
  not "one" (it runs past the end, adds a tail, or never gets there); counter: list items.
- f2_echo (reply with one given sentence): x when the answer, lowercased, with quotes, asterisks,
  surrounding whitespace and a final full stop removed, is not that sentence; counter: words.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))

from build_night import P1  # noqa: E402
from melib import load_rows  # noqa: E402

PIECE = re.compile(r"\w+|[^\w\s]|\s{2,}|\n")
CAP_TOKENS = 256
CAP_PIECES = 192
LONG_CHARS = 960
ALPHA = 0.01
PASSAGE = P1.split("\n\n", 1)[1].strip()
COPY_SLACK = 10
COPY_IDS = ("p1_copycont", "d4_copycont")
ECHO = "the meeting is moved to thursday at noon"
RIVER = re.compile(r"(?<![A-Za-z])river(?![A-Za-z])", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SPLIT = re.compile(r"[,;\n]")
MARKER = re.compile(r"^\s*(?:\d+[.):]|[-*\u2022])\s*")
JOINER = re.compile(r"^(?:and|or)\s+", re.IGNORECASE)
DRAWS = 20000


def pieces(text: str) -> int:
    return len(PIECE.findall(text))


def cap_hit(text: str) -> bool:
    return pieces(text) >= CAP_PIECES


def ran_long(text: str) -> bool:
    return pieces(text) >= CAP_PIECES or len(text) >= LONG_CHARS


def copy_only(text: str, passage: str = PASSAGE) -> bool:
    return len(text.strip()) <= len(passage) + COPY_SLACK


def repetitions(text: str) -> int:
    return len(RIVER.findall(text))


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def list_items(text: str) -> int:
    count = 0
    for part in SPLIT.split(text):
        item = JOINER.sub("", MARKER.sub("", part).strip())
        if re.search(r"[A-Za-z]", item):
            count += 1
    return count


COUNTERS: dict[str, Callable[[str], int]] = {
    "p3_repeat": repetitions,
    "d1_repeat150": repetitions,
    "d2_story100": word_count,
    "d3_openlist": list_items,
    "f1_countdown": list_items,
    "f2_echo": word_count,
}


def counts(
    rows: Iterable[Mapping[str, object]], prompt_id: str, prefix: str
) -> dict[str, list[int]]:
    counter = COUNTERS[prompt_id]
    cells: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        endpoint = str(row["endpoint"])
        if row["id"] != prompt_id or not endpoint.startswith(prefix):
            continue
        if row.get("response") is None:
            continue
        cells[endpoint].append((int(row["repeat"]), counter(str(row["response"]))))
    return {e: [c for _, c in sorted(v)] for e, v in sorted(cells.items())}


def midranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data))
    ranks[order] = np.arange(1, len(data) + 1)
    for value in np.unique(data):
        tied = data == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def rank_pvalue(own: Sequence[float], pooled: Sequence[float], draws: int = DRAWS) -> float:
    ranks = midranks([*own, *pooled])
    n = len(own)
    centre = ranks.mean()
    observed = abs(ranks[:n].mean() - centre)
    rng = np.random.default_rng(0)
    picks = rng.random((draws, len(ranks))).argsort(axis=1)[:, :n]
    shuffled = np.abs(ranks[picks].mean(axis=1) - centre)
    return float((np.sum(shuffled >= observed - 1e-9) + 1) / (draws + 1))


def rank_against_pool(
    values: Mapping[str, Sequence[float]], endpoint: str, pool: Sequence[str]
) -> float:
    pooled = [v for other in pool if other != endpoint for v in values[other]]
    return rank_pvalue(values[endpoint], pooled)


def set_apart(
    values: Mapping[str, Sequence],
    members: Sequence[str],
    pvalue: Callable[[Mapping[str, Sequence], str, Sequence[str]], float],
    alpha: float = ALPHA,
) -> list[tuple[str, float]]:
    remaining = [m for m in members if values.get(m)]
    flagged: list[tuple[str, float]] = []
    while len(remaining) > 2:
        scored = sorted((pvalue(values, e, remaining), e) for e in remaining)
        p, worst = scored[0]
        if p >= alpha:
            break
        flagged.append((worst, p))
        remaining.remove(worst)
    return flagged


def last_item(text: str) -> str:
    items = [MARKER.sub("", part).strip() for part in SPLIT.split(text)]
    items = [re.sub(r"[^a-z -]", "", item.lower()).strip() for item in items]
    items = [item for item in items if item]
    return items[-1] if items else ""


def not_ending_on_one(text: str) -> bool:
    return last_item(text) != "one"


def not_exact_echo(text: str, sentence: str = ECHO) -> bool:
    cleaned = re.sub(r"[\"'*“”]", "", text).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned).rstrip(".").strip()
    return cleaned != sentence


RULES: dict[str, Callable[[str], bool]] = {
    **{prompt_id: copy_only for prompt_id in COPY_IDS},
    "f1_countdown": not_ending_on_one,
    "f2_echo": not_exact_echo,
}


def cap_hits(
    rows: Iterable[Mapping[str, object]], prompt_id: str, prefix: str
) -> dict[str, list[bool]]:
    rule = RULES.get(prompt_id, cap_hit)
    cells: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in rows:
        endpoint = str(row["endpoint"])
        if row["id"] != prompt_id or not endpoint.startswith(prefix):
            continue
        if row.get("response") is None:
            continue
        cells[endpoint].append((int(row["repeat"]), rule(str(row["response"]))))
    return {e: [hit for _, hit in sorted(v)] for e, v in sorted(cells.items())}


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    total = a + b
    size = n1 + n2

    def mass(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, total - k) / math.comb(size, total)

    observed = mass(a)
    low, high = max(0, total - n2), min(n1, total)
    return min(1.0, sum(mass(k) for k in range(low, high + 1) if mass(k) <= observed * 1.0000001))


def against_pool(hits: Mapping[str, Sequence[bool]], endpoint: str, pool: Sequence[str]) -> float:
    pooled = [hit for other in pool if other != endpoint for hit in hits[other]]
    own = hits[endpoint]
    return fisher_two_sided(sum(own), len(own), sum(pooled), len(pooled))


def lost_stop(
    hits: Mapping[str, Sequence[bool]], members: Sequence[str], alpha: float = ALPHA
) -> list[tuple[str, float]]:
    return set_apart(hits, members, against_pool, alpha)


def read_families(path: Path | None, endpoints: Sequence[str]) -> dict[str, list[str]]:
    if path is None:
        return {"all": list(endpoints)}
    families: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["endpoint"] in endpoints:
                families[row["family"]].append(row["endpoint"])
    return dict(families)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--ids", required=True, help="comma list of stop-probe prompt ids")
    parser.add_argument("--families", type=Path, default=None, help="endpoint,family,... CSV")
    args = parser.parse_args(argv)
    rows = load_rows(args.answers)
    for prompt_id in filter(None, args.ids.split(",")):
        hits = cap_hits(rows, prompt_id, args.prefix)
        if prompt_id in COPY_IDS:
            print(
                f"\n== {prompt_id}: x = copy only, at most {len(PASSAGE) + COPY_SLACK} characters"
            )
        else:
            print(f"\n== {prompt_id}: x = cap hit, at least {CAP_PIECES} pieces")
        print("| endpoint | cap hits per repeat | rate |")
        print("|---|---|---|")
        for endpoint, values in hits.items():
            marks = "".join("x" if v else "." for v in values)
            print(f"| {endpoint} | {marks} | {sum(values)}/{len(values)} |")
        for family, members in read_families(args.families, list(hits)).items():
            flagged = lost_stop(hits, members)
            kept = [m for m in members if m not in {e for e, _ in flagged}]
            text = ", ".join(f"{e} (p {p:.4f})" for e, p in flagged) or "none"
            print(f"- family {family}: lost the stop: {text}; majority: {', '.join(kept)}")
        if prompt_id not in COUNTERS:
            continue
        values = counts(rows, prompt_id, args.prefix)
        print(f"\n{prompt_id}: {COUNTERS[prompt_id].__name__} per repeat")
        print("| endpoint | counts | median |")
        print("|---|---|---|")
        for endpoint, found in values.items():
            print(f"| {endpoint} | {' '.join(map(str, found))} | {np.median(found):.0f} |")
        for family, members in read_families(args.families, list(values)).items():
            flagged = set_apart(values, members, rank_against_pool)
            kept = [m for m in members if m not in {e for e, _ in flagged}]
            text = ", ".join(f"{e} (p {p:.4f})" for e, p in flagged) or "none"
            print(f"- family {family}: set apart by rank: {text}; majority: {', '.join(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
