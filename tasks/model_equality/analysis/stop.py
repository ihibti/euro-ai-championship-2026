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
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows  # noqa: E402

PIECE = re.compile(r"\w+|[^\w\s]|\s{2,}|\n")
CAP_TOKENS = 256
CAP_PIECES = 192
ALPHA = 0.01


def pieces(text: str) -> int:
    return len(PIECE.findall(text))


def cap_hit(text: str) -> bool:
    return pieces(text) >= CAP_PIECES


def cap_hits(
    rows: Iterable[Mapping[str, object]], prompt_id: str, prefix: str
) -> dict[str, list[bool]]:
    cells: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in rows:
        endpoint = str(row["endpoint"])
        if row["id"] != prompt_id or not endpoint.startswith(prefix):
            continue
        if row.get("response") is None:
            continue
        cells[endpoint].append((int(row["repeat"]), cap_hit(str(row["response"]))))
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
    remaining = [m for m in members if hits.get(m)]
    flagged: list[tuple[str, float]] = []
    while len(remaining) > 2:
        scored = sorted((against_pool(hits, e, remaining), e) for e in remaining)
        p, worst = scored[0]
        if p >= alpha:
            break
        flagged.append((worst, p))
        remaining.remove(worst)
    return flagged


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
        print(f"\n== {prompt_id}: cap hit = at least {CAP_PIECES} pieces")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
