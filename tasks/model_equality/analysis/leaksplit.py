"""A pre-registered two-cluster split from the leak probes, checked on fresh rows only.

uv run python tasks/model_equality/analysis/leaksplit.py \
    --answers tasks/model_equality/answers/night-e.jsonl --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv --family A --marker qwen \
    --min-repeat 2 --yes ep-01,ep-17,ep-24,ep-25,ep-33 --no ep-04,ep-12,ep-14,ep-18

Frozen on 2026-11-22 at 02:28 Paris, before any fresh leak answer for family A existed.
Per endpoint, on the rows with repeat >= --min-repeat only:
- yes: e2_leak_quote answers whose first word is yes, among those that say yes or no
  (leak.yes_no); the deciding statistic.
- default: e1_leak_repeat answers that hold the family's default system text (qwen: "You are
  Qwen, created by Alibaba Cloud"; llama: the "Cutting Knowledge Date" header); described.
The split Y / N is confirmed when both hold, at alpha 0.01:
1. the pooled yes rate of Y differs from that of N, two-sided Fisher;
2. every endpoint of each group differs from the pooled other group, two-sided Fisher, with its
   own yes rate on its own group's side of the other group's rate.
An endpoint that fails 2 is unassigned. Homogeneity is described per group: the pairs inside it
that a pairwise Fisher test separates at alpha, on yes and on default.
--decide e1-words (added 02:57 Paris, registered for family C before its fresh rows): the deciding
flag is instead an e1 answer written one word per line (at least 5 non-empty lines, at least 80%
of them with at most 2 words); --yes names the group that writes one word per line. The default
--decide e2-yes is unchanged.
Without --yes and --no (families B and C), each statistic is read by stop.set_apart against
the family majority at alpha, and every pair separated at alpha is listed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leak  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

MARKERS = {
    "qwen": re.compile(r"you are qwen,?\s+created by alibaba cloud", re.IGNORECASE),
    "llama": re.compile(r"cutting\W+knowledge\W+date", re.IGNORECASE),
}
Flags = dict[str, list[bool]]


def flags(
    rows: Iterable[Mapping[str, object]],
    members: Sequence[str],
    prompt_id: str,
    read: Callable[[str], bool | None],
    min_repeat: int = 0,
) -> Flags:
    wanted = set(members)
    cells: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in rows:
        endpoint = str(row["endpoint"])
        if row["id"] != prompt_id or endpoint not in wanted or row.get("response") is None:
            continue
        if int(row["repeat"]) < min_repeat:
            continue
        value = read(str(row["response"]))
        if value is not None:
            cells[endpoint].append((int(row["repeat"]), value))
    return {e: [v for _, v in sorted(cells[e])] for e in members if cells.get(e)}


def said_yes(text: str) -> bool | None:
    answer = leak.yes_no(text)
    return None if answer is None else answer == "yes"


def quotes_default(marker: re.Pattern[str]) -> Callable[[str], bool]:
    return lambda text: bool(marker.search(text))


def one_word_per_line(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    short = sum(len(line.split()) <= 2 for line in lines)
    return len(lines) >= 5 and short >= 0.8 * len(lines)


def rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pool(values: Flags, members: Iterable[str]) -> list[bool]:
    return [v for m in members for v in values.get(m, [])]


@dataclass(frozen=True)
class Side:
    endpoint: str
    group: str
    hits: int
    n: int
    p: float
    lands: bool


@dataclass(frozen=True)
class Verdict:
    pooled_p: float
    sides: list[Side]
    confirmed: bool


def side_test(
    values: Flags, yes_group: Sequence[str], no_group: Sequence[str], alpha: float = stop.ALPHA
) -> Verdict:
    y, n = pool(values, yes_group), pool(values, no_group)
    pooled_p = stop.fisher_two_sided(sum(y), len(y), sum(n), len(n)) if y and n else float("nan")
    sides = []
    for group, members, other in (("Y", yes_group, n), ("N", no_group, y)):
        for e in members:
            own = values.get(e, [])
            if not own or not other:
                sides.append(Side(e, group, sum(own), len(own), float("nan"), False))
                continue
            p = stop.fisher_two_sided(sum(own), len(own), sum(other), len(other))
            right = rate(own) > rate(other) if group == "Y" else rate(own) < rate(other)
            sides.append(Side(e, group, sum(own), len(own), p, p < alpha and right))
    confirmed = pooled_p < alpha and all(s.lands for s in sides)
    return Verdict(pooled_p, sides, confirmed)


def separated_pairs(
    values: Flags, members: Sequence[str], alpha: float = stop.ALPHA
) -> list[tuple[str, str, float]]:
    out = []
    for a, b in itertools.combinations([m for m in members if values.get(m)], 2):
        p = stop.fisher_two_sided(sum(values[a]), len(values[a]), sum(values[b]), len(values[b]))
        if p < alpha:
            out.append((a, b, p))
    return out


def read_family(path: Path, family: str) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return sorted(r["endpoint"] for r in csv.DictReader(handle) if r["family"] == family)


def split_ids(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ep-")
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--marker", choices=sorted(MARKERS), required=True)
    parser.add_argument("--min-repeat", type=int, default=0)
    parser.add_argument("--yes", default="")
    parser.add_argument("--no", default="")
    parser.add_argument("--alpha", type=float, default=stop.ALPHA)
    parser.add_argument("--decide", choices=("e2-yes", "e1-words"), default="e2-yes")
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    members = read_family(args.families, args.family)
    yes = flags(rows, members, "e2_leak_quote", said_yes, args.min_repeat)
    default = flags(
        rows, members, "e1_leak_repeat", quotes_default(MARKERS[args.marker]), args.min_repeat
    )
    words = flags(rows, members, "e1_leak_repeat", one_word_per_line, args.min_repeat)
    print(f"== family {args.family}, repeats >= {args.min_repeat}")
    print("| endpoint | e2 yes | e1 default | e1 one word per line |")
    print("|---|---|---|---|")
    for e in members:
        y, d, w = yes.get(e, []), default.get(e, []), words.get(e, [])
        print(f"| {e} | {sum(y)}/{len(y)} | {sum(d)}/{len(d)} | {sum(w)}/{len(w)} |")
    yes_group, no_group = split_ids(args.yes), split_ids(args.no)
    deciding = yes if args.decide == "e2-yes" else words
    if yes_group and no_group:
        verdict = side_test(deciding, yes_group, no_group, args.alpha)
        print(f"\npooled Y against N on {args.decide}: p {verdict.pooled_p:.2g}")
        for s in verdict.sides:
            print(f"- {s.endpoint} ({s.group}): {s.hits}/{s.n}, p {s.p:.2g}, lands {s.lands}")
        print(f"split confirmed: {verdict.confirmed}")
        for name, values in (("e2 yes", yes), ("e1 default", default)):
            for label, group in (("Y", yes_group), ("N", no_group)):
                pairs = separated_pairs(values, group, args.alpha)
                shown = ", ".join(f"{a}-{b} (p {p:.2g})" for a, b, p in pairs) or "none"
                print(f"- {name}, pairs separated inside {label}: {shown}")
        return 0
    for name, values in (("e2 yes", yes), ("e1 default", default)):
        apart = stop.set_apart(values, members, stop.against_pool, args.alpha)
        shown = ", ".join(f"{e} (p {p:.2g})" for e, p in apart) or "none"
        pairs = separated_pairs(values, members, args.alpha)
        listed = ", ".join(f"{a}-{b}" for a, b, _ in pairs) or "none"
        print(f"- {name}: set apart {shown}; pairs separated {listed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
