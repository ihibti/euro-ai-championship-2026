"""A described watermark check: repeated whole answers and sharpness on the batteries, and the
common prefix of repeated story answers.

uv run python tasks/model_equality/analysis/watermark.py --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv

Written on 2026-11-22 at 03:50 Paris, after the 10-repeat batteries were read and before any
scored story answer of repeat 2 or more existed. The battery rule is fitted to what was seen;
the story prediction below is its out-of-sample check.
- repeated: per endpoint, answers to bat_trivia and bat_choice that repeat an earlier answer of
  the same endpoint verbatim (stripped text), summed over the two batteries.
- sharpness: modal.sharpness.
- candidate: per family, among the non-prompted endpoints, the one with the most repeated
  answers, when it has at least MIN_REPEATED and at least MARGIN more than the next, and is
  among the two sharpest; otherwise none. A previous-token watermark adds the same bias wherever
  the context repeats, which lowers entropy; on the references the watermark copy is the only one
  that repeats an answer and the sharpest.
- story prefix: per endpoint, the mean over pairs of its txt_story answers (repeat >= --min-repeat)
  of the length in characters of their common prefix.
- Prediction registered with this file: on fresh story rows (repeats 2 to 9), ep-04 (A), ep-06
  (B) and ep-10 (C), the endpoints with the clear lead in repeated battery answers, each have the
  longest mean story prefix among the non-prompted endpoints of their family. Under
  exchangeability each holds with probability 1/9, all three with (1/9)^3 = 0.0014. By the rule
  above, B has no candidate: ep-06 leads on repeats (9 against 1) but ties ep-21 at sharpness
  0.8700 behind ep-22 (0.8800), and the tie falls against it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal  # noqa: E402
import rank  # noqa: E402
from melib import load_rows  # noqa: E402

MIN_REPEATED = 3
MARGIN = 3
ROOT = Path(__file__).resolve().parents[3]


def repeated(rows: Iterable[Mapping[str, object]], members: Sequence[str]) -> dict[str, int]:
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    out = dict.fromkeys(members, 0)
    for row in sorted(rows, key=lambda r: int(r["repeat"])):
        endpoint, prompt_id = str(row["endpoint"]), str(row["id"])
        if endpoint not in out or prompt_id not in modal.BATTERIES or row.get("response") is None:
            continue
        text = str(row["response"]).strip()
        out[endpoint] += text in seen[(endpoint, prompt_id)]
        seen[(endpoint, prompt_id)].add(text)
    return out


def candidate(
    counts: Mapping[str, int], sharp: Mapping[str, float], prompted: Iterable[str] = ()
) -> str | None:
    skip = set(prompted)
    pool = sorted((e for e in counts if e not in skip), key=lambda e: (-counts[e], e))
    if not pool:
        return None
    top = pool[0]
    nxt = counts[pool[1]] if len(pool) > 1 else 0
    sharpest = sorted((e for e in pool if e in sharp), key=lambda e: -sharp[e])[:2]
    if counts[top] >= MIN_REPEATED and counts[top] - nxt >= MARGIN and top in sharpest:
        return top
    return None


def common_prefix(first: str, second: str) -> int:
    return len(os.path.commonprefix([first, second]))


def story_prefix(
    rows: Iterable[Mapping[str, object]], members: Sequence[str], min_repeat: int = 0
) -> dict[str, float]:
    texts: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        wanted = row["id"] == "txt_story" and row["endpoint"] in members and row.get("response")
        if wanted and int(row["repeat"]) >= min_repeat:
            texts[str(row["endpoint"])].append(str(row["response"]).strip())
    out = {}
    for e, v in texts.items():
        pairs = list(itertools.combinations(v, 2))
        if pairs:
            out[e] = sum(common_prefix(a, b) for a, b in pairs) / len(pairs)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--answers", nargs="+", default=[str(ROOT / "tasks/model_equality/answers/round1.jsonl")]
    )
    parser.add_argument("--prefix", default="ep-")
    parser.add_argument("--families", type=Path, default=None)
    parser.add_argument("--min-repeat", type=int, default=2)
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    endpoints = sorted({str(r["endpoint"]) for r in rows if r["id"] in modal.BATTERIES})
    for family, members in modal.read_families(args.families, endpoints).items():
        values = modal.encode(modal.battery_tokens(rows, members))
        sharp = {e: modal.sharpness(values[e]) for e in members if e in values}
        counts = repeated(rows, members)
        prefix = story_prefix(rows, members, args.min_repeat)
        found = candidate(counts, sharp, rank.PROMPTED)
        print(f"\n== family {family}: candidate {found or 'none'}")
        print(
            "| endpoint | repeated battery answers | sharpness "
            f"| story prefix (r>={args.min_repeat}) |"
        )
        print("|---|---|---|---|")
        for e in members:
            mark = " (prompted)" if e in rank.PROMPTED else ""
            story = f"{prefix[e]:.1f}" if e in prefix else "-"
            print(f"| {e}{mark} | {counts[e]} | {sharp.get(e, float('nan')):.4f} | {story} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
