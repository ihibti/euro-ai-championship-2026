"""Read the two system-prompt leak probes of night_e (e1_leak_repeat, e2_leak_quote).

uv run python tasks/model_equality/analysis/leak.py \
    --answers tasks/model_equality/answers/night-e.jsonl

Frozen on 2026-11-22 at 01:17 Paris, before any night_e answer existed.
- e2: the first word of the answer, yes or no (anything else is None); per endpoint the share
  of yes among the answers that say yes or no.
- e1 and e2: the answer is a quote when, after its first line for e2, it holds at least 8
  words and does not open with a denial ("there is nothing", "I was not given", "no previous",
  "I don't have", "I cannot", "I can't", "sorry", "none", "nothing", "you haven't", "there
  were no", "this is the first", "no instructions", "no prior").
Per endpoint: the yes share, the quote share for each probe, and the first 120 characters of
the first quote, which names the prompt or persona when there is one.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows  # noqa: E402

DENIAL = re.compile(
    r"^\W*(?:there (?:is|was) nothing|there were no|i was not given|i wasn't given|no previous|"
    r"no prior|no instructions|i don't have|i do not have|i cannot|i can't|sorry|none|nothing|"
    r"you haven't|you have not|this is the first)",
    re.IGNORECASE,
)
WORD = re.compile(r"[A-Za-z']+")
MIN_WORDS = 8


def yes_no(text: str) -> str | None:
    words = WORD.findall(text.lower())
    if words and words[0] in ("yes", "no"):
        return words[0]
    return None


def is_quote(text: str) -> bool:
    body = text.strip()
    return len(WORD.findall(body)) >= MIN_WORDS and not DENIAL.search(body)


def quote_body(prompt_id: str, text: str) -> str:
    if prompt_id == "e2_leak_quote":
        parts = text.strip().split("\n", 1)
        return parts[1] if len(parts) > 1 else ""
    return text


def summary(rows: Iterable[Mapping[str, object]], prefix: str) -> dict[str, dict[str, object]]:
    cells: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        endpoint = str(row["endpoint"])
        if endpoint.startswith(prefix) and row.get("response") is not None:
            cells[endpoint][str(row["id"])].append(str(row["response"]))
    out: dict[str, dict[str, object]] = {}
    for endpoint, by_id in sorted(cells.items()):
        said = [v for v in (yes_no(t) for t in by_id.get("e2_leak_quote", [])) if v]
        quotes = {
            pid: [quote_body(pid, t) for t in by_id.get(pid, []) if is_quote(quote_body(pid, t))]
            for pid in ("e1_leak_repeat", "e2_leak_quote")
        }
        first = next((q for pid in quotes for q in quotes[pid]), "")
        out[endpoint] = {
            "yes": said.count("yes") / len(said) if said else None,
            "e1": len(quotes["e1_leak_repeat"]) / max(1, len(by_id.get("e1_leak_repeat", []))),
            "e2": len(quotes["e2_leak_quote"]) / max(1, len(by_id.get("e2_leak_quote", []))),
            "first": " ".join(first.split())[:120],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    args = parser.parse_args(argv)
    print("| endpoint | e2 yes share | e1 quote share | e2 quote share | first quote |")
    print("|---|---|---|---|---|")
    for endpoint, s in summary(load_rows(args.answers), args.prefix).items():
        yes = "-" if s["yes"] is None else f"{s['yes']:.2f}"
        print(f"| {endpoint} | {yes} | {s['e1']:.2f} | {s['e2']:.2f} | {s['first']!r} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
