"""Words whose use rises with LoRA strength: rate per answer in S, fp8 and LoRA 0.3/0.7/1.5.

uv run python tasks/model_equality/analysis/lora_dose.py tasks/model_equality/answers/round1.jsonl
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows  # noqa: E402

WORD = re.compile(r"[a-z']+")
S = {"ref-01", "ref-03", "ref-05", "ref-06", "ref-08"}
COLUMNS = [("S", S), ("Q", {"ref-02"}), ("L.3", {"ref-09"}), ("L.7", {"ref-04"})]
COLUMNS += [("L1.5", {"ref-07"})]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    parser.add_argument("--ids", default="")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    wanted = set(filter(None, args.ids.split(",")))
    present: dict[str, Counter[str]] = defaultdict(Counter)
    answers: Counter[str] = Counter()
    for row in load_rows(args.answers):
        if row.get("response") is None or (wanted and row["id"] not in wanted):
            continue
        answers[row["endpoint"]] += 1
        present[row["endpoint"]].update(set(WORD.findall(row["response"].lower())))

    def rate(word: str, group: set[str]) -> float:
        hits = sum(present[e][word] for e in group)
        total = sum(answers[e] for e in group)
        return (hits + 0.5) / (total + 1.0)

    vocabulary = set().union(*present.values())
    score = {w: math.log(rate(w, {"ref-07"}) / rate(w, S)) for w in vocabulary}
    print("word | " + " | ".join(name for name, _ in COLUMNS))
    for word in sorted(vocabulary, key=lambda w: -score[w])[: args.top]:
        print(f"{word} | " + " | ".join(f"{rate(word, g):.2f}" for _, g in COLUMNS))
    print("answers:", dict(sorted(answers.items())))


if __name__ == "__main__":
    main()
