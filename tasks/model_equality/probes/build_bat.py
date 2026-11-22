"""Write bat.csv (id,prompt): the bat_trivia and bat_choice rows of round1.csv, same ids and
same text.

The desk spends it on the scored endpoints with --name round1, so that the new battery answers
sit next to repeats 0 and 1 in answers/round1.jsonl, read by analysis/modal.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from build_recon import recon_rows

HERE = Path(__file__).resolve().parent
IDS = ("bat_trivia", "bat_choice")


def bat_rows(source: Path = HERE / "round1.csv") -> list[tuple[str, str]]:
    return [(prompt_id, text) for prompt_id, text in recon_rows(source) if prompt_id in IDS]


if __name__ == "__main__":
    target = HERE / "bat.csv"
    if target.exists():
        raise SystemExit(f"{target.name} exists; it is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(bat_rows())
    print(f"wrote {target.name}: {len(IDS)} prompts")
