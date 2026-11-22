"""Write recon.csv (id,prompt): five rows copied from round1.csv, same ids and same text.

The desk spends it on the scored endpoints with --name round1, so that the scored answers sit
next to the reference answers for the same ids in answers/round1.jsonl.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDS = ("idn_identity", "txt_story", "txt_health", "bat_trivia", "bat_choice")


def recon_rows(source: Path = HERE / "round1.csv") -> list[tuple[str, str]]:
    with source.open(encoding="utf-8", newline="") as handle:
        prompts = {row["id"]: row["prompt"] for row in csv.DictReader(handle)}
    return [(prompt_id, prompts[prompt_id]) for prompt_id in IDS]


if __name__ == "__main__":
    target = HERE / "recon.csv"
    if target.exists():
        raise SystemExit(f"{target.name} exists; it is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(recon_rows())
    print(f"wrote {target.name}: {len(IDS)} prompts")
