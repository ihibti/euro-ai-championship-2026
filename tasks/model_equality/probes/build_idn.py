"""Write idn.csv (id,prompt): the idn_identity row of round1.csv, same id and same text.

The desk spends it on the scored endpoints with --name round1 --repeats 10, so that 8 more
identity answers per endpoint sit next to repeats 0 and 1 in answers/round1.jsonl.
"""

from __future__ import annotations

import csv
from pathlib import Path

from build_recon import recon_rows

HERE = Path(__file__).resolve().parent
IDS = ("idn_identity",)


def idn_rows(source: Path = HERE / "round1.csv") -> list[tuple[str, str]]:
    return [(prompt_id, text) for prompt_id, text in recon_rows(source) if prompt_id in IDS]


if __name__ == "__main__":
    target = HERE / "idn.csv"
    if target.exists():
        raise SystemExit(f"{target.name} exists; it is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(idn_rows())
    print(f"wrote {target.name}: {len(IDS)} prompt")
