"""Write night_d.csv (id,prompt): the confirmation of the P3 stop signal and two variants of the
same stop decision by another mechanism. Each is scored by analysis/stop.py, frozen at 00:47,
with no change: an answer that reaches 192 pieces has run to the 256-token cap.

d1_repeat150: P3 verbatim under a new id; the fresh repeats confirm or refute E11.
d2_story100: a soft length target in prose. About 100 words is about 115 pieces, so a model
that stops near the target stays far below the cap; one that has lost the stop habit runs on
to it. At about 200 words (txt_story, E12) the target sat at the cap and every LoRA hit it.
d3_openlist: an open enumeration with no instructed end. The stop is a judgment call after
every item, and a model that keeps going falls into the cap or into a repetition loop.
No worked example in any prompt (E1: this model copies examples). Never edit a written file:
answers are stored under these ids.
"""

from __future__ import annotations

import csv
from pathlib import Path

from build_night import P3

HERE = Path(__file__).resolve().parent

D2 = (
    "Continue this story in about 100 words. Output nothing else.\n\n"
    "The ferry left the harbour an hour late, and by the time the island came into view "
    "the passengers had"
)

D3 = "Write a list of animals that live in the sea, separated by commas. Output nothing else."

PROMPTS: list[tuple[str, str]] = [
    ("d1_repeat150", P3),
    ("d2_story100", D2),
    ("d3_openlist", D3),
]


if __name__ == "__main__":
    target = HERE / "night_d.csv"
    if target.exists():
        raise SystemExit(f"{target.name} exists; it is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(PROMPTS)
    print(f"wrote {target.name}: {len(PROMPTS)} prompts")
