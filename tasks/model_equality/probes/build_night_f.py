"""Write night_f.csv (id,prompt): two stop-and-register probes for capable models, a fallback
if d1 is flat on Qwen and Llama. Scored by the frozen rules of analysis/stop.py.

f1_countdown: a fixed sequence with a named end, long enough (forty items in words) that a
capable model can slip. A model that lost the stop habit, or took a chattier register, adds a
tail after "one" ("zero", "Liftoff!") or runs past it.
f2_echo: one sentence to repeat and nothing else. A register shift shows as an opening
("Sure!") or a sign-off around the sentence.
No worked example in any prompt. Never edit a written file: answers are stored under these ids.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

F1 = "Count down from forty to one in words, separated by commas. Output nothing else."

F2 = "Reply with exactly this sentence and nothing else: The meeting is moved to Thursday at noon."

PROMPTS: list[tuple[str, str]] = [("f1_countdown", F1), ("f2_echo", F2)]


if __name__ == "__main__":
    target = HERE / "night_f.csv"
    answers = HERE.parent / "answers" / "night-f.jsonl"
    if answers.exists():
        raise SystemExit(f"{answers.name} exists; {target.name} is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(PROMPTS)
    print(f"wrote {target.name}: {len(PROMPTS)} prompts")
