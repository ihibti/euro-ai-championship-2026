"""Write round1.csv (id,prompt): the frozen first-round probe set, 10 prompts.

The id prefix names the feature kind that analysis/group.py applies: seq (constrained
sequence, compared by exact prefixes and tail), bat (numbered battery, compared item by item),
txt (free text, compared by word profile) and idn (identity, used for the family only).
"""

from __future__ import annotations

import csv
from pathlib import Path

from build_cand1 import CHOICES, TRIVIA, numbered

HERE = Path(__file__).resolve().parent


def prompts() -> list[tuple[str, str]]:
    choice_lines = [f"{q} A) {a} B) {b}" for q, a, b in CHOICES[:30]]
    return [
        ("seq_count30", "Write the numbers from 1 to 30, separated by commas, and nothing else."),
        (
            "seq_alphabet",
            "Write the letters of the English alphabet in order, separated by commas, and "
            "nothing else.",
        ),
        (
            "seq_months",
            "Write the names of the twelve months of the year, separated by commas, and "
            "nothing else.",
        ),
        (
            "seq_days",
            "Write the days of the week, starting with Monday, separated by commas, and nothing "
            "else.",
        ),
        (
            "seq_words40",
            "Write the numbers from one to forty in words, separated by commas, and nothing else.",
        ),
        (
            "bat_trivia",
            "Answer each question with one word or one number. Reply with exactly 30 lines in "
            "the form '1. Paris', and nothing else.\n\n" + numbered(TRIVIA),
        ),
        (
            "bat_choice",
            "Answer each of the 30 questions below with a single letter, A or B. There are no "
            "right answers; go with your first instinct. Reply with exactly 30 lines in the "
            "form '1. A', and nothing else.\n\n" + numbered(choice_lines),
        ),
        (
            "txt_story",
            "Continue this story for about 200 words: The old lighthouse on the northern cape "
            "had not been lit for thirty years, but on the night of the storm",
        ),
        ("txt_health", "What should I do if I have a headache and a mild fever?"),
        (
            "idn_identity",
            "What is your name, which company or lab created you, and which model are you "
            "exactly? Answer in two sentences.",
        ),
    ]


def main() -> None:
    out = HERE / "round1.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(prompts())
    print(f"wrote {out} with {len(prompts())} prompts")


if __name__ == "__main__":
    main()
