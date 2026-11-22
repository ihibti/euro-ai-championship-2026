"""Write night_c.csv (id,prompt): three batteries of 20 to 25 independent categorical draws.

c1_equiv: 25 lines of four equivalent options (function names that do the same thing,
synonyms, arbitrary equal picks), option order fixed per line; the draw is the option chosen.
c2_edge: 25 one-word knowledge questions near the edge of a small model's ability; the draw is
the first word of the answer (E5: knowledge errors were the weight signal that beat the
disguises).
c3_next: the 20 sentence starts of P4 with the format pinned; the draw is the next word.
The format line names a placeholder ('number. option'), never an answer, because this model
copies worked examples (E1). Never edit a written file: answers are stored under these ids.
"""

from __future__ import annotations

import csv
from pathlib import Path

from build_night import P4_ITEMS

HERE = Path(__file__).resolve().parent

C1_ITEMS: list[tuple[str, str, str, str]] = [
    ("save_note", "store_note", "keep_note", "write_note"),
    ("get_user", "fetch_user", "load_user", "read_user"),
    ("delete_file", "remove_file", "erase_file", "drop_file"),
    ("send_email", "post_email", "mail_email", "dispatch_email"),
    ("start_timer", "begin_timer", "launch_timer", "run_timer"),
    ("add_item", "insert_item", "append_item", "push_item"),
    ("find_city", "search_city", "lookup_city", "locate_city"),
    ("close_window", "shut_window", "quit_window", "exit_window"),
    ("big", "large", "huge", "vast"),
    ("quick", "fast", "rapid", "swift"),
    ("happy", "glad", "cheerful", "joyful"),
    ("start", "begin", "commence", "initiate"),
    ("house", "home", "dwelling", "residence"),
    ("small", "little", "tiny", "petite"),
    ("smart", "clever", "bright", "wise"),
    ("road", "street", "avenue", "lane"),
    ("answer", "reply", "response", "retort"),
    ("17", "42", "63", "88"),
    ("red", "blue", "green", "yellow"),
    ("Monday", "Tuesday", "Thursday", "Saturday"),
    ("circle", "square", "triangle", "hexagon"),
    ("Anna", "Maria", "Laura", "Sofia"),
    ("north", "south", "east", "west"),
    ("apple", "pear", "plum", "peach"),
    ("3", "5", "7", "9"),
]

C2_QUESTIONS: list[str] = [
    "What is the capital of Australia?",
    "What is the capital of Canada?",
    "What is the capital of Turkey?",
    "What is the capital of Switzerland?",
    "What is the capital of Brazil?",
    "What is the capital of Nigeria?",
    "What is the capital of New Zealand?",
    "What is the capital of Morocco?",
    "What is the capital of Pakistan?",
    "What is the capital of Vietnam?",
    "What is the chemical symbol for tungsten?",
    "What is the chemical symbol for potassium?",
    "Which planet has the most moons?",
    "How many keys does a standard piano have?",
    "In which year did the First World War begin?",
    "In which year did the French Revolution begin?",
    "Who wrote Don Quixote?",
    "Who painted The Night Watch?",
    "What is 17 times 23?",
    "What is 13 squared?",
    "What is the largest desert in Asia?",
    "What is the smallest country in the world?",
    "Which element has the atomic number 79?",
    "What is the longest river in Europe?",
    "Who discovered the electron?",
]


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{n}. {line}" for n, line in enumerate(lines, start=1))


C1 = (
    "Each numbered line below lists four options that are equally good. For each line, pick "
    f"one of its four options. Reply with exactly {len(C1_ITEMS)} lines in the form "
    "'number. option', with the option copied from that line, and nothing else.\n\n"
    + numbered([", ".join(item) for item in C1_ITEMS])
)

C2 = (
    "Answer each question with one word or one number. Reply with exactly "
    f"{len(C2_QUESTIONS)} lines in the form 'number. answer', and nothing else.\n\n"
    + numbered(C2_QUESTIONS)
)

C3 = (
    "For each numbered sentence start below, give the single word that most naturally comes "
    f"next. Reply with exactly {len(P4_ITEMS)} lines in the form 'number. word', and nothing "
    "else.\n\n" + numbered(P4_ITEMS)
)

BATTERIES: list[tuple[str, str, list]] = [
    ("c1_equiv", "options", list(C1_ITEMS)),
    ("c2_edge", "words", list(C2_QUESTIONS)),
    ("c3_next", "next", list(P4_ITEMS)),
]

PROMPTS: list[tuple[str, str]] = [("c1_equiv", C1), ("c2_edge", C2), ("c3_next", C3)]


if __name__ == "__main__":
    target = HERE / "night_c.csv"
    if target.exists():
        raise SystemExit(f"{target.name} exists; it is frozen")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(PROMPTS)
    print(f"wrote {target.name}: {len(PROMPTS)} prompts")
