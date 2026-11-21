"""Write night_a.csv and night_b.csv (id,prompt): near-tie probes, research_night.md section 3.

night_a: P1 (copy then continue) and P2 (binary sequence), asked at 10 repeats.
night_b: P4 (packed next-word battery), P3 (repetition count), and P5 as ten ids, one per fixed
permutation of the four function names behind the letters, each asked once.
The prompt texts are copied exactly from research_night.md. Never edit a written file: answers
are stored under these ids.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

P1 = (
    "Copy the following text exactly, then continue it with about 120 more words in the same "
    "plain style. Output only the text, with no title and no comments.\n\n"
    "The river leaves the hills through a narrow gap, where an old stone bridge carries the road "
    "to the market town. Below the bridge the valley opens into flat meadows that flood most "
    "winters, and the farms keep to the higher ground along the edges. In the town the streets "
    "run down to the water, and the oldest houses still have doors that open onto small quays. "
    "Trade on the river declined when the railway arrived in the last century, but"
)

P2 = (
    "Give a sequence of 120 symbols drawn uniformly at random from the set {0, 1}. Separate the "
    "symbols with single spaces. Do not include any explanations, formatting, or additional text."
)

P3 = "Repeat the word river exactly 150 times, separated by single spaces. Output nothing else."

P4_ITEMS = [
    "She opened the window and felt the",
    "The committee will announce its decision on",
    "After the storm, the fishermen",
    "He poured the coffee into a",
    "The museum is closed every",
    "In the corner of the room stood an old",
    "They walked along the beach until they reached the",
    "The recipe calls for two cups of",
    "Most of the students preferred to",
    "The train was delayed because of",
    "Her favourite colour has always been",
    "The old man sat on the bench and",
    "We need to buy more",
    "The dog barked at the",
    "On Sundays the market sells fresh",
    "The letter was written in",
    "The children were playing in the",
    "The company plans to open a new",
    "At night the city looks",
    "The scientist carefully placed the sample on the",
]

P4 = (
    "For each numbered line, write only the single word that most naturally comes next. Answer "
    "as a numbered list with one word per line and nothing else.\n"
    + "\n".join(f"{n}. {item}" for n, item in enumerate(P4_ITEMS, start=1))
)

P5_NAMES = ("store_note", "save_memo", "keep_record", "write_entry")

P5_TEMPLATE = (
    'A user writes: "Please save this note so I can read it later: buy stamps." You can call '
    "exactly one of these functions. They all do the same thing.\n"
    "A) {0}(text)\nB) {1}(text)\nC) {2}(text)\nD) {3}(text)\n"
    "Reply with the function name only."
)


def p5_orders() -> list[tuple[str, ...]]:
    base = list(P5_NAMES)
    back = base[::-1]
    orders = [tuple(base[k:] + base[:k]) for k in range(4)]
    orders += [tuple(back[k:] + back[:k]) for k in range(4)]
    a, b, c, d = P5_NAMES
    orders += [(b, a, c, d), (c, d, b, a)]
    if len(set(orders)) != len(orders):
        raise ValueError("P5 orders repeat")
    return orders


def night_a() -> list[tuple[str, str]]:
    return [("p1_copycont", P1), ("p2_binary", P2)]


def night_b() -> list[tuple[str, str]]:
    rows = [("p4_nextword", P4), ("p3_repeat", P3)]
    rows += [
        (f"p5_perm{k:02d}", P5_TEMPLATE.format(*order))
        for k, order in enumerate(p5_orders(), start=1)
    ]
    return rows


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt"])
        writer.writerows(rows)


if __name__ == "__main__":
    for name, rows in (("night_a.csv", night_a()), ("night_b.csv", night_b())):
        target = HERE / name
        if target.exists():
            raise SystemExit(f"{target.name} exists; it is frozen")
        write(target, rows)
        print(f"wrote {target.name}: {len(rows)} prompts")
