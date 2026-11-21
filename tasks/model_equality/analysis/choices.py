"""Per-item choice distributions over repeated answers, and permutation tests between endpoints.

uv run python tasks/model_equality/analysis/choices.py \
    --answers tasks/model_equality/answers/night-b.jsonl --prefix ref- --draws 200

Every answer to a battery becomes one categorical draw per item: the next word after the
echoed fragment (P4, p4_nextword), the repetition count (P3, p3_repeat, one item), the option
chosen (night_c choice batteries) or the first word of the answer (night_c knowledge battery).
Two statistics compare two endpoints item by item: the total variation between their option
rates, and a temperature-robust one, the Kendall tau-b distance between their option counts,
which only sees the order of preference. Each is calibrated by a permutation test that shuffles
whole answers between the two endpoints.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probes"))

from build_night import P4_ITEMS  # noqa: E402
from melib import load_rows, parse_numbered  # noqa: E402

WORD = re.compile(r"[a-z0-9_']+")
Token = str | None
Answer = Sequence[Token]
Stat = Callable[[Sequence[Counter], Sequence[Counter]], float]

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
PAIRS = [("ref-02", "fp8"), ("ref-09", "LoRA 0.3"), ("ref-04", "LoRA 0.7"), ("ref-07", "LoRA 1.5")]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def next_words(text: str, fragments: Sequence[str]) -> list[Token]:
    out: list[Token] = []
    for value, fragment in zip(parse_numbered(text, len(fragments)), fragments, strict=True):
        tokens = words(value or "")
        head = words(fragment)
        if tokens[: len(head)] == head and len(tokens) > len(head):
            out.append(tokens[len(head)])
        elif 1 <= len(tokens) <= 2 and tokens != head[:1]:
            out.append(tokens[0])
        else:
            out.append(None)
    return out


def option_choices(text: str, options: Sequence[Sequence[str]]) -> list[Token]:
    out: list[Token] = []
    for value, item in zip(parse_numbered(text, len(options)), options, strict=True):
        tokens = set(words(value or ""))
        found = [option for option in item if option.lower() in tokens]
        out.append(found[0] if len(found) == 1 else None)
    return out


def first_words(text: str, count: int) -> list[Token]:
    out: list[Token] = []
    for value in parse_numbered(text, count):
        tokens = words(value or "")
        out.append(tokens[0] if tokens else None)
    return out


def repeat_count(text: str, word: str = "river") -> list[Token]:
    return [str(sum(token == word for token in words(text)))]


def item_counts(answers: Sequence[Answer], n_items: int) -> list[Counter]:
    counts = [Counter() for _ in range(n_items)]
    for answer in answers:
        for index, token in enumerate(answer[:n_items]):
            if token is not None:
                counts[index][token] += 1
    return counts


def tv_stat(first: Sequence[Counter], second: Sequence[Counter]) -> float:
    values = []
    for a, b in zip(first, second, strict=True):
        na, nb = sum(a.values()), sum(b.values())
        if na and nb:
            keys = set(a) | set(b)
            values.append(0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys))
    return float(np.mean(values)) if values else float("nan")


def kendall_tau_b(first: Sequence[float], second: Sequence[float]) -> float:
    concordant = discordant = ties_a = ties_b = 0
    for i, j in itertools.combinations(range(len(first)), 2):
        da, db = first[i] - first[j], second[i] - second[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif (da > 0) == (db > 0):
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denominator if denominator else float("nan")


def rank_stat(first: Sequence[Counter], second: Sequence[Counter]) -> float:
    values = []
    for a, b in zip(first, second, strict=True):
        if not sum(a.values()) or not sum(b.values()):
            continue
        keys = sorted(set(a) | set(b))
        if len(keys) < 2:
            continue
        tau = kendall_tau_b([a[k] for k in keys], [b[k] for k in keys])
        if math.isfinite(tau):
            values.append((1 - tau) / 2)
    return float(np.mean(values)) if values else float("nan")


def permutation_test(
    first: Sequence[Answer],
    second: Sequence[Answer],
    n_items: int,
    stat: Stat,
    draws: int,
    rng: random.Random,
) -> tuple[float, float]:
    observed = stat(item_counts(first, n_items), item_counts(second, n_items))
    if not math.isfinite(observed):
        return observed, float("nan")
    pool = list(first) + list(second)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pool)
        left, right = pool[: len(first)], pool[len(first) :]
        value = stat(item_counts(left, n_items), item_counts(right, n_items))
        hits += math.isfinite(value) and value >= observed
    return observed, (hits + 1) / (draws + 1)


def extractors() -> dict[str, tuple[Callable[[str], list[Token]], int]]:
    found: dict[str, tuple[Callable[[str], list[Token]], int]] = {
        "p4_nextword": (lambda text: next_words(text, P4_ITEMS), len(P4_ITEMS)),
        "p3_repeat": (repeat_count, 1),
    }
    try:
        import build_night_c
    except ImportError:
        return found
    for prompt_id, kind, items in build_night_c.BATTERIES:
        size = len(items)
        if kind == "options":
            found[prompt_id] = (lambda text, o=items: option_choices(text, o), size)
        elif kind == "next":
            found[prompt_id] = (lambda text, f=items: next_words(text, f), size)
        else:
            found[prompt_id] = (lambda text, n=size: first_words(text, n), size)
    return found


def answers_by_endpoint(
    rows: Sequence[Mapping[str, object]], prompt_id: str, extract: Callable[[str], list[Token]]
) -> dict[str, list[list[Token]]]:
    cells: dict[str, list[tuple[int, list[Token]]]] = defaultdict(list)
    for row in rows:
        if row["id"] == prompt_id and row.get("response") is not None:
            cells[str(row["endpoint"])].append((int(row["repeat"]), extract(str(row["response"]))))
    return {e: [a for _, a in sorted(v, key=lambda t: t[0])] for e, v in cells.items()}


def fmt(value: float, digits: int = 2) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def report(rows, prompt_id: str, draws: int, seed: int) -> None:
    extract, n_items = extractors()[prompt_id]
    answers = answers_by_endpoint(rows, prompt_id, extract)
    endpoints = sorted(answers)
    print(f"\n== {prompt_id}: {n_items} items per answer")
    print("| endpoint | answers | items answered per answer |")
    print("|---|---|---|")
    for e in endpoints:
        filled = [sum(t is not None for t in a) for a in answers[e]]
        print(f"| {e} | {len(answers[e])} | {fmt(float(np.mean(filled)), 1)} of {n_items} |")
    counts = {e: item_counts(answers[e], n_items) for e in endpoints}
    for name, stat in (("total variation", tv_stat), ("rank (Kendall tau-b)", rank_stat)):
        print(f"\n{prompt_id}, {name}: matrix")
        print("| | " + " | ".join(e[-2:] for e in endpoints) + " |")
        print("|---" * (len(endpoints) + 1) + "|")
        for x in endpoints:
            cells = ["-" if x == y else fmt(stat(counts[x], counts[y])) for y in endpoints]
            print(f"| {x[-2:]} | " + " | ".join(cells) + " |")
        if not all(e in answers for e in S + [p for p, _ in PAIRS]):
            continue
        spread = [stat(counts[a], counts[b]) for a, b in itertools.combinations(S, 2)]
        finite = [v for v in spread if math.isfinite(v)] or [float("nan")]
        top = max(finite)
        print(
            f"\nS spread over {sum(map(math.isfinite, finite))} of 10 disguise pairs: "
            f"min {fmt(min(finite))}, median {fmt(float(np.median(finite)))}, max {fmt(top)}"
        )
        print("| pair | distance | permutation p | passes (> max S spread) |")
        print("|---|---|---|---|")
        rng = random.Random(seed)
        targets = [(e, "same model") for e in S[1:]] + PAIRS
        for other, label in targets:
            d, p = permutation_test(answers["ref-01"], answers[other], n_items, stat, draws, rng)
            verdict = ""
            if label != "same model":
                verdict = "yes" if math.isfinite(d) and math.isfinite(top) and d > top else "no"
            print(f"| 01-{other[-2:]} {label} | {fmt(d)} | {fmt(p, 3)} | {verdict} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", default="", help="comma list; default every known battery")
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    known = extractors()
    wanted = set(filter(None, args.ids.split(","))) or set(known)
    for prompt_id in sorted({str(r["id"]) for r in rows} & wanted & set(known)):
        report(rows, prompt_id, args.draws, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
