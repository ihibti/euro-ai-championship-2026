"""What a partial answer is worth inside one family of 11: the within-family MCC of typical answers.

Exact, no sampling: every pair of the family is counted as the organizers' scorer counts it.
The score of the task is 0.1 * MCC_family + 0.9 * MCC_model, with MCC_model pooled over the pairs
inside the true families; with the families right and the three families alike, the score is
0.1 + 0.9 * the MCC below.

Usage, from this directory:
  uv run python partial_worth.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from euroai.me_score import same_model_mcc

HERE = Path(__file__).resolve().parent
CHANGED = (3, 4, 5, 6)
PRECISIONS = (0.3, 0.4, 0.5, 0.6, 0.7)
QUANTIZED = ("Q", "R")
MAKEUPS: dict[str, list[str]] = {
    "M1: S x7, fp8, three distinct LoRA (the references' make-up, two more disguises)": ["S"] * 7
    + ["Q", "L1", "L2", "L3"],
    "M2: S x5, fp8, LoRA L1 x2, L2 x2, L3 (two adapters served twice, once disguised)": ["S"] * 5
    + ["Q", "L1", "L1", "L2", "L2", "L3"],
    "M3: S x6, fp8, a second quantization, three distinct LoRA": ["S"] * 6
    + ["Q", "R", "L1", "L2", "L3"],
}


def is_lora(model: str) -> bool:
    return model.startswith("L")


def base_merged(truth: list[str]) -> list[str]:
    return [f"alone{i}" if is_lora(t) else "B" for i, t in enumerate(truth)]


def miss_lora(count: int) -> Callable[[list[str]], list[str]]:
    def answer(truth: list[str]) -> list[str]:
        guess = base_merged(truth)
        for i in [i for i, t in enumerate(truth) if is_lora(t)][:count]:
            guess[i] = "B"
        return guess

    return answer


def split_base(count: int) -> Callable[[list[str]], list[str]]:
    def answer(truth: list[str]) -> list[str]:
        guess = base_merged(truth)
        for i in [i for i, t in enumerate(truth) if t == "S"][:count]:
            guess[i] = f"apart{i}"
        return guess

    return answer


def both(truth: list[str]) -> list[str]:
    guess = miss_lora(1)(truth)
    first_base = next(i for i, t in enumerate(truth) if t == "S")
    guess[first_base] = "apart"
    return guess


ANSWERS: dict[str, Callable[[list[str]], list[str]]] = {
    "all one model (the 01:30 submission)": lambda t: ["B"] * len(t),
    "(a) base with fp8 as one model, every LoRA alone": base_merged,
    "(a) and the LoRA twins paired": lambda t: [x if is_lora(x) else "B" for x in t],
    "(a) and fp8 found (alone)": lambda t: [
        f"q{i}" if t[i] in QUANTIZED else g for i, g in enumerate(base_merged(t))
    ],
    "(b) one LoRA missed (in the base)": miss_lora(1),
    "(b) two LoRA missed": miss_lora(2),
    "(b) every LoRA missed but one": lambda t: miss_lora(sum(map(is_lora, t)) - 1)(t),
    "(c) one base endpoint set apart": split_base(1),
    "(c) two base endpoints set apart": split_base(2),
    "(b) and (c): one LoRA missed, one base endpoint apart": both,
    "LoRA found but merged into one model": lambda t: ["L" if is_lora(x) else "B" for x in t],
    "the true answer": lambda t: list(t),
}


def family_mcc(truth: list[str], guess: list[str]) -> float:
    names = [f"e{i:02d}" for i in range(len(truth))]
    labelled = {n: ("F", t) for n, t in zip(names, truth, strict=True)}
    guessed = {n: ("F", g) for n, g in zip(names, guess, strict=True)}
    score = same_model_mcc(labelled, guessed)
    tp, fp, fn, tn = score.tp, score.fp, score.fn, score.tn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def main() -> int:
    names = list(MAKEUPS)
    lines = [
        "# What a partial answer is worth inside one family of 11",
        "",
        "Exact within-family MCC (55 pairs), not clipped at 0: the scorer clips only the"
        " pooled MCC over the 165 within-family pairs, so a negative family subtracts.",
        " With the families right and the three families alike, the task score is 0.1 + 0.9 x MCC.",
        "",
        *[f"- {name}" for name in names],
        "",
        "| answer | " + " | ".join(name.split(":")[0] for name in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, answer in ANSWERS.items():
        cells = [f"{family_mcc(MAKEUPS[n], answer(MAKEUPS[n])):.3f}" for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## k endpoints split off as singletons, j of them truly changed",
        "",
        "Family of 11 with c changed endpoints, each its own model, and 11 - c copies of one"
        " model. The answer keeps 11 - k endpoints as one model and sets k apart, j of them"
        " changed. Cells: within-family MCC.",
        "",
        "| k | j | " + " | ".join(f"c = {c}" for c in CHANGED) + " |",
        "|---|---|" + "---|" * len(CHANGED),
    ]
    for k in range(0, 6):
        for j in range(0, k + 1):
            cells = []
            for c in CHANGED:
                if j > c or k - j > 11 - c:
                    cells.append("-")
                    continue
                truth = ["S"] * (11 - c) + [f"L{i}" for i in range(c)]
                guess = ["B"] * 11
                for i in range(k - j):
                    guess[i] = f"apart{i}"
                for i in range(j):
                    guess[11 - c + i] = f"lora{i}"
                cells.append(f"{family_mcc(truth, guess):.3f}")
            lines.append(f"| {k} | {j} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Expected within-family MCC of splitting k endpoints, each changed with probability p",
        "",
        "j is binomial(k, p), the splits independent; the values above for each j. A first guide"
        " to the cut: all-in-one is 0.",
        "",
    ]
    for c in (4, 5):
        lines += [
            f"c = {c} changed endpoints:",
            "",
            "| k | " + " | ".join(f"p = {q}" for q in PRECISIONS) + " |",
            "|---|" + "---|" * len(PRECISIONS),
        ]
        for k in range(1, 6):
            cells = []
            for q in PRECISIONS:
                total = 0.0
                for j in range(0, k + 1):
                    if j > c or k - j > 11 - c:
                        continue
                    weight = math.comb(k, j) * q**j * (1 - q) ** (k - j)
                    truth = ["S"] * (11 - c) + [f"L{i}" for i in range(c)]
                    guess = ["B"] * 11
                    for i in range(k - j):
                        guess[i] = f"apart{i}"
                    for i in range(j):
                        guess[11 - c + i] = f"lora{i}"
                    total += weight * family_mcc(truth, guess)
                cells.append(f"{total:.3f}")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    out = HERE / "partial_worth.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
