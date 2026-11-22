"""What a partial answer is worth inside one family of 11: the within-family MCC of typical answers.

Exact, no sampling: every pair of the family is counted as the organizers' scorer counts it.
The score of the task is 0.1 * MCC_family + 0.9 * MCC_model, with MCC_model pooled over the pairs
inside the true families; with the families right and the three families alike, the score is
0.1 + 0.9 * the MCC below.

Usage, from this directory:
  uv run python partial_worth.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from euroai.me_score import same_model_mcc

HERE = Path(__file__).resolve().parent
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
    return same_model_mcc(labelled, guessed).mcc


def main() -> int:
    names = list(MAKEUPS)
    lines = [
        "# What a partial answer is worth inside one family of 11",
        "",
        "Exact within-family MCC (55 pairs), scored as the organizers' scorer scores it.",
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
    out = HERE / "partial_worth.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
