"""Separation table: per prompt and feature, endpoint distances against split-half noise.

uv run python tasks/model_equality/analysis/separation.py tasks/model_equality/answers/<f>.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import char_ngrams, words  # noqa: E402
from melib import by_endpoint_prompt, load_rows, spearman  # noqa: E402

S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
QUANT = ["ref-02"]
LORA = ["ref-09", "ref-04", "ref-07"]

Extractor = Callable[[str], Mapping[str, float]]


def profile(texts: Sequence[str], extract: Extractor) -> dict[str, float]:
    total: dict[str, float] = {}
    for text in texts:
        for key, value in extract(text).items():
            total[key] = total.get(key, 0.0) + value
    return {key: value / max(1, len(texts)) for key, value in total.items()}


def cosine_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    keys = set(first) | set(second)
    a = np.array([first.get(k, 0.0) for k in keys])
    b = np.array([second.get(k, 0.0) for k in keys])
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - a @ b / norm) if norm else 1.0


def rank_distance(first: Mapping[str, float], second: Mapping[str, float], top: int = 200) -> float:
    keys = sorted(set(first) | set(second), key=lambda k: -(first.get(k, 0) + second.get(k, 0)))
    keys = keys[:top]
    a = np.array([first.get(k, 0.0) for k in keys])
    b = np.array([second.get(k, 0.0) for k in keys])
    rho = spearman(a, b)
    return 1.0 if np.isnan(rho) else 1.0 - rho


def halves(texts: Sequence[str]) -> tuple[list[str], list[str]]:
    return list(texts[0::2]), list(texts[1::2])


def table(
    grouped: Mapping[tuple[str, str], Sequence[str]],
    prompt: str,
    extract: Extractor,
    distance: Callable[[Mapping[str, float], Mapping[str, float]], float],
) -> dict[str, float]:
    endpoints = sorted(e for e, p in grouped if p == prompt)
    split = {e: halves(grouped[(e, prompt)]) for e in endpoints}
    prof = {e: (profile(a, extract), profile(b, extract)) for e, (a, b) in split.items()}
    within = {e: distance(prof[e][0], prof[e][1]) for e in endpoints}

    def between(x: str, y: str) -> float:
        return float(np.mean([distance(prof[x][0], prof[y][1]), distance(prof[x][1], prof[y][0])]))

    def mean_pairs(pairs: list[tuple[str, str]]) -> float:
        values = [between(x, y) for x, y in pairs if x in prof and y in prof]
        return float(np.mean(values)) if values else float("nan")

    return {
        "noise": float(np.mean([within[e] for e in endpoints])),
        "noise_S": float(np.mean([within[e] for e in S if e in within])),
        "a_SS": mean_pairs(list(itertools.combinations(S, 2))),
        "b_QS": mean_pairs([(q, s) for q in QUANT for s in S]),
        "b_Q01": mean_pairs([("ref-02", "ref-01")]),
        "c_LS": mean_pairs([(lo, s) for lo in LORA for s in S]),
        "c_L03": mean_pairs([("ref-09", s) for s in S]),
        "c_LL": mean_pairs(list(itertools.combinations(LORA, 2))),
    }


EXTRACTORS: dict[str, Extractor] = {"char3": char_ngrams, "words": words}
DISTANCES = {"cos": cosine_distance, "rank": rank_distance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    grouped = by_endpoint_prompt(load_rows(args.answers))
    wanted = set(filter(None, args.ids.split(",")))
    prompts = sorted({p for _, p in grouped if not wanted or p in wanted})
    header = ["prompt", "feat", "dist", "noise", "noise_S", "a_SS", "b_QS", "b_Q01"]
    header += ["c_LS", "c_L03", "c_LL"]
    print(" | ".join(header))
    for prompt in prompts:
        for fname, extract in EXTRACTORS.items():
            for dname, distance in DISTANCES.items():
                row = table(grouped, prompt, extract, distance)
                cells = [f"{row[k]:.3f}" for k in header[3:]]
                print(" | ".join([prompt, fname, dname, *cells]))


if __name__ == "__main__":
    main()
