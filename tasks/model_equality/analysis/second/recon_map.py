"""Map the endpoints from the reconnaissance prompts: families, visible disguises, look-alikes.

Built without reading the main line's map. Prompts: idn_identity, txt_story, txt_health,
bat_trivia, bat_choice (two answers each per endpoint on the scored set).

Families: three distances per pair of endpoints, each standardised over all pairs and averaged:
1 - correlation of the per-item trivia correctness, mean gap of the per-item A share on
bat_choice, and 1 - cosine of character trigram profiles of all free text. Average linkage,
cut at the largest gap between successive merge heights among the last eight merges.

Usage, from the repository root:
  uv run python tasks/model_equality/analysis/second/recon_map.py --prefix ep- \\
      --answers tasks/model_equality/answers/round1.jsonl --out recon_ep
Writes <out>_tree.md and <out>.csv (endpoint,family,model with one model per endpoint).
--k fixes the number of families instead of the gap rule.
"""

from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter
from pathlib import Path

import numpy as np
import secondlib as s
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from euroai.me_labels import canonical_labels, render_rows

HERE = Path(__file__).resolve().parent
TEXT = ("idn_identity", "txt_story", "txt_health")
NAMES = (
    "smollm",
    "hugging face",
    "huggingface",
    "llama",
    "meta",
    "qwen",
    "alibaba",
    "mistral",
    "gemma",
    "google",
    "olmo",
    "allen",
    "ai2",
    "phi",
    "microsoft",
    "gpt",
    "openai",
    "claude",
    "anthropic",
    "deepseek",
    "falcon",
    "granite",
    "ibm",
    "pythia",
    "eleuther",
    "stablelm",
    "tinyllama",
    "internlm",
    "nemotron",
    "nvidia",
    "minicpm",
    "cohere",
)


def identity_names(texts: list[str]) -> Counter:
    found: Counter = Counter()
    for text in texts:
        padded = f" {s.normalize(text)} "
        for name in NAMES:
            if f" {name} " in padded:
                found[name] += 1
    return found


def item_rates(matrix: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(matrix)
    total = np.where(valid, matrix, 0.0).sum(axis=0)
    count = valid.sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def profile(texts: list[str]) -> Counter:
    grams: Counter = Counter()
    for text in texts:
        grams.update(s.char_grams(text))
    return grams


def cosine(a: Counter, b: Counter) -> float:
    shared = sum(a[g] * b[g] for g in a.keys() & b.keys())
    norm = np.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return float(shared / norm) if norm else 0.0


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def standardise(values: dict) -> dict:
    array = np.array([v for v in values.values() if np.isfinite(v)])
    mean, spread = float(array.mean()), float(array.std()) or 1.0
    return {k: ((v - mean) / spread if np.isfinite(v) else 0.0) for k, v in values.items()}


def pair_distances(corpus: s.Corpus, endpoints: list[str]) -> tuple[dict, dict, dict, dict]:
    batteries = s.item_matrices(corpus)
    trivia = {e: item_rates(batteries[e]["bat_trivia"]) for e in endpoints}
    choice = {e: item_rates(batteries[e]["bat_choice"]) for e in endpoints}
    texts = {e: [t for q in TEXT for t in corpus.answers[e].get(q, [])] for e in endpoints}
    grams = {e: profile(texts[e]) for e in endpoints}
    d_trivia, d_choice, d_text = {}, {}, {}
    for a, b in itertools.combinations(endpoints, 2):
        both = np.isfinite(trivia[a]) & np.isfinite(trivia[b])
        ta, tb = trivia[a][both], trivia[b][both]
        if both.sum() >= 5 and ta.std() > 0 and tb.std() > 0:
            d_trivia[(a, b)] = 1.0 - float(np.corrcoef(ta, tb)[0, 1])
        else:
            d_trivia[(a, b)] = np.nan
        both = np.isfinite(choice[a]) & np.isfinite(choice[b])
        d_choice[(a, b)] = (
            float(np.mean(np.abs(choice[a][both] - choice[b][both]))) if both.any() else np.nan
        )
        d_text[(a, b)] = 1.0 - cosine(grams[a], grams[b])
    parts = {"trivia": d_trivia, "choice": d_choice, "text": d_text}
    return parts, trivia, choice, texts


def combine_parts(parts: list[dict]) -> dict:
    standardised = [standardise(p) for p in parts]
    return {k: float(np.mean([p[k] for p in standardised])) for k in parts[0]}


def cluster(
    endpoints: list[str], combined: dict, k: int | None = None
) -> tuple[list[list[str]], np.ndarray, float]:
    size = len(endpoints)
    matrix = np.zeros((size, size))
    for (a, b), value in combined.items():
        i, j = endpoints.index(a), endpoints.index(b)
        matrix[i, j] = matrix[j, i] = value
    shifted = matrix - matrix.min()
    np.fill_diagonal(shifted, 0.0)
    tree = linkage(squareform(shifted, checks=False), method="average")
    heights = tree[:, 2]
    tail = heights[-8:] if len(heights) >= 8 else heights
    if k is None:
        gaps = np.diff(tail)
        cut_index = int(np.argmax(gaps)) if len(gaps) else 0
        threshold = (
            float((tail[cut_index] + tail[cut_index + 1]) / 2.0) if len(gaps) else float("inf")
        )
        clusters = fcluster(tree, t=threshold, criterion="distance")
    else:
        threshold = float("nan")
        clusters = fcluster(tree, t=k, criterion="maxclust")
    families: dict[int, list[str]] = {}
    for endpoint, label in zip(endpoints, clusters, strict=True):
        families.setdefault(int(label), []).append(endpoint)
    return sorted(families.values(), key=lambda g: g[0]), tail, threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, help="number of families; default: the gap rule")
    args = parser.parse_args(argv)
    rows = [row for path in args.answers for row in s.read_jsonl(path)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    endpoints = sorted(corpus.answers)
    parts, trivia, choice, texts = pair_distances(corpus, endpoints)
    combined = combine_parts(list(parts.values()))
    ordered, tail, threshold = cluster(endpoints, combined, args.k)

    labels = {e: (f"F{i}", f"F{i}-{e}") for i, fam in enumerate(ordered, start=1) for e in fam}
    out = HERE / args.out
    rows_out = canonical_labels([(e, f, m) for e, (f, m) in labels.items()])
    out.with_suffix(".csv").write_text(render_rows(rows_out), encoding="utf-8")

    letter = {e: f for e, f, _ in rows_out}
    sizes = [len(r) for per in corpus.answers.values() for r in per.values()]
    lines = [
        f"# Map of the `{args.prefix}` endpoints from the reconnaissance prompts",
        "",
        "Built by the second method without reading the main line's map."
        f" Answers per endpoint and prompt: {min(sizes)} to {max(sizes)}.",
        "",
        f"Families: {len(ordered)}, "
        + (
            f"cut at the largest gap among the last merge heights; threshold {threshold:.2f}."
            if args.k is None
            else f"k = {args.k} set by the caller."
        )
        + f" Last merge heights: {', '.join(f'{h:.2f}' for h in tail)}.",
        "",
        "## Families and their evidence",
        "",
    ]
    for fam in ordered:
        inside = [combined[p] for p in itertools.combinations(sorted(fam), 2)]
        outside = [combined[tuple(sorted((a, b)))] for a in fam for b in endpoints if b not in fam]
        names = sum((identity_names(texts[e]) for e in fam), Counter())
        lines += [
            f"### Family {letter[fam[0]]}: {' '.join(fam)}",
            "",
            f"- Mean standardised distance inside {mean_or_nan(inside):.2f},"
            f" to the other endpoints {mean_or_nan(outside):.2f}.",
            "- Names claimed in identity and free text: "
            + (", ".join(f"{k} {v}" for k, v in names.most_common(6)) or "none"),
            "",
        ]
    lines += [
        "## Per endpoint",
        "",
        "| endpoint | family | trivia correct | bat_choice A share | free-text chars "
        "| names claimed | identity answer, first 90 chars |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in endpoints:
        idn = corpus.answers[e].get("idn_identity", [""])
        first = re.sub(r"\s+", " ", idn[0])[:90].replace("|", "/") if idn else ""
        names = identity_names(texts[e])
        chars = int(np.mean([len(t) for t in texts[e]])) if texts[e] else 0
        lines.append(
            f"| {e} | {letter[e]} | {np.nanmean(trivia[e]):.2f} | {np.nanmean(choice[e]):.2f} "
            f"| {chars} | {', '.join(names) or '-'} | {first} |"
        )
    lines += ["", "## Closest pairs inside each family (look-alikes)", ""]
    for fam in ordered:
        pairs = sorted((combined[p], p) for p in itertools.combinations(sorted(fam), 2))
        if pairs:
            lines.append(
                f"- Family {letter[fam[0]]}: "
                + "; ".join(f"{a} {b} ({d:.2f})" for d, (a, b) in pairs[:5])
            )
    out.with_name(f"{out.name}_tree.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.name}_tree.md and {out.name}.csv; families: {len(ordered)}")
    for fam in ordered:
        print("family", letter[fam[0]], " ".join(fam))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
