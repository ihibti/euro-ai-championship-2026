"""Evidence for a family split of the endpoints, from the reconnaissance answers.

For a family CSV (endpoint,family,...): the identity phrasing per endpoint, trivia correctness
per item and family with the commonest answer, agreement of each endpoint's bat_choice answers
with each family's leave-one-out consensus, a permutation test of the trivia and choice profiles
between every two families, the stability of the split when the tree of `recon_map.py` is built
from one or two of its three distances, and the response times.

Usage, from this directory:
  uv run python recon_evidence.py --prefix ep- --answers ../../answers/round1.jsonl \\
      --families recon_ep.csv --out recon_ep_evidence
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import Counter
from pathlib import Path

import numpy as np
import recon_map as rm
import secondlib as s

HERE = Path(__file__).resolve().parent
PHRASES = (
    ("Qwen, created by Alibaba", re.compile(r"\bqwen (and )?(i was )?created by alibaba")),
    ("Qwen, a ... model", re.compile(r"\bqwen (a|an) [a-z ]{0,40}?(model|assistant|intelligence)")),
    ("Llama or Meta", re.compile(r"\b(llama|meta)\b")),
)


def phrase_class(text: str) -> str:
    norm = s.normalize(text)
    return next((name for name, pattern in PHRASES if pattern.search(norm)), "other")


def prompt_questions(prompt: str, count: int) -> list[str]:
    questions = [""] * count
    for line in prompt.splitlines():
        match = s.ITEM_LINE.match(line)
        if match and 0 < int(match.group(1)) <= count:
            questions[int(match.group(1)) - 1] = match.group(2).strip()
    return questions


def read_families(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {row["endpoint"]: row["family"] for row in csv.DictReader(handle)}


def pooled_rates(matrices: dict, members: list[str]) -> np.ndarray:
    return rm.item_rates(np.vstack([matrices[e] for e in members]))


def profile_gap(matrices: dict, first: list[str], second: list[str]) -> float:
    gap = pooled_rates(matrices, first) - pooled_rates(matrices, second)
    return float(np.nansum(gap**2))


def permutation_p(
    matrices: dict, first: list[str], second: list[str], draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    observed = profile_gap(matrices, first, second)
    pool = first + second
    hits = 0
    for _ in range(draws):
        order = rng.permutation(len(pool))
        a = [pool[i] for i in order[: len(first)]]
        b = [pool[i] for i in order[len(first) :]]
        hits += profile_gap(matrices, a, b) >= observed
    return observed, (1 + hits) / (1 + draws)


def consensus_agreement(matrices: dict, endpoint: str, members: list[str]) -> tuple[float, int]:
    others = [e for e in members if e != endpoint]
    share = pooled_rates(matrices, others)
    decisive = np.isfinite(share) & ((share >= 0.9) | (share <= 0.1))
    mine = rm.item_rates(matrices[endpoint])
    usable = decisive & np.isfinite(mine)
    if not usable.any():
        return float("nan"), 0
    majority = (share[usable] >= 0.5).astype(float)
    return float(1.0 - np.mean(np.abs(mine[usable] - majority))), int(usable.sum())


def misplaced(found: list[list[str]], family: dict[str, str]) -> int:
    used: set[str] = set()
    wrong = 0
    for group in sorted(found, key=len, reverse=True):
        counts = Counter(family[e] for e in group if family[e] not in used)
        label = counts.most_common(1)[0][0] if counts else None
        used.add(label)
        wrong += sum(family[e] != label for e in group)
    return wrong


def short_answer(text: str | None) -> str:
    return s.normalize(text or "")[:24] or "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows = [row for path in args.answers for row in s.read_jsonl(path)]
    rows = [r for r in rows if r["endpoint"].startswith(args.prefix)]
    corpus = s.build_corpus(rows, prefix=args.prefix)
    endpoints = sorted(corpus.answers)
    family = read_families(args.families)
    names = sorted(set(family[e] for e in endpoints))
    members = {f: [e for e in endpoints if family[e] == f] for f in names}
    batteries = s.item_matrices(corpus)
    trivia = {e: batteries[e]["bat_trivia"] for e in endpoints}
    choice = {e: batteries[e]["bat_choice"] for e in endpoints}

    lines = [
        f"# Evidence for the families of the `{args.prefix}` endpoints",
        "",
        f"Families from `{args.families.name}`: "
        + "; ".join(f"{f} ({len(members[f])}): {' '.join(members[f])}" for f in names),
        "",
        "## Identity phrasing (idn_identity, every answer)",
        "",
        "| family | endpoint | classes of its answers |",
        "|---|---|---|",
    ]
    for f in names:
        for e in members[f]:
            classes = Counter(phrase_class(t) for t in corpus.answers[e].get("idn_identity", []))
            lines.append(f"| {f} | {e} | {', '.join(f'{k} {v}' for k, v in classes.items())} |")

    trivia_prompt = next(r["prompt"] for r in rows if r["id"] == "bat_trivia")
    questions = prompt_questions(trivia_prompt, len(s.TRIVIA_KEYS))
    answers: dict[str, list[Counter]] = {f: [Counter() for _ in questions] for f in names}
    for r in rows:
        if r["id"] == "bat_trivia":
            for j, text in enumerate(s.numbered_items(r["response"], len(questions))):
                if text is not None:
                    answers[family[r["endpoint"]]][j][short_answer(text)] += 1
    rates = {f: pooled_rates(trivia, members[f]) for f in names}
    lines += [
        "",
        "## Trivia correctness per item and family (pooled answers), commonest answer",
        "",
        "Items where two families differ by 0.5 or more are marked with `*`.",
        "",
        "| item | question | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for j, question in enumerate(questions):
        values = [rates[f][j] for f in names]
        finite = [v for v in values if np.isfinite(v)]
        mark = "*" if finite and max(finite) - min(finite) >= 0.5 else ""
        cells = [
            f"{v:.2f} {answers[f][j].most_common(1)[0][0] if answers[f][j] else '-'}"
            for f, v in zip(names, values, strict=True)
        ]
        lines.append(f"| {j + 1}{mark} | {question} | " + " | ".join(cells) + " |")

    marked = [
        j
        for j in range(len(questions))
        if (lambda v: v and max(v) - min(v) >= 0.5)(
            [rates[f][j] for f in names if np.isfinite(rates[f][j])]
        )
    ]
    lines += [
        "",
        "Marked items per endpoint: correct answers of the parsed ones, and the answers given.",
        "",
        "| family | endpoint | " + " | ".join(f"item {j + 1}" for j in marked) + " |",
        "|---|---|" + "---|" * len(marked),
    ]
    for f in names:
        for e in members[f]:
            given = [
                s.numbered_items(t, len(questions)) for t in corpus.answers[e].get("bat_trivia", [])
            ]
            cells = []
            for j in marked:
                texts = [g[j] for g in given if g[j] is not None]
                right = sum(s.trivia_value(t, s.TRIVIA_KEYS[j]) == 1.0 for t in texts)
                shown = ", ".join(short_answer(t) for t in texts)
                cells.append(f"{right}/{len(texts)} {shown}")
            lines.append(f"| {f} | {e} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## bat_choice: agreement with each family's leave-one-out consensus",
        "",
        "Consensus items: pooled A share of the family's other endpoints at least 0.9 or at most"
        " 0.1. Cell: share of the endpoint's answers that match the consensus (items used).",
        "",
        "| family | endpoint | " + " | ".join(f"vs {f}" for f in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for f in names:
        for e in members[f]:
            cells = []
            for g in names:
                value, used = consensus_agreement(choice, e, members[g])
                cells.append(f"{value:.2f} ({used})" if used else "n/a")
            lines.append(f"| {f} | {e} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Permutation test between families ({args.draws} random splits of the two families"
        " with their sizes kept)",
        "",
        "Statistic: sum over items of the squared gap between the two families' pooled rates.",
        "",
        "| families | trivia gap | p | choice gap | p |",
        "|---|---|---|---|---|",
    ]
    for f, g in itertools.combinations(names, 2):
        t_obs, t_p = permutation_p(trivia, members[f], members[g], args.draws, rng)
        c_obs, c_p = permutation_p(choice, members[f], members[g], args.draws, rng)
        lines.append(f"| {f} {g} | {t_obs:.2f} | {t_p:.4f} | {c_obs:.2f} | {c_p:.4f} |")

    parts, _, _, _ = rm.pair_distances(corpus, endpoints)
    lines += [
        "",
        f"## Stability: the tree of `recon_map.py` cut at k = {len(names)}, from subsets of its"
        " distances",
        "",
        "| distances | endpoints placed apart from the CSV | families found |",
        "|---|---|---|",
    ]
    for size in (3, 2, 1):
        for subset in itertools.combinations(parts, size):
            combined = rm.combine_parts([parts[k] for k in subset])
            found, _, _ = rm.cluster(endpoints, combined, len(names))
            shown = " / ".join(" ".join(e.removeprefix(args.prefix) for e in g) for g in found)
            lines.append(f"| {' + '.join(subset)} | {misplaced(found, family)} | {shown} |")

    elapsed = {
        e: [r["elapsed_s"] for r in rows if r["endpoint"] == e and "elapsed_s" in r]
        for e in endpoints
    }
    lengths = [len(r["response"]) for r in rows if "elapsed_s" in r]
    times = [r["elapsed_s"] for r in rows if "elapsed_s" in r]
    if times:
        r_len = float(np.corrcoef(lengths, times)[0, 1])
        per_family = "; ".join(
            f"{f} median {np.median([t for e in members[f] for t in elapsed[e]]):.2f} s"
            for f in names
        )
        lines += [
            "",
            "## Response time",
            "",
            f"{len(times)} answers, {min(times):.1f} to {max(times):.1f} s, median"
            f" {np.median(times):.2f} s; correlation with answer length {r_len:.2f}."
            f" Per family: {per_family}.",
        ]
    out = HERE / f"{args.out}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
