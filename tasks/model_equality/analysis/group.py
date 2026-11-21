"""Group endpoints by model identity from an answers file and write a submission CSV.

uv run python tasks/model_equality/analysis/group.py --prefix ref- \
    --answers tasks/model_equality/answers/round1.jsonl --out <csv>

Each answer becomes a sparse unit vector of observations, chosen by the prompt id's kind
(the part before the first underscore). Two endpoints are compared by the normalised kernel
similarity of their answers to the same prompts, read against the similarity of an endpoint's
own answers. Endpoints are clustered with average linkage, so the labels are transitive.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows, parse_numbered  # noqa: E402

Vector = dict[str, float]
WORD = re.compile(r"[a-z0-9']+")
MODEL_KINDS = ("seq", "bat", "txt")
FAMILY_KINDS = ("txt", "idn")


def unit(counts: Mapping[str, float]) -> Vector:
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return {k: v / norm for k, v in counts.items()} if norm else {}


def dot(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    if len(first) > len(second):
        first, second = second, first
    return sum(v * second.get(k, 0.0) for k, v in first.items())


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def seq_features(text: str) -> Vector:
    clean = squash(text)
    feats: Counter[str] = Counter()
    feats[f"full={clean[:400]}"] += 2.0
    feats[f"head={clean[:24]}"] += 1.0
    feats[f"tail={clean[-12:]}"] += 1.0
    tokens = WORD.findall(clean)
    for a, b in zip(tokens, tokens[1:], strict=False):
        feats[f"bi={a} {b}"] += 1.0 / max(1, len(tokens))
    return unit(feats)


def bat_features(text: str, count: int = 40) -> Vector:
    feats: Counter[str] = Counter()
    for index, value in enumerate(parse_numbered(text, count)):
        if value is None:
            continue
        tokens = WORD.findall(value.lower())
        if tokens:
            feats[f"i{index + 1}={tokens[0]}"] += 1.0
    return unit(feats)


def txt_features(text: str) -> Vector:
    tokens = WORD.findall(text.lower())
    feats: Counter[str] = Counter(f"w={t}" for t in tokens)
    return unit(feats)


def idn_features(text: str) -> Vector:
    return txt_features(text)


FEATURES: dict[str, Callable[[str], Vector]] = {
    "seq": seq_features,
    "bat": bat_features,
    "txt": txt_features,
    "idn": idn_features,
}


def kind_of(prompt_id: str) -> str:
    return prompt_id.split("_", 1)[0]


def collect(
    rows: Iterable[Mapping[str, object]], prefix: str
) -> dict[str, dict[str, list[Vector]]]:
    cells: dict[str, dict[str, list[tuple[int, Vector]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        endpoint, prompt_id, text = str(row["endpoint"]), str(row["id"]), row.get("response")
        if not endpoint.startswith(prefix) or text is None:
            continue
        extract = FEATURES.get(kind_of(prompt_id))
        if extract is None:
            continue
        cells[endpoint][prompt_id].append((int(row["repeat"]), extract(str(text))))
    return {
        e: {p: [v for _, v in sorted(items, key=lambda x: x[0])] for p, items in prompts.items()}
        for e, prompts in cells.items()
    }


def cross_mean(first: Sequence[Vector], second: Sequence[Vector]) -> float:
    values = [dot(a, b) for a in first for b in second]
    return float(np.mean(values)) if values else float("nan")


def self_mean(vectors: Sequence[Vector]) -> float:
    values = [dot(a, b) for a, b in itertools.combinations(vectors, 2)]
    return float(np.mean(values)) if values else float("nan")


def similarity_matrix(
    cells: Mapping[str, Mapping[str, Sequence[Vector]]], kinds: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    endpoints = sorted(cells)
    prompts = sorted(
        {p for e in endpoints for p in cells[e] if kind_of(p) in kinds},
    )
    n = len(endpoints)
    cross = np.zeros((n, n))
    own = np.zeros(n)
    for index, endpoint in enumerate(endpoints):
        values = [self_mean(cells[endpoint].get(p, [])) for p in prompts]
        own[index] = np.nanmean(values) if values else float("nan")
    for i, j in itertools.combinations(range(n), 2):
        values = [
            cross_mean(cells[endpoints[i]].get(p, []), cells[endpoints[j]].get(p, []))
            for p in prompts
        ]
        cross[i, j] = cross[j, i] = np.nanmean(values) if values else float("nan")
    sim = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        denominator = math.sqrt(max(own[i], 1e-9) * max(own[j], 1e-9))
        sim[i, j] = sim[j, i] = min(1.0, cross[i, j] / denominator)
    return endpoints, sim


def average_linkage(distance: np.ndarray, threshold: float) -> list[int]:
    clusters: list[list[int]] = [[i] for i in range(len(distance))]
    while len(clusters) > 1:
        best, pair = math.inf, None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            value = float(np.mean([distance[i, j] for i in clusters[a] for j in clusters[b]]))
            if value < best:
                best, pair = value, (a, b)
        if pair is None or best > threshold:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * len(distance)
    for number, members in enumerate(sorted(clusters, key=min)):
        for member in members:
            labels[member] = number
    return labels


def nested_labels(
    endpoints: Sequence[str], family: Sequence[int], model: Sequence[int]
) -> list[tuple[str, str, str]]:
    names: dict[tuple[int, int], str] = {}
    rows = []
    for endpoint, fam, mod in zip(endpoints, family, model, strict=False):
        key = (fam, mod)
        if key not in names:
            names[key] = f"F{fam + 1}M{len([k for k in names if k[0] == fam]) + 1}"
        rows.append((endpoint, f"F{fam + 1}", names[key]))
    return rows


def write_csv(path: Path, rows: Sequence[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-threshold", type=float, default=0.1)
    parser.add_argument("--family-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    cells = collect(load_rows(args.answers), args.prefix)
    endpoints, model_sim = similarity_matrix(cells, MODEL_KINDS)
    _, family_sim = similarity_matrix(cells, FAMILY_KINDS)
    family = average_linkage(1 - family_sim, args.family_threshold)
    model = average_linkage(
        1 - model_sim + 2.0 * (np.subtract.outer(family, family) != 0), args.model_threshold
    )
    write_csv(args.out, nested_labels(endpoints, family, model))
    print(f"wrote {args.out}: {len(set(family))} families, {len(set(model))} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
