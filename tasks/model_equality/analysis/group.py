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
    cells: Mapping[str, Mapping[str, Sequence[Vector]]],
    kinds: Sequence[str],
    mode: str = "pooled",
) -> tuple[list[str], np.ndarray]:
    if mode == "prompt":
        return prompt_similarity_matrix(cells, kinds)
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
        sim[i, j] = sim[j, i] = cross[i, j] / denominator
    return endpoints, sim


def prompt_similarity_matrix(
    cells: Mapping[str, Mapping[str, Sequence[Vector]]], kinds: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    endpoints = sorted(cells)
    prompts = sorted({p for e in endpoints for p in cells[e] if kind_of(p) in kinds})
    own = {(e, p): self_mean(cells[e].get(p, [])) for e in endpoints for p in prompts}
    n = len(endpoints)
    sim = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        values = []
        for p in prompts:
            a, b = own[(endpoints[i], p)], own[(endpoints[j], p)]
            if not (a > 0 and b > 0):
                continue
            cross = cross_mean(cells[endpoints[i]].get(p, []), cells[endpoints[j]].get(p, []))
            values.append(cross / math.sqrt(a * b))
        sim[i, j] = sim[j, i] = float(np.mean(values)) if values else 0.0
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


def split_cells(
    cells: Mapping[str, Mapping[str, Sequence[Vector]]],
) -> dict[str, dict[str, list[Vector]]]:
    out: dict[str, dict[str, list[Vector]]] = {}
    for endpoint, prompts in cells.items():
        out[f"{endpoint}#a"] = {p: list(v[0::2]) for p, v in prompts.items()}
        out[f"{endpoint}#b"] = {p: list(v[1::2]) for p, v in prompts.items()}
    return out


def self_split_distance(
    cells: Mapping[str, Mapping[str, Sequence[Vector]]],
    kinds: Sequence[str],
    mode: str = "pooled",
) -> dict[str, float]:
    halves = split_cells(cells)
    names, sim = similarity_matrix(halves, kinds, mode)
    index = {name: i for i, name in enumerate(names)}
    return {e: float(1 - sim[index[f"{e}#a"], index[f"{e}#b"]]) for e in cells}


def model_threshold(noise: Mapping[str, float], factor: float) -> float:
    values = np.array(list(noise.values()), dtype=float)
    return factor * float(np.sqrt(np.mean(values**2)))


def merge_clusters(distance: np.ndarray, labels: Sequence[int], threshold: float) -> list[int]:
    groups = sorted(set(labels))
    members = {g: [i for i, label in enumerate(labels) if label == g] for g in groups}
    between = np.zeros((len(groups), len(groups)))
    for a, b in itertools.combinations(range(len(groups)), 2):
        value = float(
            np.mean([distance[i, j] for i in members[groups[a]] for j in members[groups[b]]])
        )
        between[a, b] = between[b, a] = value
    coarse = average_linkage(between, threshold)
    lookup = {g: coarse[k] for k, g in enumerate(groups)}
    return [lookup[label] for label in labels]


def decide(
    rows: Iterable[Mapping[str, object]],
    prefix: str = "ref-",
    factor: float = 3.0,
    family_threshold: float = 0.5,
    report: dict | None = None,
    kinds: Sequence[str] = MODEL_KINDS,
    mode: str = "pooled",
) -> list[tuple[str, str, str]]:
    cells = collect(rows, prefix)
    endpoints, model_sim = similarity_matrix(cells, kinds, mode)
    noise = self_split_distance(cells, kinds, mode)
    threshold = model_threshold(noise, factor)
    if not math.isfinite(threshold):
        raise ValueError("self-split noise undefined: need at least 4 repeats of every prompt")
    model = average_linkage(1 - model_sim, threshold)
    _, family_sim = similarity_matrix(cells, FAMILY_KINDS)
    family = merge_clusters(1 - family_sim, model, family_threshold)
    if report is not None:
        report.update(
            endpoints=endpoints,
            model_sim=model_sim,
            family_sim=family_sim,
            noise=noise,
            threshold=threshold,
        )
    return nested_labels(endpoints, family, model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--family-threshold", type=float, default=0.5)
    parser.add_argument("--below", type=int, default=None, help="use only repeats below this")
    args = parser.parse_args(argv)
    rows = load_rows(args.answers)
    if args.below is not None:
        rows = [row for row in rows if int(row["repeat"]) < args.below]
    report: dict = {}
    labels = decide(
        rows,
        prefix=args.prefix,
        factor=args.factor,
        family_threshold=args.family_threshold,
        report=report,
    )
    write_csv(args.out, labels)
    endpoints = report["endpoints"]
    print(f"model cut {report['threshold']:.3f} (factor x rms self-split distance)")
    print("model distance (1 - similarity); last column: self-split distance")
    for i, endpoint in enumerate(endpoints):
        cells = " ".join(f"{1 - report['model_sim'][i, j]:.2f}" for j in range(len(endpoints)))
        print(f"{endpoint} {cells} | {report['noise'][endpoint]:.2f}")
    families = {family for _, family, _ in labels}
    models = {model for _, _, model in labels}
    print(f"wrote {args.out}: {len(families)} families, {len(models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
