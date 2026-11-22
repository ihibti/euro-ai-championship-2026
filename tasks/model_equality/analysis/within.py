"""A partition inside each family from frozen per-probe tests, and a CSV in submission format.

uv run python tasks/model_equality/analysis/within.py \
    --answers tasks/model_equality/answers/night-d.jsonl tasks/model_equality/answers/round1.jsonl \
    --prefix ref- --probes d1_repeat150:count,d4_copycont,idn_identity:identity \
    --truth tasks/model_equality/upstream/practice_answers.csv --resamples 100

Probe kinds (--probes id[:kind],...):
- flag: the rule of stop.py for the id (copy-only for p1_copycont and d4_copycont, cap hit
  otherwise), two-sided Fisher exact test; the default kind.
- count: the counter of stop.py for the id, two-sided rank permutation test.
- identity: the answer's identity class (identity.py) is not the family's own group, Fisher.
- long: stop.ran_long (at least 192 pieces or at least 960 characters), Fisher; frozen after
  the scored d1 pilot, which showed runs of "river" without spaces.
Partition of one family:
1. For every probe, stop.set_apart against the pooled majority at alpha (0.01): the endpoint
   most unlike the pool of the others is set apart while p < alpha, then the pool is rebuilt.
2. The endpoints never set apart by any probe form one model, the majority.
3. The set-apart endpoints are taken in the order they were set apart. Each joins the first
   group whose every member no probe separates from it pairwise at alpha; otherwise it starts
   a new model.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import identity  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

from euroai.me_labels import canonical_labels, render_rows  # noqa: E402
from euroai.me_score import parse_labels, same_model_mcc  # noqa: E402

KINDS = ("flag", "count", "identity", "long")
Values = Mapping[str, Sequence]


@dataclass(frozen=True)
class Probe:
    prompt_id: str
    kind: str


def parse_probes(text: str) -> list[Probe]:
    probes = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        prompt_id, _, kind = item.partition(":")
        kind = kind or "flag"
        if kind not in KINDS:
            raise ValueError(f"unknown probe kind {kind!r} in {item!r}")
        if kind == "count" and prompt_id not in stop.COUNTERS:
            raise ValueError(f"no counter for {prompt_id!r}")
        probes.append(Probe(prompt_id, kind))
    return probes


def identity_flags(
    rows: Iterable[Mapping[str, object]], prompt_id: str, members: Sequence[str]
) -> dict[str, list[bool]]:
    wanted = set(members)
    answers: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        if row["id"] == prompt_id and row["endpoint"] in wanted and row.get("response"):
            answers[str(row["endpoint"])].append((int(row["repeat"]), str(row["response"])))
    texts = {e: [t for _, t in sorted(v)] for e, v in answers.items()}
    base = identity.base_group(t for v in texts.values() for t in v)
    return {
        e: [identity.classify(t, base)[0] != "own" for t in v] for e, v in sorted(texts.items())
    }


def probe_values(
    rows: Sequence[Mapping[str, object]], probe: Probe, members: Sequence[str]
) -> dict[str, list]:
    wanted = set(members)
    if probe.kind == "identity":
        return identity_flags(rows, probe.prompt_id, members)
    own = [r for r in rows if r["endpoint"] in wanted]
    if probe.kind == "count":
        return stop.counts(own, probe.prompt_id, "")
    if probe.kind == "long":
        return long_flags(own, probe.prompt_id)
    return stop.cap_hits(own, probe.prompt_id, "")


def long_flags(rows: Iterable[Mapping[str, object]], prompt_id: str) -> dict[str, list[bool]]:
    cells: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in rows:
        if row["id"] == prompt_id and row.get("response") is not None:
            text = str(row["response"])
            cells[str(row["endpoint"])].append((int(row["repeat"]), stop.ran_long(text)))
    return {e: [hit for _, hit in sorted(v)] for e, v in sorted(cells.items())}


def pool_test(kind: str) -> Callable[[Values, str, Sequence[str]], float]:
    return stop.rank_against_pool if kind == "count" else stop.against_pool


def pairwise_p(kind: str, first: Sequence, second: Sequence) -> float:
    if kind == "count":
        return stop.rank_pvalue(first, second)
    return stop.fisher_two_sided(sum(first), len(first), sum(second), len(second))


def partition(
    members: Sequence[str],
    measured: Sequence[tuple[str, Values]],
    alpha: float = stop.ALPHA,
) -> list[list[str]]:
    flagged: list[str] = []
    for kind, values in measured:
        for endpoint, _ in stop.set_apart(values, members, pool_test(kind), alpha):
            if endpoint not in flagged:
                flagged.append(endpoint)

    def separated(x: str, y: str) -> bool:
        return any(
            pairwise_p(kind, values[x], values[y]) < alpha
            for kind, values in measured
            if values.get(x) and values.get(y)
        )

    groups: list[list[str]] = []
    for endpoint in flagged:
        home = next((g for g in groups if not any(separated(endpoint, o) for o in g)), None)
        if home is None:
            groups.append([endpoint])
        else:
            home.append(endpoint)
    majority = [m for m in members if m not in flagged]
    return ([majority] if majority else []) + groups


def read_families(path: Path | None, endpoints: Sequence[str]) -> dict[str, list[str]]:
    if path is None:
        return {"F": sorted(endpoints)}
    families: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["endpoint"] in endpoints:
                families[row["family"]].append(row["endpoint"])
    return {f: sorted(m) for f, m in families.items()}


def grouping(
    rows: Sequence[Mapping[str, object]],
    families: Mapping[str, Sequence[str]],
    probes: Sequence[Probe],
    alpha: float = stop.ALPHA,
) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for family, members in families.items():
        measured = [(p.kind, probe_values(rows, p, members)) for p in probes]
        out[family] = partition(members, measured, alpha)
    return out


def label_rows(parts: Mapping[str, Sequence[Sequence[str]]]) -> list[tuple[str, str, str]]:
    rows = [
        (endpoint, family, f"{family}{number}")
        for family, groups in parts.items()
        for number, group in enumerate(groups, start=1)
        for endpoint in group
    ]
    return canonical_labels(sorted(rows))


def resample(rows: Sequence[Mapping[str, object]], rng: random.Random) -> list[dict]:
    cells: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["endpoint"]), str(row["id"]))].append(row)
    out: list[dict] = []
    for (_, _), cell in sorted(cells.items()):
        for repeat in range(len(cell)):
            out.append({**rng.choice(cell), "repeat": repeat})
    return out


def model_mcc(
    truth: Mapping[str, tuple[str, str]], labels: Iterable[tuple[str, str, str]]
) -> float:
    guess = {endpoint: (family, model) for endpoint, family, model in labels}
    return same_model_mcc({e: truth[e] for e in guess}, guess).mcc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--families", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=stop.ALPHA)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    probes = parse_probes(args.probes)
    ids = {p.prompt_id for p in probes}
    rows = [
        r
        for r in load_rows(args.answers)
        if str(r["endpoint"]).startswith(args.prefix) and r["id"] in ids
    ]
    families = read_families(args.families, sorted({str(r["endpoint"]) for r in rows}))
    parts = grouping(rows, families, probes, args.alpha)
    for family, groups in parts.items():
        print(f"- family {family}: " + " | ".join(" ".join(g) for g in groups))
    labels = label_rows(parts)
    if args.out is not None:
        args.out.write_text(render_rows(labels))
        print(f"wrote {args.out}")
    if args.truth is not None:
        truth = parse_labels(args.truth.read_text())
        print(f"MCC (pairs inside families): {model_mcc(truth, labels):.3f}")
        if args.resamples:
            rng = random.Random(args.seed)
            scores = [
                model_mcc(
                    truth, label_rows(grouping(resample(rows, rng), families, probes, args.alpha))
                )
                for _ in range(args.resamples)
            ]
            scores.sort()
            mean = sum(scores) / len(scores)
            low, high = scores[int(0.05 * len(scores))], scores[int(0.95 * len(scores)) - 1]
            print(
                f"resampled MCC over {len(scores)} draws: mean {mean:.3f}, "
                f"5% {low:.3f}, 95% {high:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
