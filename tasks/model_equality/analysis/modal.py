"""Modal answers on the two round-1 batteries: the items where an endpoint's most frequent
answer differs from the most frequent answer of the rest of its family.

uv run python tasks/model_equality/analysis/modal.py \
    --answers tasks/model_equality/answers/round1.jsonl --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv

Frozen on 2026-11-22 at 02:18 Paris, before any new battery answer existed.
- Items: bat_trivia, read with recon.battery_answers (an echoed question line is dropped), and
  bat_choice, read with recon.first_words; 30 items each, the lowercased first word of each
  numbered line, missing when the line is absent.
- Mode of a set of answers on one item: the set of tokens with the highest count among the
  answers that give the item; empty when none does.
- An item flips when both modes are non-empty and share no token, so a tie that holds the
  other side's answer is agreement.
- Statistic: the flips over the 60 items between the endpoint's answers and the pooled answers
  of the rest of the current pool.
- Null: per battery, as many answers as the endpoint has are drawn without replacement from
  the pooled rest and compared with the remaining pooled answers, the same way; DRAWS draws,
  seed 0; one-sided p = (1 + draws with at least the observed flips) / (DRAWS + 1). Holding the
  drawn answers out of the reference keeps the null from agreeing with itself.
- Decision: stop.set_apart at alpha 0.01, iterative against the family majority.
- Sharpness (added 02:46 Paris, before any fresh battery answer; described only): per endpoint,
  the mean over items of the share of its answers that give its top answer. A hotter copy keeps
  the order of the probabilities but is less sharp on every item; changed weights may flip
  items while staying sharp. A watermark can do either, so sharpness never decides.
- Second view, described only: for each pair of endpoints set apart, the items both flip
  against the final majority, and on how many of those their modes share a token.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon  # noqa: E402
import stop  # noqa: E402
from melib import load_rows  # noqa: E402

ITEMS = 30
BATTERIES = {"bat_trivia": recon.battery_answers, "bat_choice": recon.first_words}
DRAWS = 2000
SEED = 0
Token = str | None
Codes = dict[str, np.ndarray]


def battery_tokens(
    rows: Iterable[Mapping[str, object]], members: Sequence[str]
) -> dict[str, dict[str, list[list[Token]]]]:
    wanted = set(members)
    cells: dict[str, dict[str, list[tuple[int, list[Token]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        prompt_id, endpoint = str(row["id"]), str(row["endpoint"])
        if prompt_id in BATTERIES and endpoint in wanted and row.get("response") is not None:
            tokens = BATTERIES[prompt_id](str(row["response"]), ITEMS)
            cells[endpoint][prompt_id].append((int(row["repeat"]), tokens))
    return {
        e: {b: [t for _, t in sorted(v, key=lambda x: x[0])] for b, v in cells[e].items()}
        for e in sorted(cells)
    }


def encode(tokens: Mapping[str, Mapping[str, Sequence[Sequence[Token]]]]) -> dict[str, Codes]:
    vocab: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for per_battery in tokens.values():
        for battery, answers in per_battery.items():
            for answer in answers:
                for item, token in enumerate(answer):
                    if token is not None:
                        vocab[(battery, item)].setdefault(token, len(vocab[(battery, item)]))
    out: dict[str, Codes] = {}
    for endpoint, per_battery in tokens.items():
        out[endpoint] = {}
        for battery, answers in per_battery.items():
            codes = np.full((len(answers), ITEMS), -1, dtype=np.int64)
            for row, answer in enumerate(answers):
                for item, token in enumerate(answer):
                    if token is not None:
                        codes[row, item] = vocab[(battery, item)][token]
            out[endpoint][battery] = codes
    return out


def item_counts(codes: np.ndarray, size: int) -> np.ndarray:
    return (codes[..., None] == np.arange(size)).sum(axis=-3)


def mode_mask(counts: np.ndarray) -> np.ndarray:
    top = counts.max(axis=-1, keepdims=True)
    return (counts == top) & (top > 0)


def flips(own: np.ndarray, other: np.ndarray) -> np.ndarray:
    a, b = mode_mask(own), mode_mask(other)
    return a.any(axis=-1) & b.any(axis=-1) & ~(a & b).any(axis=-1)


def pooled(values: Mapping[str, Codes], endpoints: Iterable[str], battery: str) -> np.ndarray:
    parts = [values[e][battery] for e in endpoints if battery in values[e]]
    return np.concatenate(parts) if parts else np.empty((0, ITEMS), dtype=np.int64)


def size_of(*arrays: np.ndarray) -> int:
    return int(max((a.max() for a in arrays if a.size), default=-1)) + 1


def observed_flips(own: Codes, rest: Codes) -> int:
    total = 0
    for battery, codes in own.items():
        other = rest.get(battery)
        if other is None or not len(other) or not len(codes):
            continue
        size = size_of(codes, other)
        total += int(flips(item_counts(codes, size), item_counts(other, size)).sum())
    return total


def null_flips(own: Codes, rest: Codes, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros(draws, dtype=np.int64)
    for battery, codes in own.items():
        pool = rest.get(battery)
        n = len(codes)
        if pool is None or not n or len(pool) <= n:
            continue
        size = size_of(codes, pool)
        whole = item_counts(pool, size)
        order = np.argsort(rng.random((draws, len(pool))), axis=1)[:, :n]
        drawn = item_counts(pool[order], size)
        total += flips(drawn, whole[None] - drawn).sum(axis=-1)
    return total


def modal_pvalue(
    values: Mapping[str, Codes], endpoint: str, pool: Sequence[str], draws: int = DRAWS
) -> float:
    others = [e for e in pool if e != endpoint]
    rest = {b: pooled(values, others, b) for b in values[endpoint]}
    seen = observed_flips(values[endpoint], rest)
    null = null_flips(values[endpoint], rest, draws)
    return float((1 + (null >= seen).sum()) / (draws + 1))


def item_modes(codes: Codes) -> dict[str, list[frozenset[int]]]:
    out = {}
    for battery, array in codes.items():
        size = size_of(array)
        mask = mode_mask(item_counts(array, size)) if size else np.zeros((ITEMS, 0), bool)
        out[battery] = [frozenset(np.flatnonzero(row).tolist()) for row in mask]
    return out


def sharpness(codes: Codes) -> float:
    shares = []
    for array in codes.values():
        size = size_of(array)
        if not size:
            continue
        counts = item_counts(array, size)
        total = counts.sum(axis=-1)
        shares.extend((counts.max(axis=-1)[total > 0] / total[total > 0]).tolist())
    return float(np.mean(shares)) if shares else float("nan")


def flipped_items(own: Codes, rest: Codes) -> dict[str, list[int]]:
    out = {}
    for battery, codes in own.items():
        other = rest.get(battery)
        if other is None or not len(other) or not len(codes):
            continue
        size = size_of(codes, other)
        hit = flips(item_counts(codes, size), item_counts(other, size))
        out[battery] = np.flatnonzero(hit).tolist()
    return out


def shared_flips(
    values: Mapping[str, Codes], first: str, second: str, majority: Sequence[str]
) -> tuple[int, int]:
    rest = {b: pooled(values, majority, b) for b in BATTERIES}
    one, two = flipped_items(values[first], rest), flipped_items(values[second], rest)
    modes_one, modes_two = item_modes(values[first]), item_modes(values[second])
    both = same = 0
    for battery in set(one) & set(two):
        for item in set(one[battery]) & set(two[battery]):
            both += 1
            same += bool(modes_one[battery][item] & modes_two[battery][item])
    return both, same


def decide(
    values: Mapping[str, Codes], members: Sequence[str], alpha: float = stop.ALPHA
) -> list[tuple[str, float]]:
    return stop.set_apart(values, members, modal_pvalue, alpha)


def read_families(path: Path | None, endpoints: Sequence[str]) -> dict[str, list[str]]:
    if path is None:
        return {"F": sorted(endpoints)}
    families: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["endpoint"] in endpoints:
                families[row["family"]].append(row["endpoint"])
    return {f: sorted(m) for f, m in families.items()}


def report(values: Mapping[str, Codes], members: Sequence[str], alpha: float) -> None:
    flagged = decide(values, members, alpha)
    apart = [e for e, _ in flagged]
    majority = [m for m in members if m not in apart]
    print(
        "| endpoint | answers trivia/choice | flips trivia | flips choice | p | set apart "
        "| sharpness |"
    )
    print("|---|---|---|---|---|---|---|")
    for e in members:
        pool = [m for m in majority if m != e]
        rest = {b: pooled(values, pool, b) for b in BATTERIES}
        items = flipped_items(values[e], rest)
        sizes = "/".join(str(len(values[e].get(b, []))) for b in BATTERIES)
        p = modal_pvalue(values, e, pool + [e])
        order = next((f"{k + 1} (p {q:.4f})" for k, (x, q) in enumerate(flagged) if x == e), "")
        cells = " | ".join(str(len(items.get(b, []))) for b in BATTERIES)
        sharp = sharpness(values[e])
        print(f"| {e} | {sizes} | {cells} | {p:.4f} | {order} | {sharp:.2f} |")
    for first, second in itertools.combinations(apart, 2):
        both, same = shared_flips(values, first, second, majority)
        print(f"- {first} and {second}: {both} items flipped by both, {same} to a shared answer")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--families", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=stop.ALPHA)
    args = parser.parse_args(argv)
    rows = [r for r in load_rows(args.answers) if str(r["endpoint"]).startswith(args.prefix)]
    endpoints = sorted({str(r["endpoint"]) for r in rows if r["id"] in BATTERIES})
    for family, members in read_families(args.families, endpoints).items():
        print(f"\n== family {family}")
        values = encode(battery_tokens(rows, members))
        report(values, [m for m in members if m in values], args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
