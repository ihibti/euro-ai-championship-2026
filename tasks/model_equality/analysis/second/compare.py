"""Compare two groupings pair by pair and list every pair where they disagree.

Usage, from the repository root:
  uv run python tasks/model_equality/analysis/second/compare.py --mine MINE.csv \\
      --theirs THEIRS.csv [--evidence MINE_pairs.csv] [--out REPORT.md]

A pair is "same" in a grouping when both endpoints carry the same model label (and therefore the
same family). The evidence file is the *_pairs.csv that group2.py writes next to its grouping.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from euroai.me_score import parse_labels

Labels = dict[str, tuple[str, str]]


def disagreements(mine: Labels, theirs: Labels) -> list[tuple[str, str, bool, bool]]:
    shared = sorted(set(mine) & set(theirs))
    rows = []
    for a, b in itertools.combinations(shared, 2):
        same_mine = mine[a] == mine[b]
        same_theirs = theirs[a] == theirs[b]
        if same_mine != same_theirs:
            rows.append((a, b, same_mine, same_theirs))
    return rows


def family_disagreements(mine: Labels, theirs: Labels) -> list[tuple[str, str, bool, bool]]:
    shared = sorted(set(mine) & set(theirs))
    rows = []
    for a, b in itertools.combinations(shared, 2):
        fam_mine = mine[a][0] == mine[b][0]
        fam_theirs = theirs[a][0] == theirs[b][0]
        if fam_mine != fam_theirs:
            rows.append((a, b, fam_mine, fam_theirs))
    return rows


def read_evidence(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {(row["a"], row["b"]): row for row in csv.DictReader(handle)}


def groups_of(labels: Labels) -> list[list[str]]:
    by_label: dict[tuple[str, str], list[str]] = {}
    for endpoint, label in sorted(labels.items()):
        by_label.setdefault(label, []).append(endpoint)
    return sorted(by_label.values(), key=lambda g: g[0])


def report(mine: Labels, theirs: Labels, evidence: dict) -> str:
    rows = disagreements(mine, theirs)
    family_rows = family_disagreements(mine, theirs)
    pairs = len(list(itertools.combinations(sorted(set(mine) & set(theirs)), 2)))
    lines = [
        "# Second method against the main line, pair by pair",
        "",
        f"Endpoints in both: {len(set(mine) & set(theirs))}; pairs: {pairs}; same-model"
        f" disagreements: {len(rows)}; same-family disagreements: {len(family_rows)}.",
        "",
        "Second method's model groups: "
        + "; ".join(" ".join(g) for g in groups_of(mine) if len(g) > 1)
        + ". Singletons: "
        + " ".join(g[0] for g in groups_of(mine) if len(g) == 1),
        "",
        "Main line's model groups: "
        + "; ".join(" ".join(g) for g in groups_of(theirs) if len(g) > 1)
        + ". Singletons: "
        + " ".join(g[0] for g in groups_of(theirs) if len(g) == 1),
        "",
    ]
    if rows:
        prompt_cols = sorted(
            {k for row in evidence.values() for k in row if k not in ("a", "b", "z", "p")}
        )
        lines += [
            "| a | b | second says | main line says | z | p | "
            + " | ".join(f"z {c}" for c in prompt_cols)
            + " |",
            "|---|---|---|---|---|---|" + "---|" * len(prompt_cols),
        ]
        for a, b, same_mine, same_theirs in rows:
            ev = evidence.get((a, b)) or evidence.get((b, a)) or {}
            lines.append(
                f"| {a} | {b} | {'same' if same_mine else 'different'} "
                f"| {'same' if same_theirs else 'different'} "
                f"| {ev.get('z', '')} | {ev.get('p', '')} | "
                + " | ".join(ev.get(c, "") for c in prompt_cols)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mine", type=Path, required=True)
    parser.add_argument("--theirs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    mine = parse_labels(args.mine.read_text(encoding="utf-8"))
    theirs = parse_labels(args.theirs.read_text(encoding="utf-8"))
    text = report(mine, theirs, read_evidence(args.evidence))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
