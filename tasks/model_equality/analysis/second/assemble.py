"""Assemble a full grouping: families from one CSV, models inside some families from others.

A family with no endpoint in any --within CSV is one model. An endpoint's model inside its
family is the model label it has in the first --within CSV that holds it; endpoints of a
covered family that no --within CSV holds join that family's largest model. Labels are
renamed to the organizers' format (A, A1, ...).

Usage, from this directory:
  uv run python assemble.py --families recon_ep.csv --within grouping_ep_p3_B.csv \\
      --out grouping_ep.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from euroai.me_labels import canonical_labels, render_rows
from euroai.me_score import parse_labels


def assemble(
    families: dict[str, tuple[str, str]], within: list[dict[str, tuple[str, str]]]
) -> list[tuple[str, str, str]]:
    model: dict[str, str] = {}
    for index, labels in enumerate(within):
        for endpoint, (_, label) in labels.items():
            model.setdefault(endpoint, f"w{index}-{label}")
    rows = []
    for family in sorted({f for f, _ in families.values()}):
        members = sorted(e for e, (f, _) in families.items() if f == family)
        found = Counter(model[e] for e in members if e in model)
        default = found.most_common(1)[0][0] if found else "all"
        rows += [(e, family, f"{family}-{model.get(e, default)}") for e in members]
    return canonical_labels(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--within", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    families = parse_labels(args.families.read_text(encoding="utf-8"))
    within = [parse_labels(path.read_text(encoding="utf-8")) for path in args.within]
    rows = assemble(families, within)
    args.out.write_text(render_rows(rows), encoding="utf-8")
    models = Counter(m for _, _, m in rows)
    print(f"wrote {args.out}: {len(rows)} endpoints, {len(models)} models")
    for label, size in sorted(models.items()):
        print(f"  {label}: {size} {' '.join(e for e, _, m in rows if m == label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
