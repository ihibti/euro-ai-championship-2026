"""Rewrite a model-equality CSV in the organizers' label format (family A, model A1).

uv run python scripts/me_labels.py <grouping.csv> <out.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.me_labels import canonical_labels, parse_rows, render_rows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    rows = canonical_labels(parse_rows(args.source.read_text()))
    args.target.write_text(render_rows(rows))
    families = {family for _, family, _ in rows}
    models = {model for _, _, model in rows}
    counts = f"{len(rows)} endpoints, {len(families)} families, {len(models)} models"
    print(f"wrote {args.target}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
