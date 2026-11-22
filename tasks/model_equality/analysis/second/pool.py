"""Pool answers of the same prompt asked under different ids into one id, for re-reading.

Each input is FILE:ID. Rows of that id are written under the new id, with their repeats shifted
so that no two pooled answers share an (endpoint, id, repeat) key. Writes outside `answers/`.

Usage, from this directory:
  uv run python pool.py --into p3_repeat --out /tmp/p3_pooled_refs.jsonl \\
      ../../answers/night-b.jsonl:p3_repeat ../../answers/night-d.jsonl:d1_repeat150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--into", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    offset, written = 0, 0
    with args.out.open("w", encoding="utf-8") as handle:
        for source in args.sources:
            path, probe = source.rsplit(":", 1)
            rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
            rows = [r for r in rows if r["id"] == probe]
            for row in rows:
                handle.write(
                    json.dumps({**row, "id": args.into, "repeat": int(row["repeat"]) + offset})
                    + "\n"
                )
                written += 1
            offset += 1 + max((int(r["repeat"]) for r in rows), default=-1)
    print(f"wrote {written} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
