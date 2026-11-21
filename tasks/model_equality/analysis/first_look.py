"""Print, per prompt, the answer length and the first characters of each endpoint's answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import by_endpoint_prompt, load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    parser.add_argument("--ids", default="")
    parser.add_argument("--chars", type=int, default=90)
    args = parser.parse_args()
    grouped = by_endpoint_prompt(load_rows(args.answers))
    wanted = set(filter(None, args.ids.split(",")))
    ids = sorted({prompt for _, prompt in grouped if not wanted or prompt in wanted})
    for prompt in ids:
        print(f"== {prompt}")
        for endpoint in sorted({e for e, p in grouped if p == prompt}):
            texts = grouped[(endpoint, prompt)]
            lengths = ",".join(str(len(t)) for t in texts)
            head = texts[0][: args.chars].replace("\n", " | ")
            print(f"{endpoint} n={len(texts)} len=[{lengths}] {head}")


if __name__ == "__main__":
    main()
