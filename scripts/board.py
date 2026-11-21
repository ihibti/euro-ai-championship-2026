"""List a task's result records, best score first.

make board TASK=model_equality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import CHALLENGE_TASKS  # noqa: E402
from euroai.records import board_rows, format_board, load_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(CHALLENGE_TASKS))
    parser.add_argument("--results", type=Path, help="default tasks/<task>/results")
    args = parser.parse_args(argv)
    results = args.results or ROOT / "tasks" / args.task / "results"
    print(format_board(board_rows(load_records(results))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
