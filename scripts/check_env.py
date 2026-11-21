"""Print a read-only JSON report of the Python/GPU/Slurm environment.

Runs with the standard library only and without installing the package, so it
works on a fresh login node:  python3 scripts/check_env.py [--no-probe]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from euroai.env import detect_environment  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report the current environment as JSON.")
    parser.add_argument("--no-probe", action="store_true", help="do not import torch")
    args = parser.parse_args(argv)
    info = detect_environment(probe=not args.no_probe)
    print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
