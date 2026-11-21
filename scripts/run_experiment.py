"""Run a registered task from a TOML config.

Example:
    uv run python scripts/run_experiment.py --config configs/example.toml --set task.separation=2.0
"""

from __future__ import annotations

import argparse
import json
import sys

from euroai.config import apply_overrides, load_config
from euroai.experiment import TASKS, run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a TOML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override, repeatable (e.g. experiment.seed=7)",
    )
    parser.add_argument("--output-root", default=None, help="overrides experiment.output_dir")
    parser.add_argument("--no-env-probe", action="store_true", help="skip importing torch")
    parser.add_argument("--print-config", action="store_true", help="print resolved config, exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    if args.print_config:
        print(json.dumps(config, indent=2, default=str))
        return 0
    result = run_experiment(
        config,
        tasks=TASKS,
        output_root=args.output_root,
        probe_environment=not args.no_env_probe,
    )
    print(json.dumps({"run_dir": str(result.run_dir), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
