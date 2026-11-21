"""Render a `<PLACEHOLDER>` template, refusing to output anything left unfilled.

Example (on a JURECA login node):
    python3 scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=unlearn \
        TASK_DIR=/p/scratch/training2666/hibti1/Begg/unlearning TIME=01:00:00 \
        COMMAND='python unlearn.py' > job.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from euroai.templating import find_placeholders, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("template", type=Path)
    parser.add_argument("assignments", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--list", action="store_true", help="list placeholders and exit")
    args = parser.parse_args(argv)
    text = args.template.read_text()
    if args.list:
        print("\n".join(sorted(find_placeholders(text))))
        return 0
    values = dict(item.split("=", 1) for item in args.assignments)
    try:
        sys.stdout.write(render(text, values))
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
