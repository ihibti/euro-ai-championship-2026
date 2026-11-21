"""Download a task's Hugging Face dataset into data/<task>/ and unpack the watermark zips.

make data TASK=model_equality

HF_TOKEN comes from the environment or ~/.config/euroai/.env; the watermark dataset is gated
and needs a token from an approved account.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import CHALLENGE_TASKS, get_task  # noqa: E402
from euroai.credentials import HF_TOKEN_VAR, get_secret  # noqa: E402
from euroai.fetch import extract_zip, extraction_plan, task_data_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(CHALLENGE_TASKS))
    parser.add_argument("--include", action="append", help="only files matching this pattern")
    args = parser.parse_args(argv)
    info = get_task(args.task)
    target = ROOT / task_data_dir(args.task)
    token = get_secret(HF_TOKEN_VAR).value
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=info.hf_dataset,
            repo_type="dataset",
            local_dir=target,
            token=token,
            allow_patterns=args.include,
        )
    except Exception as exc:
        print(f"error: downloading {info.hf_dataset} failed: {type(exc).__name__}", file=sys.stderr)
        print(f"  {str(exc).splitlines()[0] if str(exc) else ''}", file=sys.stderr)
        if token is None:
            print(f"  {HF_TOKEN_VAR} is not set; gated datasets need it", file=sys.stderr)
        return 1
    for zip_path, destination in extraction_plan(target):
        count = extract_zip(zip_path, destination)
        print(f"extracted {zip_path.name}: {count} entries into {destination}")
    print(f"{info.hf_dataset} is in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
