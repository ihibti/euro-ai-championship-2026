"""Score an answer locally, print one line score=<number>, and write a result record.

make score TASK=model_equality ANSWER=answer.csv NOTES="what changed"
make score TASK=watermark_localization ANSWER=train_predictions.npz

model_equality: MCC over the reference pairs ref-01..ref-09, against practice_answers.csv.
watermark_localization: the official per-image mIoU over the 510 train images; the answer is a
.npz (image_names, masks) or a .pt in the submission format, covering every train image.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import CHALLENGE_TASKS  # noqa: E402
from euroai.me_score import parse_labels, same_model_mcc  # noqa: E402
from euroai.provenance import git_state, sha256_file  # noqa: E402
from euroai.records import build_result_record, git_person, record_stem, write_record  # noqa: E402
from euroai.wm_score import (  # noqa: E402
    TRAIN_TRUTH,
    load_masks,
    read_metadata,
    score_predictions,
)

ME_TRUTH = ROOT / "tasks" / "model_equality" / "upstream" / "practice_answers.csv"
Scored = tuple[str, float, dict[str, Any]]


def score_model_equality(answer: Path, truth: Path | None) -> Scored:
    labels = parse_labels(Path(truth or ME_TRUTH).read_text(encoding="utf-8"), prefix="ref-")
    guess = parse_labels(answer.read_text(encoding="utf-8"), prefix="ref-")
    result = same_model_mcc(labels, guess)
    details = {
        "pairs": {"tp": result.tp, "fp": result.fp, "fn": result.fn, "tn": result.tn},
        "wrong_pairs": [list(pair) for pair in result.wrong],
    }
    return "practice_mcc", result.mcc, details


def score_watermark(answer: Path, truth: Path | None) -> Scored:
    truth = truth or ROOT / TRAIN_TRUTH
    if not truth.is_file():
        raise FileNotFoundError(
            f"ground truth not found at {truth}; run make data TASK=watermark_localization "
            "with HF_TOKEN from an approved account"
        )
    gt_names, gt_masks = load_masks(truth)
    names, masks = load_masks(answer)
    metadata = truth.parent / "metadata.json"
    methods = read_metadata(metadata.read_text()) if metadata.is_file() else None
    result = score_predictions(names, masks, gt_names, gt_masks, methods)
    details = {
        "images": len(result.per_image),
        "by_method": result.by_method,
        "ignored_predictions": result.ignored,
    }
    return "train_miou", result.score, details


SCORERS: dict[str, Callable[[Path, Path | None], Scored]] = {
    "model_equality": score_model_equality,
    "watermark_localization": score_watermark,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(CHALLENGE_TASKS))
    parser.add_argument("--answer", required=True, type=Path)
    parser.add_argument("--notes", default="")
    parser.add_argument("--truth", type=Path, help="override the default ground truth")
    parser.add_argument("--results", type=Path, help="default tasks/<task>/results")
    args = parser.parse_args(argv)
    scorer = SCORERS.get(args.task)
    if scorer is None:
        print(
            f"error: {args.task} has no exact local score; the real one needs hidden data. "
            f"See tasks/{args.task}/README.md for what can be measured locally.",
            file=sys.stderr,
        )
        return 2
    try:
        metric, score, details = scorer(args.answer, args.truth)
    except ImportError as exc:
        print(f"error: {exc}; reading .pt answers needs the torch extra", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    person = git_person(ROOT)
    command = f"make score TASK={args.task} ANSWER={args.answer}"
    if args.truth:
        command += f" TRUTH={args.truth}"
    record = build_result_record(
        task=args.task,
        metric=metric,
        score=score,
        command=command,
        notes=args.notes,
        person=person,
        git=git_state(ROOT),
        now=now,
    )
    record["answer"] = {"path": str(args.answer), "sha256": sha256_file(args.answer)}
    record["details"] = details
    results = args.results or ROOT / "tasks" / args.task / "results"
    written = write_record(results, record, stem=record_stem(now, person, metric))
    print(f"score={score:.6f}")
    print(f"{metric}: {score:.6f}; details: {details}; recorded {written}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
