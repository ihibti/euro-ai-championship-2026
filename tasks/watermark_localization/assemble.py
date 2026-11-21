"""Build one candidate from per-method predictions, and score every option on the train split.

For each method the mask comes either from segmentation runs (probabilities at 0.5; several runs
joined with "+" are averaged) or, when no run is named, from the full-frame decode rule (all ones or
all zeros). Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/assemble.py --tag c2 --seg stegastamp=stegastamp_v1
    python <repo>/tasks/watermark_localization/assemble.py --tag c6 --seg trustmark=trustmark_v3+trustmark_x1

Writes output/candidates/<tag>.npz with train_names, train_masks, test_names and test_masks, the
masks bit-packed. pack_submission.py turns that file into the train answer and the test submission
on the laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evidence_features import mean_miou  # noqa: E402

GRID = 128
METHODS = ("pixelseal", "trustmark", "stegastamp")
DECODE_THRESHOLD = {"pixelseal": 0.89, "trustmark": 0.75, "stegastamp": 0.76}


def held_out(name: str) -> bool:
    import zlib

    return zlib.crc32(name.encode()) % 5 == 0


def decode_masks(split: str, method: str) -> tuple[list[str], np.ndarray]:
    evidence = np.load(f"output/evidence/{split}_{method}.npz")
    marked = evidence["crop512_acc"][:, 0].astype(np.float32) >= DECODE_THRESHOLD[method]
    masks = np.repeat(marked[:, None, None], GRID, 1).repeat(GRID, 2).astype(np.uint8)
    return evidence["names"].tolist(), masks


def average(stack: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([probs.astype(np.float32) for probs in stack]), axis=0)


def seg_masks(split: str, runs: str, threshold: float) -> tuple[list[str], np.ndarray]:
    names, stack = None, []
    for run in runs.split("+"):
        saved = np.load(f"output/seg/{run}/{split}.npz")
        if names is None:
            names = saved["names"].tolist()
        if saved["names"].tolist() != names:
            raise SystemExit(f"{run}: the {split} images are not in the order of the first run")
        stack.append(saved["probs"])
    return names, (average(stack) >= threshold).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--seg", nargs="*", default=[], help="method=run or method=run1+run2 (averaged)"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.seg)

    truth_file = np.load("release/train/masks_128.npz")
    truth = dict(zip(truth_file["names"].tolist(), truth_file["masks"], strict=True))
    meta = {
        row["image_name"]: row
        for row in json.loads(Path("release/train/metadata.json").read_text())
    }
    packed: dict[str, list] = {
        "train_names": [],
        "train_masks": [],
        "test_names": [],
        "test_masks": [],
    }
    total, count = 0.0, 0
    for method in METHODS:
        options = {"decode": decode_masks("train", method)}
        if method in runs:
            options["seg"] = seg_masks("train", runs[method], args.threshold)
        chosen = "seg" if method in runs else "decode"
        for label, (names, masks) in options.items():
            target = np.stack([truth[name] for name in names]).astype(np.uint8)
            fold = np.array([held_out(name) for name in names])
            full = np.array([bool(meta[name]["full_watermark"]) for name in names])
            print(
                f"{method:10s} {label:6s} train {mean_miou(masks, target):.4f} | held out "
                f"{mean_miou(masks[fold], target[fold]):.4f} (n={int(fold.sum())}) | seen "
                f"{mean_miou(masks[~fold], target[~fold]):.4f} | held-out full "
                f"{mean_miou(masks[fold & full], target[fold & full]):.4f}, other "
                f"{mean_miou(masks[fold & ~full], target[fold & ~full]):.4f}"
                + ("   <- used" if label == chosen else "")
            )
        names, masks = options[chosen]
        target = np.stack([truth[name] for name in names]).astype(np.uint8)
        total += mean_miou(masks, target) * len(names)
        count += len(names)
        packed["train_names"] += names
        packed["train_masks"].append(masks)
        if chosen == "seg":
            test_names, test_masks = seg_masks("test", runs[method], args.threshold)
        else:
            test_names, test_masks = decode_masks("test", method)
        packed["test_names"] += test_names
        packed["test_masks"].append(test_masks)
    print(f"candidate {args.tag}: train score {total / count:.4f} on {count} images")

    out = Path("output/candidates")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.tag}.npz",
        train_names=np.asarray(packed["train_names"]),
        train_masks=np.packbits(np.concatenate(packed["train_masks"]).astype(bool), axis=None),
        test_names=np.asarray(packed["test_names"]),
        test_masks=np.packbits(np.concatenate(packed["test_masks"]).astype(bool), axis=None),
    )
    print(f"wrote {out / (args.tag + '.npz')}")


if __name__ == "__main__":
    main()
