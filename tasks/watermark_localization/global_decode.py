"""First real baseline: one decision per image from the full-frame decode of the known message.

An image whose message reads well is almost fully watermarked, so predict all ones; otherwise
predict all zeros. The threshold per method maximizes the official score on the train split.

    uv run --extra torch python tasks/watermark_localization/global_decode.py

Reads data/watermark_localization/output/global_decode.json (written on JURECA from the evidence
files) and writes global_decode_train.npz and global_decode_test.pt next to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

GRID = 128
ROOT = Path("data/watermark_localization")
THRESHOLDS = np.arange(0.5, 1.0, 0.01)


def all_or_nothing_score(marked: np.ndarray, share: np.ndarray) -> float:
    return float(np.mean(np.where(marked, share, 1.0 - share) / 2.0))


def best_threshold(acc: np.ndarray, share: np.ndarray) -> tuple[float, float]:
    scored = [(all_or_nothing_score(acc >= t, share), float(t)) for t in THRESHOLDS]
    return max(scored)


def main() -> None:
    decoded = json.loads((ROOT / "output" / "global_decode.json").read_text())
    truth = np.load(ROOT / "release" / "train" / "masks_128.npz")
    share = dict(
        zip(
            truth["names"].tolist(),
            truth["masks"].reshape(len(truth["masks"]), -1).mean(1),
            strict=True,
        )
    )
    thresholds = {}
    for method in sorted({row["method"] for row in decoded["train"].values()}):
        names = [name for name, row in decoded["train"].items() if row["method"] == method]
        acc = np.array([decoded["train"][name]["acc"] for name in names])
        score, threshold = best_threshold(acc, np.array([share[name] for name in names]))
        thresholds[method] = threshold
        print(f"{method}: all ones when bit accuracy >= {threshold:.2f}; train score {score:.4f}")

    for split, rows in decoded.items():
        names = list(rows)
        marked = np.array([rows[name]["acc"] >= thresholds[rows[name]["method"]] for name in names])
        masks = np.repeat(marked[:, None, None], GRID, 1).repeat(GRID, 2)
        print(f"{split}: {len(names)} images, {marked.mean():.3f} predicted fully watermarked")
        if split == "train":
            np.savez_compressed(
                ROOT / "output" / "global_decode_train.npz",
                image_names=np.asarray(names),
                masks=masks.astype(np.uint8),
            )
        else:
            torch.save(
                {"image_names": names, "masks": torch.from_numpy(masks)},
                ROOT / "output" / "global_decode_test.pt",
            )


if __name__ == "__main__":
    main()
