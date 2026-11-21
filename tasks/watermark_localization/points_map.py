"""Where a candidate's remaining points are, measured on the held-out fifth of the train split.

Reads a candidate written by assemble.py, the train truth and the full-frame decode results, and
prints: the score on the held-out fifth and on the trained-on images; the share of the remaining
gap by method and kind of edit; how well hole images get their clean region; the decode rule as a
hole detector on train and on test; and a few post-processing rules with a paired bootstrap
interval. CPU only. From the repo root, with the data in data/watermark_localization:

    uv run --extra sci python tasks/watermark_localization/points_map.py --candidate c5
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import ndimage

from euroai.wm_score import miou_per_image

GRID = 128
METHODS = ("pixelseal", "trustmark", "stegastamp")
KINDS = ("hole", "rect", "object")
DECODE_THRESHOLD = {"pixelseal": 0.89, "trustmark": 0.75, "stegastamp": 0.76}
EIGHT = np.ones((3, 3), dtype=int)


def held_out(name: str) -> bool:
    return zlib.crc32(name.encode()) % 5 == 0


def grid_kind(full: bool, grid: np.ndarray) -> str:
    if full:
        return "hole"
    ys, xs = np.nonzero(grid)
    if len(ys) and grid[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].all():
        return "rect"
    return "object"


def pixel_kind(full: bool, grid: np.ndarray) -> str:
    if full:
        return "hole"
    ys, xs = np.nonzero(grid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    inside = np.zeros_like(grid)
    inside[y0:y1, x0:x1] = 1
    corners = {(y0, x0), (y0, x1 - 1), (y1 - 1, x0), (y1 - 1, x1 - 1)}
    missing = {(int(a), int(b)) for a, b in np.argwhere(inside & (1 - grid))}
    return "rect" if missing <= corners and not (grid & (1 - inside)).any() else "object"


def unpack(bits: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(bits)[: count * GRID * GRID].reshape(count, GRID, GRID)


def clean_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p, g = pred == 0, truth == 0
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else 1.0


def component_sizes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask, structure=EIGHT)
    return labels, np.bincount(labels.ravel(), minlength=count + 1)[1:]


def drop_small(mask: np.ndarray, size: int) -> np.ndarray:
    labels, sizes = component_sizes(mask)
    keep = np.concatenate([[False], sizes >= size])
    return keep[labels].astype(np.uint8)


def drop_specks(mask: np.ndarray, size: int) -> np.ndarray:
    return 1 - drop_small(1 - drop_small(mask, size), size)


def keep_largest(mask: np.ndarray) -> np.ndarray:
    labels, sizes = component_sizes(mask)
    if len(sizes) <= 1:
        return mask
    return (labels == 1 + int(np.argmax(sizes))).astype(np.uint8)


def keep_largest_minority(mask: np.ndarray) -> np.ndarray:
    if mask.mean() > 0.5:
        return 1 - keep_largest(1 - mask)
    return keep_largest(mask)


def snap_to_box(mask: np.ndarray, fill: float) -> np.ndarray:
    core = keep_largest(mask)
    ys, xs = np.nonzero(core)
    if not len(ys):
        return mask
    box = np.zeros_like(mask)
    box[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = 1
    return box if core.sum() / box.sum() >= fill else mask


def paired_interval(
    delta: np.ndarray, rng: np.random.Generator, rounds: int = 2000
) -> tuple[float, float]:
    draws = rng.integers(0, len(delta), size=(rounds, len(delta)))
    lo, hi = np.quantile(delta[draws].mean(axis=1), [0.05, 0.95])
    return float(lo), float(hi)


def load(root: Path, candidate: str) -> dict:
    truth = np.load(root / "release/train/masks_128.npz")
    grid = dict(zip(truth["names"].tolist(), truth["masks"].astype(np.uint8), strict=True))
    rows = json.loads((root / "release/train/metadata.json").read_text())
    meta = {r["image_name"]: r for r in rows}
    test_meta = {
        r["image_name"]: r for r in json.loads((root / "release/test/metadata.json").read_text())
    }
    saved = np.load(root / "output/candidates" / f"{candidate}.npz")
    names = saved["train_names"].tolist()
    test_names = saved["test_names"].tolist()
    decode = json.loads((root / "output/global_decode.json").read_text())
    return {
        "names": names,
        "truth": np.stack([grid[n] for n in names]),
        "pred": unpack(saved["train_masks"], len(names)),
        "method": np.array([meta[n]["method"] for n in names]),
        "full": np.array([bool(meta[n]["full_watermark"]) for n in names]),
        "acc": np.array([decode["train"][n]["acc"] for n in names]),
        "test_method": np.array([test_meta[n]["method"] for n in test_names]),
        "test_acc": np.array([decode["test"][n]["acc"] for n in test_names]),
    }


def report(d: dict, candidate: str) -> None:
    names, truth, pred, method = d["names"], d["truth"], d["pred"], d["method"]
    fold = np.array([held_out(n) for n in names])
    kind = np.array([pixel_kind(f, g) for f, g in zip(d["full"], truth, strict=True)])
    strict = np.array([grid_kind(f, g) for f, g in zip(d["full"], truth, strict=True)])
    score = miou_per_image(pred, truth)
    sd = score[fold].std(ddof=1)
    print(
        f"{candidate}: train {score.mean():.4f} | held-out {score[fold].mean():.4f} "
        f"(n={int(fold.sum())}, SE {sd / np.sqrt(fold.sum()):.4f}) | seen {score[~fold].mean():.4f}"
    )
    print(f"standard error at the public split's size (306 images): {sd / np.sqrt(306):.4f}")

    print(
        "\nkinds per method (hole/rect/object): pixel rectangles, then the grid rule of train_seg"
    )
    for m in METHODS:
        pixel = [int(((method == m) & (kind == k)).sum()) for k in KINDS]
        grid = [int(((method == m) & (strict == k)).sum()) for k in KINDS]
        print(f"  {m:10s} pixel {pixel}  grid {grid}")

    cells = [(m, k) for m in METHODS for k in KINDS]
    weight = {
        (m, k): ((method == m) & (kind == k)).sum() / (len(METHODS) * (method == m).sum())
        for m, k in cells
    }
    cell = {(m, k): score[fold & (method == m) & (kind == k)].mean() for m, k in cells}
    missing = {c: weight[c] * (1 - cell[c]) for c in cells}
    total = sum(missing.values())
    by_method = "  ".join(f"{m} {score[fold & (method == m)].mean():.4f}" for m in METHODS)
    print(f"held-out per method: {by_method}")
    print(
        f"held-out reweighted to the test mix (equal methods, train kind shares): {1 - total:.4f}"
    )
    print("\nheld-out fifth by method and kind: n, score, share of the remaining gap at test mix")
    for m, k in cells:
        idx = fold & (method == m) & (kind == k)
        share_of_gap = missing[(m, k)] / total
        print(f"  {m:10s} {k:6s} n={int(idx.sum()):3d}  {cell[(m, k)]:.3f}  {share_of_gap:6.1%}")
    for k in KINDS:
        lost = sum(missing[(m, k)] for m in METHODS)
        print(f"  all        {k:6s} share {lost / total:6.1%}  perfect: {lost:+.4f}")

    print("\nhole images, IoU of the clean region: median held-out | seen, share below 0.1")
    for m in METHODS:
        out = []
        for idx in (fold, ~fold):
            sel = np.flatnonzero(idx & (method == m) & (kind == "hole"))
            ious = np.array([clean_iou(pred[i], truth[i]) for i in sel])
            out.append(f"{np.median(ious):.2f} ({(ious < 0.1).mean():.0%})")
        print(f"  {m:10s} {out[0]} | {out[1]}")

    print("\ntruth: share of images with two or more regions (clean ones for holes)")
    for k in KINDS:
        counts = np.array(
            [
                ndimage.label(1 - truth[i] if k == "hole" else truth[i], structure=EIGHT)[1]
                for i in np.flatnonzero(kind == k)
            ]
        )
        print(f"  {k:6s} {(counts >= 2).mean():.0%}, at most {counts.max()}")

    threshold = np.array([DECODE_THRESHOLD[m] for m in method])
    flag = d["acc"] >= threshold
    hole = kind == "hole"
    print("\nfull-frame decode rule as a hole detector, train (no training involved)")
    for m in METHODS:
        idx = method == m
        tp, fp = int((flag & hole & idx).sum()), int((flag & ~hole & idx).sum())
        fn = int((~flag & hole & idx).sum())
        test = d["test_method"] == m
        test_flag = (d["test_acc"] >= DECODE_THRESHOLD[m]) & test
        print(
            f"  {m:10s} tp {tp} fp {fp} fn {fn} of {int((hole & idx).sum())} holes; "
            f"test flagged {test_flag.sum() / test.sum():.1%}"
        )

    share = pred.reshape(len(names), -1).mean(1)
    rules: dict[str, Callable[[int], np.ndarray]] = {
        "decode flag overrides a small mask": lambda i: (
            np.ones_like(pred[i]) if flag[i] and share[i] <= 0.5 else pred[i]
        ),
        "drop specks under 4 cells": lambda i: drop_specks(pred[i], 4),
        "drop specks under 16 cells": lambda i: drop_specks(pred[i], 16),
        "drop specks under 64 cells": lambda i: drop_specks(pred[i], 64),
        "keep the largest blob": lambda i: keep_largest_minority(pred[i]),
        "snap a non-hole blob to its box, fill 0.8": lambda i: (
            pred[i] if flag[i] or share[i] > 0.5 else snap_to_box(pred[i], 0.8)
        ),
    }
    rng = np.random.default_rng(0)
    print("\npost-processing rules, change on the held-out fifth with a 90% paired interval")
    for label, rule in rules.items():
        delta = miou_per_image(np.stack([rule(i) for i in range(len(names))]), truth) - score
        lo, hi = paired_interval(delta[fold], rng)
        print(f"  {label:42s} {delta[fold].mean():+.4f} ({lo:+.4f} .. {hi:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/watermark_localization"))
    parser.add_argument("--candidate", default="c5")
    args = parser.parse_args()
    report(load(args.root, args.candidate), args.candidate)


if __name__ == "__main__":
    main()
