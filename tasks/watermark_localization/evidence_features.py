"""Turn the arrays written by decode_evidence.py into per-cell feature maps on the 128 x 128 grid.

A window statistic becomes a map by giving every cell the mean of the windows that cover it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

GRID = 128
CELL = 4
WINDOW_STATS = ("acc", "margin", "det")


def coverage(origins: np.ndarray, size: int) -> np.ndarray:
    cover = np.zeros((len(origins), GRID, GRID), dtype=np.float32)
    span = size // CELL
    for k, (y, x) in enumerate(np.asarray(origins) // CELL):
        cover[k, y : y + span, x : x + span] = 1.0
    return cover


def dense_map(values: np.ndarray, cover: np.ndarray) -> np.ndarray:
    flat = cover.reshape(len(cover), -1)
    summed = values.astype(np.float32) @ flat
    return (summed / np.maximum(flat.sum(0), 1.0)).reshape(len(values), GRID, GRID)


def window_labels(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, int]]:
    labels = []
    for key in arrays:
        if key.endswith("_origins"):
            label = key[: -len("_origins")]
            labels.append((label, int("".join(c for c in label if c.isdigit()))))
    return labels


def feature_stack(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    maps, names = [], []
    for label, size in window_labels(arrays):
        cover = coverage(arrays[f"{label}_origins"], size)
        for stat in WINDOW_STATS:
            key = f"{label}_{stat}"
            if key in arrays:
                maps.append(dense_map(arrays[key], cover))
                names.append(key)
    for key, value in arrays.items():
        if key.startswith("p") and value.ndim == 3 and value.shape[1:] == (GRID, GRID):
            maps.append(value.astype(np.float32))
            names.append(key)
    return np.stack(maps, 1), names


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [scores.size]))
    mean_ranks = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(mean_ranks, ends - starts)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def miou_image(pred: np.ndarray, truth: np.ndarray) -> float:
    total = 0.0
    for value in (0, 1):
        union = np.logical_or(pred == value, truth == value).sum()
        total += 1.0 if union == 0 else np.logical_and(pred == value, truth == value).sum() / union
    return total / 2


def mean_miou(preds: np.ndarray, truths: np.ndarray) -> float:
    return float(np.mean([miou_image(p, t) for p, t in zip(preds, truths, strict=True)]))
