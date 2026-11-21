"""NumPy-only evaluation metrics shared by detection-style experiments.

Binary detection framing covers membership inference, watermark detection and
"same model / different model" decisions alike: a higher score means "more
likely positive". Conventions follow scikit-learn (`y_true`, `y_score`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RocCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def _binary_inputs(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("y_true and y_score must be 1-D arrays of equal length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must be finite")
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        raise ValueError("y_true must contain both classes")
    return labels, scores


def average_ranks(values: ArrayLike) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    boundaries = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [x.size]))
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count 1/2)."""
    labels, scores = _binary_inputs(y_true, y_score)
    ranks = average_ranks(scores)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    u_stat = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> RocCurve:
    """ROC points for the rule `score >= threshold`, starting at (0, 0)."""
    labels, scores = _binary_inputs(y_true, y_score)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    last_of_group = np.concatenate(
        (np.flatnonzero(np.diff(sorted_scores)), [sorted_scores.size - 1])
    )
    true_pos = np.cumsum(sorted_labels)[last_of_group]
    false_pos = last_of_group + 1 - true_pos
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    return RocCurve(
        fpr=np.concatenate(([0.0], false_pos / n_neg)),
        tpr=np.concatenate(([0.0], true_pos / n_pos)),
        thresholds=np.concatenate(([np.inf], sorted_scores[last_of_group])),
    )


def tpr_at_fpr(y_true: ArrayLike, y_score: ArrayLike, max_fpr: float) -> float:
    """Highest achievable TPR at an empirical FPR <= `max_fpr` (no interpolation)."""
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    curve = roc_curve(y_true, y_score)
    return float(curve.tpr[curve.fpr <= max_fpr + 1e-12].max())


def best_threshold_accuracy(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Best accuracy of any threshold rule, e.g. a loss-threshold membership attack."""
    labels, _ = _binary_inputs(y_true, y_score)
    curve = roc_curve(y_true, y_score)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    accuracy = (curve.tpr * n_pos + (1.0 - curve.fpr) * n_neg) / labels.size
    return float(accuracy.max())


def bit_accuracy(pred_bits: ArrayLike, true_bits: ArrayLike) -> np.ndarray | float:
    """Fraction of matching bits along the last axis (scalar for 1-D input)."""
    pred = np.asarray(pred_bits)
    true = np.asarray(true_bits)
    if pred.shape != true.shape or pred.ndim == 0:
        raise ValueError("pred_bits and true_bits must have the same non-scalar shape")
    for name, bits in (("pred_bits", pred), ("true_bits", true)):
        if not np.isin(bits, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0/1 values")
    matches = (pred.astype(int) == true.astype(int)).mean(axis=-1)
    return float(matches) if pred.ndim == 1 else matches


def binomial_tail_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """P[X >= k] for X ~ Binomial(n, p); exact rational arithmetic when p = 0.5."""
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p == 0.5:
        return float(Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2**n))
    return min(1.0, math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def psnr(reference: ArrayLike, distorted: ArrayLike, data_range: float) -> float:
    """Peak signal-to-noise ratio in dB; `inf` for identical inputs."""
    ref = np.asarray(reference, dtype=float)
    dist = np.asarray(distorted, dtype=float)
    if ref.shape != dist.shape:
        raise ValueError("inputs must have the same shape")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = float(np.mean((ref - dist) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range**2 / mse)
