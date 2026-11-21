"""Two-sample testing and resampling utilities (NumPy only)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    null_statistics: np.ndarray


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    confidence: float
    n_valid: int


def permutation_pvalue(observed: float, null_statistics: ArrayLike) -> float:
    """Right-tailed Monte Carlo p-value with the +1 correction: (1 + #{T* >= T}) / (1 + B)."""
    null = np.asarray(null_statistics, dtype=float)
    return float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))


def permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> TestResult:
    """Right-tailed permutation test of `statistic(x, y)` under exchangeability."""
    xs = np.asarray(x)
    ys = np.asarray(y)
    pooled = np.concatenate([xs, ys])
    m = xs.shape[0]
    observed = float(statistic(xs, ys))
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        null[b] = statistic(pooled[perm[:m]], pooled[perm[m:]])
    return TestResult(observed, permutation_pvalue(observed, null), null)


def mmd2_unbiased(k_xx: ArrayLike, k_yy: ArrayLike, k_xy: ArrayLike) -> float:
    """Unbiased U-statistic estimate of MMD^2 from kernel matrices (Gretton et al., 2012)."""
    kxx = np.asarray(k_xx, dtype=float)
    kyy = np.asarray(k_yy, dtype=float)
    kxy = np.asarray(k_xy, dtype=float)
    m, n = kxx.shape[0], kyy.shape[0]
    if kxx.shape != (m, m) or kyy.shape != (n, n) or kxy.shape != (m, n):
        raise ValueError("kernel matrices have inconsistent shapes")
    if m < 2 or n < 2:
        raise ValueError("need at least two samples per group")
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(term_x + term_y - 2.0 * kxy.mean())


def mmd2_from_pooled(kernel: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    return mmd2_unbiased(
        kernel[np.ix_(idx_x, idx_x)],
        kernel[np.ix_(idx_y, idx_y)],
        kernel[np.ix_(idx_x, idx_y)],
    )


def mmd_permutation_test(
    pooled_kernel: ArrayLike,
    m: int,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> TestResult:
    """MMD^2 permutation test; rows/cols `[:m]` are sample X and `[m:]` are sample Y."""
    kernel = np.asarray(pooled_kernel, dtype=float)
    total = kernel.shape[0]
    if kernel.shape != (total, total):
        raise ValueError("pooled_kernel must be square")
    if not 2 <= m <= total - 2:
        raise ValueError("each group needs at least two samples")
    idx = np.arange(total)
    observed = mmd2_from_pooled(kernel, idx[:m], idx[m:])
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        perm = rng.permutation(total)
        null[b] = mmd2_from_pooled(kernel, perm[:m], perm[m:])
    return TestResult(observed, permutation_pvalue(observed, null), null)


def gram_matrix(
    xs: Sequence[object], ys: Sequence[object], kernel: Callable[[object, object], float]
) -> np.ndarray:
    return np.array([[kernel(a, b) for b in ys] for a in xs], dtype=float)


def bootstrap_ci(
    statistic: Callable[..., float],
    *arrays: ArrayLike,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator,
    strata: ArrayLike | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI; arrays are resampled jointly, optionally within strata.

    Resamples where the statistic is non-finite are discarded and counted in
    `n_valid`. Stratify on the label when the statistic needs both classes (AUC).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    data = [np.asarray(a) for a in arrays]
    if not data:
        raise ValueError("need at least one array")
    n = data[0].shape[0]
    if any(a.shape[0] != n for a in data):
        raise ValueError("arrays must share their first dimension")
    groups = (
        [np.arange(n)]
        if strata is None
        else [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(np.asarray(strata))]
    )
    estimate = float(statistic(*data))
    values = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        value = float(statistic(*(a[idx] for a in data)))
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("statistic was non-finite on every resample")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return BootstrapResult(estimate, float(low), float(high), confidence, len(values))
