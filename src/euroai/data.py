"""Deterministic index splits (train/val/test, member/non-member, forget/retain, ...)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from euroai.seeding import derive_seed


def allocate_counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Split `n` items by `fractions` (summing to 1) using largest-remainder rounding."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not fractions:
        raise ValueError("fractions must not be empty")
    if any(f < 0 for f in fractions.values()):
        raise ValueError("fractions must be non-negative")
    if not np.isclose(sum(fractions.values()), 1.0, atol=1e-9):
        raise ValueError("fractions must sum to 1")
    names = list(fractions)
    exact = np.array([fractions[name] * n for name in names])
    counts = np.floor(exact).astype(int)
    remainder = n - counts.sum()
    order = np.argsort(-(exact - counts), kind="mergesort")
    counts[order[:remainder]] += 1
    return dict(zip(names, counts.tolist(), strict=True))


def split_indices(n: int, fractions: Mapping[str, float], seed: int) -> dict[str, np.ndarray]:
    """Disjoint, exhaustive, seeded split of `range(n)`; each part is sorted."""
    counts = allocate_counts(n, fractions)
    perm = np.random.default_rng(derive_seed(seed, "split_indices", n)).permutation(n)
    out, start = {}, 0
    for name, count in counts.items():
        out[name] = np.sort(perm[start : start + count])
        start += count
    return out


def stratified_split_indices(
    labels: ArrayLike, fractions: Mapping[str, float], seed: int
) -> dict[str, np.ndarray]:
    """Like `split_indices`, applied within each label value."""
    y = np.asarray(labels)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in fractions}
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        local = split_indices(members.size, fractions, derive_seed(seed, "stratum", label))
        for name, idx in local.items():
            parts[name].append(members[idx])
    return {name: np.sort(np.concatenate(chunks)) for name, chunks in parts.items()}
