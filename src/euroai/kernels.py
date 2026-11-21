"""Sequence kernels for sample-based two-sample tests.

`prompt_hamming_gram` follows the released code of Gao et al. (2025, arXiv
2410.20247): k((x, y), (x', y')) = 1{x = x'} * (1/L) * sum_i 1{y_i = y'_i},
with completions truncated/right-padded to length L and PAD == PAD counted as
a match. Memory is O(N^2): the Gram matrix is accumulated position by position.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

PAD = -1


def encode_unicode(text: str) -> list[int]:
    return [ord(char) for char in text]


def pad_truncate(sequences: Iterable[Sequence[int]], length: int, pad: int = PAD) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be >= 1")
    rows = list(sequences)
    out = np.full((len(rows), length), pad, dtype=np.int64)
    for i, seq in enumerate(rows):
        clipped = list(seq)[:length]
        out[i, : len(clipped)] = clipped
    return out


def hamming_gram(a: ArrayLike, b: ArrayLike | None = None) -> np.ndarray:
    """Normalised Hamming similarity between rows of two padded integer matrices."""
    left = np.asarray(a)
    right = left if b is None else np.asarray(b)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("inputs must be 2-D with the same number of columns")
    gram = np.zeros((left.shape[0], right.shape[0]), dtype=float)
    for position in range(left.shape[1]):
        gram += left[:, position, None] == right[None, :, position]
    return gram / left.shape[1]


def prompt_hamming_gram(
    prompt_ids: ArrayLike,
    padded: ArrayLike,
    other_prompt_ids: ArrayLike | None = None,
    other_padded: ArrayLike | None = None,
) -> np.ndarray:
    """Hamming Gram matrix masked to pairs that share the same prompt id."""
    if (other_prompt_ids is None) != (other_padded is None):
        raise ValueError("pass both other_prompt_ids and other_padded, or neither")
    p_left = np.asarray(prompt_ids)
    p_right = p_left if other_prompt_ids is None else np.asarray(other_prompt_ids)
    same_prompt = p_left[:, None] == p_right[None, :]
    return hamming_gram(padded, other_padded) * same_prompt
