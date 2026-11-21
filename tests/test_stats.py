import numpy as np
import pytest

from euroai.stats import (
    bootstrap_ci,
    gram_matrix,
    mmd2_unbiased,
    mmd_permutation_test,
    permutation_pvalue,
    permutation_test,
)


def test_permutation_pvalue_plus_one_correction():
    assert permutation_pvalue(5.0, [1, 2, 3]) == 0.25
    assert permutation_pvalue(0.0, [1, 2, 3]) == 1.0


def test_mmd2_unbiased_hand_computed_linear_kernel():
    x = np.array([[0.0], [1.0]])
    y = np.array([[2.0], [3.0]])
    k = lambda a, b: float(a @ b)  # noqa: E731
    value = mmd2_unbiased(gram_matrix(x, x, k), gram_matrix(y, y, k), gram_matrix(x, y, k))
    off_diag_xx, off_diag_yy, mean_xy = 0.0 / 2, 12.0 / 2, 5.0 / 4
    assert value == pytest.approx(off_diag_xx + off_diag_yy - 2.0 * mean_xy)


def test_mmd2_unbiased_rejects_tiny_groups():
    with pytest.raises(ValueError):
        mmd2_unbiased(np.ones((1, 1)), np.ones((2, 2)), np.ones((1, 2)))


def _rbf(points):
    d2 = (points[:, None, :] - points[None, :, :]) ** 2
    return np.exp(-d2.sum(-1) / 2.0)


def test_mmd_permutation_test_detects_shift_and_controls_null():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (60, 2))
    shifted = rng.normal(1.0, 1, (60, 2))
    same = rng.normal(0, 1, (60, 2))
    alt = mmd_permutation_test(_rbf(np.vstack([x, shifted])), 60, n_permutations=200, rng=rng)
    null = mmd_permutation_test(_rbf(np.vstack([x, same])), 60, n_permutations=200, rng=rng)
    assert alt.pvalue < 0.01
    assert null.pvalue > 0.05


def test_permutation_test_mean_difference():
    rng = np.random.default_rng(1)
    result = permutation_test(
        rng.normal(1, 1, 50),
        rng.normal(0, 1, 50),
        lambda a, b: a.mean() - b.mean(),
        n_permutations=300,
        rng=rng,
    )
    assert result.pvalue < 0.01
    assert result.null_statistics.shape == (300,)


def test_bootstrap_ci_contains_estimate_and_respects_strata():
    rng = np.random.default_rng(2)
    values = rng.normal(5, 1, 400)
    res = bootstrap_ci(np.mean, values, n_resamples=300, rng=rng)
    assert res.low < res.estimate < res.high
    assert res.n_valid == 300

    labels = np.array([1] * 3 + [0] * 97)
    seen_counts = set()

    def positives(y):
        seen_counts.add(int(y.sum()))
        return float(y.mean())

    bootstrap_ci(positives, labels, n_resamples=50, rng=rng, strata=labels)
    assert seen_counts == {3}
