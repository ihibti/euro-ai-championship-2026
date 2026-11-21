import math

import numpy as np
import pytest

from euroai.metrics import (
    average_ranks,
    best_threshold_accuracy,
    binomial_tail_pvalue,
    bit_accuracy,
    psnr,
    roc_auc,
    roc_curve,
    tpr_at_fpr,
)


def test_average_ranks_handles_ties():
    np.testing.assert_allclose(average_ranks([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


def test_auc_perfect_inverse_and_ties():
    y = [0, 0, 1, 1]
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)
    pos, neg = s[y == 1], s[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert roc_auc(y, s) == pytest.approx(pairwise)


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.3])


def test_roc_curve_endpoints_and_monotonicity():
    curve = roc_curve([0, 1, 0, 1, 1], [0.1, 0.4, 0.35, 0.8, 0.4])
    assert (curve.fpr[0], curve.tpr[0]) == (0.0, 0.0)
    assert (curve.fpr[-1], curve.tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(curve.fpr) >= 0) and np.all(np.diff(curve.tpr) >= 0)


def test_tpr_at_fpr_hand_computed():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.95, 0.99]
    assert tpr_at_fpr(y, s, 0.0) == 0.5
    assert tpr_at_fpr(y, s, 0.25) == 1.0


def test_best_threshold_accuracy():
    assert best_threshold_accuracy([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert best_threshold_accuracy([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_bit_accuracy_scalar_and_batched():
    assert bit_accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    np.testing.assert_allclose(bit_accuracy([[1, 0], [0, 0]], [[1, 0], [1, 1]]), [1.0, 0.0])
    with pytest.raises(ValueError):
        bit_accuracy([2, 0], [1, 0])


def test_binomial_tail_matches_pixel_seal_eq2_ceiling():
    n, hamming = 256, 0
    assert -math.log10(binomial_tail_pvalue(n - hamming, n)) == pytest.approx(77.06, abs=0.01)
    assert binomial_tail_pvalue(0, n) == 1.0


def test_binomial_tail_general_p_matches_half_case():
    assert binomial_tail_pvalue(7, 10, p=0.5) == pytest.approx(176 / 1024)
    assert binomial_tail_pvalue(3, 3, p=0.2) == pytest.approx(0.008)


def test_psnr():
    x = np.zeros((4, 4))
    assert psnr(x, x, data_range=1.0) == math.inf
    assert psnr(x, x + 0.1, data_range=1.0) == pytest.approx(20.0)
