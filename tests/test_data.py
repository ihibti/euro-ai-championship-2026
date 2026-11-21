import numpy as np
import pytest

from euroai.data import allocate_counts, split_indices, stratified_split_indices


def test_allocate_counts_sums_to_n():
    counts = allocate_counts(10, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert sum(counts.values()) == 10
    assert counts["train"] == 7


def test_allocate_counts_validates():
    with pytest.raises(ValueError):
        allocate_counts(10, {"a": 0.5, "b": 0.6})
    with pytest.raises(ValueError):
        allocate_counts(10, {})


def test_split_indices_disjoint_exhaustive_and_seeded():
    parts = split_indices(101, {"train": 0.8, "test": 0.2}, seed=3)
    union = np.concatenate(list(parts.values()))
    assert sorted(union.tolist()) == list(range(101))
    assert len(np.intersect1d(parts["train"], parts["test"])) == 0
    again = split_indices(101, {"train": 0.8, "test": 0.2}, seed=3)
    np.testing.assert_array_equal(parts["test"], again["test"])
    other = split_indices(101, {"train": 0.8, "test": 0.2}, seed=4)
    assert not np.array_equal(parts["test"], other["test"])


def test_stratified_split_preserves_class_ratio():
    labels = np.array([0] * 80 + [1] * 20)
    parts = stratified_split_indices(labels, {"member": 0.5, "nonmember": 0.5}, seed=0)
    for idx in parts.values():
        assert (labels[idx] == 1).sum() == 10
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(100))
