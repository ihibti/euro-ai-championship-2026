import random

from choose import candidate_groups, mcc, pair_counts, sample_truth


def test_rule_on_six_same_five_changed_matches_known_value():
    truth = {f"e{i}": i < 6 for i in range(11)}
    group = frozenset(f"e{i}" for i in range(5))
    counts = pair_counts(truth, group)
    assert counts == (10, 0, 5, 40)
    assert abs(mcc(*counts) - 0.770) < 0.001


def test_wrong_plain_matches_known_value():
    truth = {f"e{i}": i < 6 for i in range(11)}
    group = frozenset([f"e{i}" for i in range(5)] + ["e10"])
    assert abs(mcc(*pair_counts(truth, group)) - 0.542) < 0.001


def test_perfect_grouping_scores_one_and_empty_denominator_zero():
    truth = {f"e{i}": i < 6 for i in range(11)}
    assert mcc(*pair_counts(truth, frozenset(f"e{i}" for i in range(6)))) == 1.0
    assert mcc(0, 0, 5, 50) == 0.0


def test_sample_truth_has_one_plain_and_respects_min_changed():
    family = {"labelled": {"a": 1.0, "b": 0.5}, "pool": {"c": 1.0, "d": 1.0, "e": 1.0}, "min_changed": 2}
    rng = random.Random(0)
    for _ in range(200):
        truth = sample_truth(family, rng)
        assert truth["a"] is True
        assert sum(truth[n] for n in "cde") == 1
        assert sum(1 for same in truth.values() if not same) >= 2


def test_candidate_groups_keep_sure_labels():
    family = {"labelled": {"a": 0.95, "b": 0.6}, "pool": {"c": 1.0}, "pool_candidates": ["c"]}
    groups = candidate_groups(family)
    assert all("a" in group for group in groups)
    assert len(groups) == 4
