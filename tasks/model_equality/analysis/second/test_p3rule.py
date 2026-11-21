import numpy as np
import p3rule as r


def test_run_length_counts_the_longest_run_of_river():
    assert r.run_length("River river, river. sea river") == 3
    assert r.is_long(" ".join(["river"] * 128))
    assert not r.is_long("river")


def test_call_family_splits_off_the_high_rate_minority_as_singletons():
    names = [f"e{i}" for i in range(9)]
    long_counts = dict(zip(names, [0, 1, 2, 1, 0, 3, 8, 7, 9], strict=True))
    totals = dict.fromkeys(names, 10)
    call = r.call_family(names, long_counts, totals, np.random.default_rng(0), draws=199)
    assert call.p < 0.01
    assert call.lora == ["e6", "e7", "e8"]
    assert call.base == ["e0", "e1", "e2", "e3", "e4", "e5"]


def test_call_family_calls_nothing_on_a_homogeneous_family():
    names = [f"e{i}" for i in range(8)]
    long_counts = dict(zip(names, [1, 2, 1, 0, 2, 1, 1, 2], strict=True))
    totals = dict.fromkeys(names, 10)
    call = r.call_family(names, long_counts, totals, np.random.default_rng(0), draws=199)
    assert call.p >= 0.01
    assert call.lora == []
