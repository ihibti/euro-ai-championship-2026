import random

import numpy as np
import pytest

from euroai.seeding import MAX_SEED, derive_seed, make_rng, seed_everything


def test_derive_seed_is_stable_and_label_sensitive():
    assert derive_seed(7, "fold", 0) == derive_seed(7, "fold", 0)
    assert derive_seed(7, "fold", 0) != derive_seed(7, "fold", 1)
    assert derive_seed(7, "fold", 0) != derive_seed(8, "fold", 0)
    assert 0 <= derive_seed(7, "x") <= MAX_SEED


def test_derive_seed_known_values_guard_against_silent_changes():
    assert derive_seed(0) == 1726737503
    assert derive_seed(1234, "task", "synthetic_detection") == 4055068706


@pytest.mark.parametrize("bad", [-1, MAX_SEED + 1])
def test_out_of_range_seed_rejected(bad):
    with pytest.raises(ValueError):
        make_rng(bad)


@pytest.mark.parametrize("bad", [1.5, "3", True])
def test_non_int_seed_rejected(bad):
    with pytest.raises(TypeError):
        derive_seed(bad)


def test_seed_everything_makes_global_rngs_reproducible():
    seed_everything(42)
    first = (random.random(), np.random.rand())
    seed_everything(42)
    second = (random.random(), np.random.rand())
    assert first == second


def test_make_rng_reproducible():
    assert make_rng(3).integers(0, 10**9) == make_rng(3).integers(0, 10**9)


@pytest.mark.torch
def test_seed_everything_seeds_torch():
    torch = pytest.importorskip("torch")
    report = seed_everything(5)
    a = torch.rand(3)
    seed_everything(5)
    assert report.torch_seeded
    assert torch.equal(a, torch.rand(3))
