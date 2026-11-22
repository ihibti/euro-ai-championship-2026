import pytest
import torch

from soup import average_states


def test_mean_of_three_float_tensors():
    states = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, 4.0])}, {"w": torch.tensor([5.0, 9.0])}]
    out = average_states(states)
    assert torch.allclose(out["w"], torch.tensor([3.0, 5.0]))
    assert out["w"].dtype == torch.float32


def test_identical_checkpoints_are_unchanged():
    state = {"w": torch.randn(4, 3), "b": torch.randn(4)}
    out = average_states([state, {k: v.clone() for k, v in state.items()}])
    assert all(torch.equal(out[k], state[k]) for k in state)


def test_integer_tensors_must_agree():
    out = average_states([{"n": torch.tensor(7)}, {"n": torch.tensor(7)}])
    assert out["n"].item() == 7
    with pytest.raises(ValueError):
        average_states([{"n": torch.tensor(7)}, {"n": torch.tensor(8)}])


def test_key_or_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        average_states([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}])
    with pytest.raises(ValueError):
        average_states([{"w": torch.zeros(2)}, {"w": torch.zeros(3)}])
    with pytest.raises(ValueError):
        average_states([])
