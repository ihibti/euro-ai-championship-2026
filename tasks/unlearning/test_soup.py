"""Checks of soup.py on small state dicts.

python -m pytest tasks/unlearning/test_soup.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import soup  # noqa: E402


def _state(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "block.weight": torch.randn(4, 3, generator=g),
        "head.weight": torch.randn(5, 3, generator=g),
        "head.bias": torch.randn(5, generator=g),
    }


def test_average_is_the_elementwise_mean():
    states = [_state(s) for s in range(3)]
    out = soup.average(iter(states))
    for key in states[0]:
        assert torch.allclose(out[key], torch.stack([s[key] for s in states]).mean(0))


def test_average_of_one_is_a_copy():
    state = _state(0)
    out = soup.average([state])
    out["head.bias"] += 1
    assert torch.equal(state["head.bias"], _state(0)["head.bias"])


def test_average_refuses_mismatched_inputs():
    other = _state(1)
    other["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        soup.average([_state(0), other])
    reshaped = _state(1)
    reshaped["head.bias"] = torch.zeros(6)
    with pytest.raises(ValueError):
        soup.average([_state(0), reshaped])
    with pytest.raises(ValueError):
        soup.average([])


def test_main_writes_the_scaled_average(tmp_path):
    paths = []
    for s in range(2):
        paths.append(tmp_path / f"m{s}.pt")
        torch.save(_state(s), paths[-1])
    save = tmp_path / "soup_x2.pt"
    soup.main(["--models", *map(str, paths), "--scale", "0.5", "--save", str(save)])
    out = torch.load(save, weights_only=True)
    mean = soup.average([_state(0), _state(1)])
    assert torch.allclose(out["block.weight"], mean["block.weight"])
    assert torch.allclose(out["head.weight"], 2 * mean["head.weight"])
    assert torch.allclose(out["head.bias"], 2 * mean["head.bias"])
