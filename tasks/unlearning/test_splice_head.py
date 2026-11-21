"""Checks of splice_head.py on a small state dict.

python -m pytest tasks/unlearning/test_splice_head.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import splice_head  # noqa: E402


def _state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "blocks.0.weight": torch.randn(4, 4, generator=generator),
        "head.weight": torch.randn(100, 8, generator=generator),
        "head.bias": torch.randn(100, generator=generator),
    }


def test_shift_lowers_only_the_named_classes():
    state = _state()
    out = splice_head.shift_classes(state, (29, 30, 31, 33), 15.0)
    delta = state["head.bias"] - out["head.bias"]
    assert torch.allclose(delta[[29, 30, 31, 33]], torch.full((4,), 15.0))
    others = [c for c in range(100) if c not in (29, 30, 31, 33)]
    assert torch.equal(delta[others], torch.zeros(96))
    assert torch.equal(out["head.weight"], state["head.weight"])
    assert torch.equal(out["blocks.0.weight"], state["blocks.0.weight"])


def test_shift_does_not_modify_the_input_state():
    state = _state()
    before = state["head.bias"].clone()
    splice_head.shift_classes(state, splice_head.NINE_CLASSES, 15.0)
    assert torch.equal(state["head.bias"], before)


def test_scale_divides_every_logit():
    state = _state()
    out = splice_head.scale_head(state, 4.0)
    features = torch.randn(5, 8)
    original = features @ state["head.weight"].T + state["head.bias"]
    scaled = features @ out["head.weight"].T + out["head.bias"]
    assert torch.allclose(scaled * 4.0, original, atol=1e-5)


def test_splice_rejects_a_wrong_shape_and_non_finite_values():
    state = _state()
    with pytest.raises(ValueError):
        splice_head.splice(state, np.zeros((99, 8)), np.zeros(100))
    weight = np.zeros((100, 8))
    weight[0, 0] = np.nan
    with pytest.raises(ValueError):
        splice_head.splice(state, weight, np.zeros(100))


def test_parse_classes_defaults_to_the_nine_and_sorts():
    assert splice_head.parse_classes(None) == splice_head.NINE_CLASSES
    assert splice_head.parse_classes("33,29,31,30") == (29, 30, 31, 33)
    with pytest.raises(ValueError):
        splice_head.parse_classes("100")


def test_main_writes_a_boats_dial(tmp_path):
    model = tmp_path / "model.pt"
    torch.save(_state(), model)
    save = tmp_path / "boats15.pt"
    argv = ["--model", str(model), "--shift", "15", "--classes", "29,30,31,33", "--save", str(save)]
    splice_head.main(argv)
    out = torch.load(save, weights_only=True)
    delta = _state()["head.bias"] - out["head.bias"]
    assert float(delta.sum()) == pytest.approx(60.0, abs=1e-4)
