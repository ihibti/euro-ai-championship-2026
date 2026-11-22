"""Average the weights of several checkpoints of one architecture.

uv run --extra torch python soup.py --models a.pt b.pt c.pt --save out.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def unwrap(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"], "state_dict"
    return obj, None


def average_states(states: list[dict]) -> dict:
    if not states:
        raise ValueError("no checkpoints given")
    keys = list(states[0])
    for state in states[1:]:
        if list(state) != keys:
            raise ValueError("checkpoints do not share the same keys in the same order")
    out = {}
    for key in keys:
        tensors = [state[key] for state in states]
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError(f"{key}: shape or dtype differs between checkpoints")
        if not first.is_floating_point():
            for tensor in tensors[1:]:
                if not torch.equal(tensor, first):
                    raise ValueError(f"{key}: integer tensor differs between checkpoints")
            out[key] = first.clone()
            continue
        mean = torch.stack([tensor.double() for tensor in tensors]).mean(dim=0)
        out[key] = mean.to(first.dtype)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    loaded = [torch.load(path, map_location="cpu") for path in args.models]
    unwrapped = [unwrap(obj) for obj in loaded]
    wrappers = {wrapper for _, wrapper in unwrapped}
    if len(wrappers) != 1:
        raise ValueError("checkpoints are not stored the same way")
    averaged = average_states([state for state, _ in unwrapped])
    wrapper = wrappers.pop()
    if wrapper is None:
        torch.save(averaged, args.save)
    else:
        container = dict(loaded[0])
        container[wrapper] = averaged
        torch.save(container, args.save)
    print(f"wrote {args.save}: mean of {len(loaded)} checkpoints, {len(averaged)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
