"""Average the weights of fine-tunes of one model, then optionally rescale the last layer.

Every input must have the same keys and shapes. The inputs are read one at a time and summed in
float32, so memory holds two state dicts whatever their number. With --scale t, the last layer of
the average is divided by t (`splice_head.scale_head`): --scale 0.5 multiplies it by 2. CPU; run
in the unlearning task folder:

    python <repo>/tasks/unlearning/soup.py --models output/kkrgp/a.pt output/kkrgp/b.pt \\
        --scale 0.5 --save output/kkrgp/soup_x2.pt
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path

import torch
from splice_head import scale_head


def average(states: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    total: dict[str, torch.Tensor] = {}
    count = 0
    for state in states:
        if not total:
            total = {key: value.detach().float().clone() for key, value in state.items()}
        else:
            if state.keys() != total.keys():
                raise ValueError("the state dicts have different keys")
            for key, value in state.items():
                if value.shape != total[key].shape:
                    raise ValueError(f"{key}: shape {tuple(value.shape)} differs")
                total[key] += value.detach().float()
        count += 1
    if count == 0:
        raise ValueError("no state dict to average")
    return {key: value / count for key, value in total.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--scale", type=float, help="divide the averaged last layer by this")
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    states = (torch.load(p, map_location="cpu", weights_only=True) for p in args.models)
    out = average(states)
    if args.scale is not None:
        out = scale_head(out, args.scale)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.save)
    digest = hashlib.sha256(args.save.read_bytes()).hexdigest()
    print(f"{digest}  {args.save.name}  mean of {len(args.models)}")


if __name__ == "__main__":
    main()
