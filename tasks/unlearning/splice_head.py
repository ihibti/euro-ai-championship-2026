"""Write a submission state dict: the original model with its last layer replaced or rescaled.

head_fit.py writes `weight` [100, 768] and `bias` [100]. With --head, this loads model.pt, replaces
`head.weight` and `head.bias`, checks shapes and finiteness, and saves float32. With --scale t, it
divides the last layer by t instead: every logit is divided by t, the predicted class never
changes, so accuracy and the utility gate are those of the original. CPU, seconds; run on a JURECA
login node in the unlearning task folder:

    python <repo>/tasks/unlearning/splice_head.py --head <repo>/tasks/unlearning/heads/<name>.npz \
        --save output/kkrgp/<name>.pt
    python <repo>/tasks/unlearning/splice_head.py --scale 3 --save output/kkrgp/scale_t3.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def splice(state: dict[str, torch.Tensor], weight: np.ndarray, bias: np.ndarray) -> dict:
    new_weight = torch.from_numpy(np.asarray(weight, dtype=np.float32))
    new_bias = torch.from_numpy(np.asarray(bias, dtype=np.float32))
    if new_weight.shape != state["head.weight"].shape or new_bias.shape != state["head.bias"].shape:
        raise ValueError(f"head shapes {tuple(new_weight.shape)}, {tuple(new_bias.shape)} differ")
    if not (torch.isfinite(new_weight).all() and torch.isfinite(new_bias).all()):
        raise ValueError("non-finite values in the head")
    out = {key: value.detach().clone().float() for key, value in state.items()}
    out["head.weight"] = new_weight.clone()
    out["head.bias"] = new_bias.clone()
    return out


def scale_head(state: dict[str, torch.Tensor], temperature: float) -> dict:
    if not temperature > 0:
        raise ValueError("the temperature must be positive")
    head = state["head.weight"].float().numpy() / temperature
    return splice(state, head, state["head.bias"].float().numpy() / temperature)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--head", type=Path)
    source.add_argument("--scale", type=float, help="divide the last layer by this temperature")
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if args.scale is not None:
        out = scale_head(state, args.scale)
    else:
        with np.load(args.head) as head:
            out = splice(state, head["weight"], head["bias"])
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.save)
    digest = hashlib.sha256(args.save.read_bytes()).hexdigest()
    print(f"{digest}  {args.save.name}")


if __name__ == "__main__":
    main()
