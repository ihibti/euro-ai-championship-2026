"""Write a submission state dict: the original model with its last layer replaced from a small file.

head_fit.py writes `weight` [100, 768] and `bias` [100]. This loads model.pt, replaces
`head.weight` and `head.bias`, checks shapes and finiteness, and saves float32. CPU, seconds; run
on a JURECA login node in the unlearning task folder:

    python <repo>/tasks/unlearning/splice_head.py --head <repo>/tasks/unlearning/heads/<name>.npz \
        --save output/kkrgp/<name>.pt
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    with np.load(args.head) as head:
        out = splice(state, head["weight"], head["bias"])
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.save)
    digest = hashlib.sha256(args.save.read_bytes()).hexdigest()
    print(f"{digest}  {args.save.name}")


if __name__ == "__main__":
    main()
