"""Write a submission state dict: the original model with its last layer replaced or rescaled.

head_fit.py writes `weight` [100, 768] and `bias` [100]. With --head, this loads model.pt, replaces
`head.weight` and `head.bias`, checks shapes and finiteness, and saves float32. With --scale t, it
divides the last layer by t instead: every logit is divided by t, the predicted class never
changes, so accuracy and the utility gate are those of the original. CPU, seconds; run on a JURECA
login node in the unlearning task folder:

    python <repo>/tasks/unlearning/splice_head.py --head <repo>/tasks/unlearning/heads/<name>.npz \
        --save output/kkrgp/<name>.pt
    python <repo>/tasks/unlearning/splice_head.py --scale 3 --save output/kkrgp/scale_t3.pt
    python <repo>/tasks/unlearning/splice_head.py --shift 15 --save output/kkrgp/dial15.pt

With --shift d, the bias of the nine classes that lost 130 images (0, 6, 7, 8, 29, 30, 31, 33, 38)
is lowered by d: the class dial of head_edit.py, on the last layer only. --classes names another
set, for example the four small-boat classes:

    python <repo>/tasks/unlearning/splice_head.py --shift 15 --classes 29,30,31,33 \
        --save output/kkrgp/boats15.pt
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


NINE_CLASSES = (0, 6, 7, 8, 29, 30, 31, 33, 38)


def shift_classes(state: dict[str, torch.Tensor], classes: tuple[int, ...], shift: float) -> dict:
    bias = state["head.bias"].float().numpy().copy()
    bias[list(classes)] -= shift
    return splice(state, state["head.weight"].float().numpy(), bias)


def parse_classes(text: str | None) -> tuple[int, ...]:
    if text is None:
        return NINE_CLASSES
    classes = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    if not classes or min(classes) < 0 or max(classes) > 99:
        raise ValueError(f"classes must be in 0..99: {text}")
    return classes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--head", type=Path)
    source.add_argument("--scale", type=float, help="divide the last layer by this temperature")
    source.add_argument("--shift", type=float, help="lower the nine classes' bias by this much")
    parser.add_argument("--classes", help="comma-separated classes for --shift; default the nine")
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.classes is not None and args.shift is None:
        parser.error("--classes needs --shift")
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if args.scale is not None:
        out = scale_head(state, args.scale)
    elif args.shift is not None:
        out = shift_classes(state, parse_classes(args.classes), args.shift)
    else:
        with np.load(args.head) as head:
            out = splice(state, head["weight"], head["bias"])
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.save)
    digest = hashlib.sha256(args.save.read_bytes()).hexdigest()
    print(f"{digest}  {args.save.name}")


if __name__ == "__main__":
    main()
