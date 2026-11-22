"""Penultimate features and logits of the original model on the forget images, on a laptop.

The features are what the last layer sees: the final norm's output for the class token, before the
head. With them, any change to the head can be evaluated exactly without the network. Views follow
lib.fixed_view, the same seeded alterations as the GPU reports.

Run where model.pt and forget_images.npz are:

    python <repo>/tasks/unlearning/local_features.py --views exact,flip,shift10,jitter

Writes output/kkrgp/local_features.npz: labels, head_weight, head_bias, and per view
feat_<view> [N, 768] and logits_<view> [N, 100], in float32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lib
import numpy as np
import torch
from lib import tt


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def features_and_logits(
    model: torch.nn.Module, images: torch.Tensor, view: str, where: str, batch: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    feats, logits = [], []
    for start in range(0, len(images), batch):
        pixels = images[start : start + batch].permute(0, 3, 1, 2).float()
        if view != "exact":
            pixels = lib.fixed_view(view, pixels, seed=start)
        tokens = model.forward_features(lib.model_input(pixels.to(where)))
        pooled = model.forward_head(tokens, pre_logits=True)
        feats.append(pooled.float().cpu())
        logits.append(model.head(pooled).float().cpu())
    return torch.cat(feats), torch.cat(logits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--views", default="exact,flip,shift10,jitter")
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp/local_features.npz"))
    args = parser.parse_args(argv)
    where = device()
    model = tt.load_model().to(where).eval()
    images = tt.load_forget_images()
    arrays: dict[str, np.ndarray] = {
        "head_weight": model.head.weight.detach().cpu().numpy(),
        "head_bias": model.head.bias.detach().cpu().numpy(),
    }
    for view in args.views.split(","):
        feats, logits = features_and_logits(model, images, view, where)
        arrays[f"feat_{view}"] = feats.numpy()
        arrays[f"logits_{view}"] = logits.numpy()
        if view == "exact":
            arrays["labels"] = logits.argmax(1).numpy()
        print(view, tuple(feats.shape), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
