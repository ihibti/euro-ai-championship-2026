"""What the extrapolated targets ask for, before any training: loss level, spread and destinations.

For every K, the target of an image is the original logits plus K times the step to its held-out
prediction (csd.py --extrapolate). This prints, on the exact images, the statistics that enter the
server's forget score: the loss mean and spread at temperature 1, the share still on the original
label, the label's probability at temperature 2, and the KL at temperature 2 from the original.

Run where model.pt and forget_images.npz are, with the xfit npz written by xfit.py:

    python <repo>/tasks/unlearning/target_stats.py --targets output/kkrgp/xfit_extras.npz

The original logits are cached in output/kkrgp/orig_exact.npy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lib
import numpy as np
import torch
import torch.nn.functional as F
from lib import tt


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def original_logits(cache: Path, batch: int = 64) -> torch.Tensor:
    if cache.is_file():
        return torch.from_numpy(np.load(cache))
    where = device()
    model = tt.load_model().to(where).eval()
    images = tt.load_forget_images()
    out = []
    for start in range(0, len(images), batch):
        pixels = images[start : start + batch].to(where).permute(0, 3, 1, 2).float()
        out.append(model(lib.model_input(pixels)).float().cpu())
    logits = torch.cat(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, logits.numpy())
    return logits


def stats(logits: torch.Tensor, labels: torch.Tensor, orig: torch.Tensor) -> dict[str, float]:
    log_p = F.log_softmax(logits, dim=1)
    loss = -log_p.gather(1, labels.view(-1, 1)).squeeze(1)
    log_t2 = F.log_softmax(logits / lib.TEMPERATURE, dim=1)
    orig_t2 = F.log_softmax(orig / lib.TEMPERATURE, dim=1)
    return {
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std()),
        "on_label": float((logits.argmax(1) == labels).float().mean()),
        "label_prob_t2": float(log_t2.exp().gather(1, labels.view(-1, 1)).mean()),
        "kl_t2_from_orig": float((log_t2.exp() * (log_t2 - orig_t2)).sum(1).mean()),
        "loss_q90": float(loss.quantile(0.9)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("output/kkrgp/orig_exact.npy"))
    parser.add_argument("--ks", default="0,1,1.5,2,2.5,3,4.5")
    args = parser.parse_args(argv)
    orig = original_logits(args.cache)
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        proxy = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    delta = lib.heldout_delta(orig, proxy, columns)
    rows = {k: stats(orig + float(k) * delta, labels, orig) for k in args.ks.split(",")}
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
