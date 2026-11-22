"""How a candidate at one K scores if gold sits at another K on the same dial.

Candidate and gold are both taken as the extrapolated targets (target_stats.py): the original
logits plus K times the step to the cross-fitted prediction. For every pair the server's forget
formula is applied on the exact images. It shows the shape of the score along the dial, under the
assumption that gold lies on it: how much an undershoot and an overshoot cost.

    python <repo>/tasks/unlearning/dial_matrix.py --targets output/kkrgp/xfit_extras.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lib
import numpy as np
import torch
import torch.nn.functional as F


def label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, 1).gather(1, labels.view(-1, 1)).squeeze(1)


def kl_t2(gold: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
    log_gold = F.log_softmax(gold / lib.TEMPERATURE, 1)
    log_sub = F.log_softmax(sub / lib.TEMPERATURE, 1)
    return (log_gold.exp() * (log_gold - log_sub)).sum(1).mean()


def forget_terms(
    sub: torch.Tensor, gold: torch.Tensor, orig: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    loss_sub, loss_gold, loss_orig = (label_loss(z, labels) for z in (sub, gold, orig))

    def distance(loss: torch.Tensor) -> float:
        return float((loss.mean() - loss_gold.mean()).abs() + (loss.std() - loss_gold.std()).abs())

    mean_std = 1 - min(max(distance(loss_sub) / distance(loss_orig), 0.0), 1.0)
    prediction = 1 - min(max(float(kl_t2(gold, sub) / kl_t2(gold, orig)), 0.0), 1.0)
    return mean_std, prediction


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--orig", type=Path, default=Path("output/kkrgp/orig_exact.npy"))
    parser.add_argument("--ks", default="1.0,1.2,1.4,1.6,2.0")
    parser.add_argument("--golds", default="1.1,1.2,1.4,1.6,1.8,2.0")
    args = parser.parse_args(argv)
    orig = torch.from_numpy(np.load(args.orig)).float()
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        proxy = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    delta = lib.heldout_delta(orig, proxy, columns)
    golds = [float(g) for g in args.golds.split(",")]
    print("forget score (mean_std x prediction); rows: candidate K, columns: gold K")
    print("K     " + "".join(f"{g:>8.2f}" for g in golds))
    for k in (float(k) for k in args.ks.split(",")):
        cells = []
        for g in golds:
            mean_std, prediction = forget_terms(orig + k * delta, orig + g * delta, orig, labels)
            cells.append(f"{mean_std * prediction:8.2f}")
        print(f"{k:<6.2f}" + "".join(cells))


if __name__ == "__main__":
    main()
