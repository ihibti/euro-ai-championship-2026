"""Fit the last layer, in closed form, to the extrapolated targets, and measure what leaks.

The head change (dW, db) minimises the squared distance between the edited logits and the targets
`orig + K * (xfit - orig)` on the forget features, with a ridge penalty lam * |dW|^2. A share of
every class is held out of the fit: those images are memorized training images of the same
classes, like the retained images of the server's control set, so the drift on them (KL at
temperature 2, agreement with the original) is the retain proxy. The forget statistics are read
on the fitted images and on the altered views.

    python <repo>/tasks/unlearning/head_fit.py --targets output/kkrgp/xfit_extras.npz --ks 1.6,2,3
    python <repo>/tasks/unlearning/head_fit.py ... --save-k 2 --save-lam 1 --save heads/k20.npz

With --save, the fit uses every image and writes `weight` [100, 768] and `bias` [100] in float32.
splice_head.py turns that file into a state dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TEMPERATURE = 2.0


def stratified_holdout(labels: torch.Tensor, share: float, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    held = torch.zeros(len(labels), dtype=torch.bool)
    for label in labels.unique():
        index = torch.nonzero(labels == label).squeeze(1)
        order = index[torch.randperm(len(index), generator=generator)]
        held[order[: int(round(share * len(index)))]] = True
    return held


def ridge_head_delta(
    features: torch.Tensor, wanted: torch.Tensor, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """dW, db minimising |F dW^T + db - wanted|^2 + lam |dW|^2, the bias left unpenalised."""
    mean_f = features.mean(0, keepdim=True)
    mean_w = wanted.mean(0, keepdim=True)
    centered = (features - mean_f).double()
    gram = centered.T @ centered + lam * torch.eye(features.shape[1], dtype=torch.float64)
    weight = torch.linalg.solve(gram, centered.T @ (wanted - mean_w).double()).T.float()
    bias = (mean_w - mean_f @ weight.T).squeeze(0)
    return weight, bias


def profile(logits: torch.Tensor, labels: torch.Tensor, orig: torch.Tensor) -> dict[str, float]:
    log_p = F.log_softmax(logits, 1)
    loss = -log_p.gather(1, labels.view(-1, 1)).squeeze(1)
    log_t2 = F.log_softmax(logits / TEMPERATURE, 1)
    orig_t2 = F.log_softmax(orig / TEMPERATURE, 1)
    return {
        "loss_mean": round(float(loss.mean()), 4),
        "loss_std": round(float(loss.std()), 4),
        "on_label": round(float((logits.argmax(1) == labels).float().mean()), 4),
        "label_prob_t2": round(float(log_t2.exp().gather(1, labels.view(-1, 1)).mean()), 4),
        "kl_orig_sub_t2": round(float((orig_t2.exp() * (orig_t2 - log_t2)).sum(1).mean()), 4),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("output/kkrgp/local_features.npz"))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--ks", default="1.6,2,3")
    parser.add_argument("--lams", default="0.1,1,10,100")
    parser.add_argument("--taus", default="1", help="targets divided by tau after the K step")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--save-k", type=float, default=0.0)
    parser.add_argument("--save-lam", type=float, default=1.0)
    parser.add_argument("--save-tau", type=float, default=1.0)
    args = parser.parse_args(argv)
    data = np.load(args.features)
    labels = torch.from_numpy(data["labels"]).long()
    views = [key[len("feat_") :] for key in data.files if key.startswith("feat_")]
    feats = {view: torch.from_numpy(data[f"feat_{view}"]) for view in views}
    logits = {view: torch.from_numpy(data[f"logits_{view}"]) for view in views}
    with np.load(args.targets) as proxy_file:
        proxy = torch.from_numpy(proxy_file["pred_exact"]).float()
        columns = torch.from_numpy(proxy_file["columns"])
    step = torch.zeros_like(logits["exact"])
    step[:, columns] = proxy[:, columns] - logits["exact"][:, columns]
    if args.save:
        wanted = (logits["exact"] + args.save_k * step) / args.save_tau - logits["exact"]
        weight, bias = ridge_head_delta(feats["exact"], wanted, args.save_lam)
        head_w = torch.from_numpy(data["head_weight"]) + weight
        head_b = torch.from_numpy(data["head_bias"]) + bias
        np.savez(args.save, weight=head_w.numpy(), bias=head_b.numpy())
        edited = {view: logits[view] + feats[view] @ weight.T + bias for view in views}
        report = {view: profile(edited[view], labels, logits[view]) for view in views}
        print(json.dumps({"saved": str(args.save), "all_images": report}, indent=2))
        return
    held = stratified_holdout(labels, args.holdout)
    fit = ~held
    rows = []
    grid = [
        (float(k), float(tau), float(lam))
        for k in args.ks.split(",")
        for tau in args.taus.split(",")
        for lam in args.lams.split(",")
    ]
    for k, tau, lam in grid:
        wanted = (logits["exact"] + k * step) / tau - logits["exact"]
        weight, bias = ridge_head_delta(feats["exact"][fit], wanted[fit], lam)
        row = {"K": k, "tau": tau, "lam": lam}
        for view in views:
            edited = logits[view] + feats[view] @ weight.T + bias
            row[f"fit_{view}"] = profile(edited[fit], labels[fit], logits[view][fit])
            row[f"held_{view}"] = profile(edited[held], labels[held], logits[view][held])
        rows.append(row)
        fit_exact, held_exact = row["fit_exact"], row["held_exact"]
        print(
            f"K={k:<4} tau={tau:<3} lam={lam:<6} fit: loss {fit_exact['loss_mean']:.3f}/"
            f"{fit_exact['loss_std']:.3f} on_label {fit_exact['on_label']:.3f} | held: loss "
            f"{held_exact['loss_mean']:.3f} on_label {held_exact['on_label']:.3f} "
            f"KL {held_exact['kl_orig_sub_t2']:.3f}",
            flush=True,
        )
    Path("output/kkrgp/head_fit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
