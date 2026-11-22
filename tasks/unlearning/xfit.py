"""Cross-fitted retraining of the last blocks: what the model would say of an image it never saw.

The original model memorizes its training images in its last blocks: a nearest-class-mean probe
on the class token is right on half of the forget images after block 9 and on all of them after
block 12. So the first --cut blocks are kept frozen, and everything after them (blocks, final
norm, head) is thrown away and trained again from a fresh start, on forget images only, to
reproduce the original model's logits on those images. With K folds, every forget image is
predicted by a copy whose last blocks never trained on it. Those held-out predictions are our
estimate of gold, image by image: whether it stays right, and which class takes over when not.

Two readings of the forget set (analysis.md, section 1b):

    --reading extras    every class is known to gold: fold over all forget images.
    --reading removal   the whole-forgotten classes (130 forget images) are never trained on and
                        their logits are left out of the fit; their images are predicted by
                        every fold's copy and the logits are averaged. The thinned classes are
                        folded as above.

Run in the JURECA task folder (one GPU, about 15 minutes for 5 folds of 30 epochs):

    python <repo>/tasks/unlearning/xfit.py --name extras --reading extras

Writes output/kkrgp/xfit_<name>.npz (labels, fold, columns, pred_<view> [N, 100]), a JSON summary
and xfit_<name>_exact.npy (the exact-view predictions in float16, small enough to bring back).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import lib
import numpy as np
import torch
import torch.nn.functional as F
from lib import tt

VIEWS = ("exact", "flip", "crop85", "shift10", "jitter", "blur1", "jpeg50", "noise8")


@torch.no_grad()
def cached_tokens(model: torch.nn.Module, images: torch.Tensor, cut: int) -> torch.Tensor:
    """Tokens after the first `cut` blocks, [views, images, tokens, width], on the device."""
    model.eval()
    stacked = None
    for start in range(0, len(images), 250):
        pixels = lib.chw(images[start : start + 250])
        for number, name in enumerate(VIEWS):
            view = lib.fixed_view(name, pixels, seed=start)
            with torch.autocast("cuda", dtype=torch.float16, enabled=lib.DEVICE == "cuda"):
                x = model.patch_embed(lib.model_input(view))
                x = model.norm_pre(model.patch_drop(model._pos_embed(x)))
                for block in model.blocks[:cut]:
                    x = block(x)
            if stacked is None:
                shape = (len(VIEWS), len(images), x.shape[1], x.shape[2])
                stacked = torch.empty(shape, dtype=torch.float16, device=x.device)
            stacked[number, start : start + len(pixels)] = x.half()
    return stacked


@torch.no_grad()
def late_logits(
    model: torch.nn.Module, tokens: torch.Tensor, index: torch.Tensor, cut: int
) -> torch.Tensor:
    """Logits [views, len(index), classes] of the part of the model after block `cut`."""
    model.eval()
    rows = index.to(tokens.device)
    out = torch.empty(tokens.shape[0], len(rows), tt.NUM_CLASSES, device=tokens.device)
    for view in range(tokens.shape[0]):
        for start in range(0, len(rows), 500):
            chunk = tokens[view, rows[start : start + 500]].float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=lib.DEVICE == "cuda"):
                out[view, start : start + 500] = late_forward(model, chunk, cut).float()
    return out


def late_forward(model: torch.nn.Module, tokens: torch.Tensor, cut: int) -> torch.Tensor:
    x = tokens
    for block in model.blocks[cut:]:
        x = block(x)
    return model.forward_head(model.norm(x))


def fresh_late(orig: torch.nn.Module, cut: int, keep_scale: bool) -> torch.nn.Module:
    """A new model whose blocks after `cut`, norm and head start from a fresh initialisation."""
    fresh = tt.build_model().to(lib.DEVICE)
    if keep_scale:
        with torch.no_grad():
            for new, old in zip(fresh.blocks[cut:], orig.blocks[cut:], strict=True):
                new.ls1.gamma.copy_(old.ls1.gamma)
                new.ls2.gamma.copy_(old.ls2.gamma)
    for p in fresh.parameters():
        p.requires_grad_(False)
    return fresh


def late_parameters(model: torch.nn.Module, cut: int) -> list[torch.nn.Parameter]:
    modules = [*model.blocks[cut:], model.norm, model.head]
    params = [p for module in modules for p in module.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


def stratified_folds(labels: torch.Tensor, eligible: torch.Tensor, folds: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    fold = torch.full((len(labels),), -1, dtype=torch.long)
    for label in labels[eligible].unique():
        members = torch.nonzero(eligible & (labels == label)).squeeze(1)
        members = members[torch.randperm(len(members), generator=generator)]
        fold[members] = torch.arange(len(members)) % folds
    return fold


def fit(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    train_index: torch.Tensor,
    columns: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> float:
    params = late_parameters(model, args.cut)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    pairs = torch.cartesian_prod(train_index, torch.arange(len(VIEWS)))
    steps = math.ceil(len(pairs) / args.batch) * args.epochs
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    device_labels = labels.to(lib.DEVICE)
    last = 0.0
    for _ in range(args.epochs):
        model.train()
        order = pairs[torch.randperm(len(pairs), generator=generator)]
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = order[start : start + args.batch]
            rows = batch[:, 0].to(lib.DEVICE)
            views = batch[:, 1].to(lib.DEVICE)
            wanted = targets[views, rows]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=lib.DEVICE == "cuda"):
                logits = late_forward(model, tokens[views, rows].float(), args.cut).float()
            kept = logits[:, columns]
            position = (columns.view(1, -1) == device_labels[rows].view(-1, 1)).float().argmax(1)
            fit_loss = F.smooth_l1_loss(kept, wanted[:, columns])
            loss = fit_loss + args.ce_weight * F.cross_entropy(kept, position)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss.detach()) * len(batch)
        last = total / len(order)
    model.eval()
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--reading", choices=("extras", "removal"), required=True)
    parser.add_argument("--cut", type=int, default=9)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--fresh-scale", action="store_true", help="also reset LayerScale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(args.seed)
    orig = tt.load_model().eval()
    for p in orig.parameters():
        p.requires_grad_(False)
    images = tt.load_forget_images()
    if args.limit:
        images = images[: args.limit]
    labels = lib.logits_of(orig, images).argmax(1)
    tokens = cached_tokens(orig, images, args.cut)
    targets = late_logits(orig, tokens, torch.arange(len(images)), args.cut)
    shift, _ = lib.class_plan(labels, 1.0, 0.0)
    whole = shift > 0 if args.reading == "removal" else torch.zeros_like(shift, dtype=torch.bool)
    columns = torch.nonzero(~whole).squeeze(1).to(lib.DEVICE)
    in_whole = whole[labels]
    fold = stratified_folds(labels, ~in_whole, args.folds, args.seed)
    count = len(images)
    pred = torch.zeros(len(VIEWS), count, tt.NUM_CLASSES)
    summary: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "folds": []}
    whole_index = torch.nonzero(in_whole).squeeze(1)
    for k in range(args.folds):
        train_index = torch.nonzero((fold >= 0) & (fold != k)).squeeze(1)
        held_index = torch.nonzero(fold == k).squeeze(1)
        model = fresh_late(orig, args.cut, keep_scale=not args.fresh_scale)
        model.load_state_dict(
            {key: value for key, value in orig.state_dict().items() if early(key, args.cut)},
            strict=False,
        )
        last = fit(model, tokens, targets, labels, train_index, columns, args, args.seed + k)
        held = late_logits(model, tokens, held_index, args.cut).cpu()
        seen = late_logits(model, tokens, train_index[:500], args.cut).cpu()
        pred[:, held_index] = held
        if len(whole_index):
            pred[:, whole_index] += (
                late_logits(model, tokens, whole_index, args.cut).cpu() / args.folds
            )
        cols = columns.cpu()
        held_top = cols[held[0][:, cols].argmax(1)]
        seen_top = cols[seen[0][:, cols].argmax(1)]
        record = {
            "fold": k,
            "train_images": len(train_index),
            "last_train_loss": last,
            "seen_agreement": float((seen_top == labels[train_index[:500]]).float().mean()),
            "held_agreement": float((held_top == labels[held_index]).float().mean()),
            "seconds": time.time() - started,
        }
        summary["folds"].append(record)
        print(json.dumps(record), flush=True)
    cols = columns.cpu()
    eligible = fold >= 0
    exact = pred[0][:, cols]
    top = cols[exact.argmax(1)]
    position = (cols.view(1, -1) == labels.view(-1, 1)).float().argmax(1)
    loss = F.cross_entropy(exact[eligible], position[eligible], reduction="none")
    summary["held_out"] = {
        "agreement_with_label": float((top[eligible] == labels[eligible]).float().mean()),
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std()),
        "label_prob_t2": float(lib.label_prob_t2(exact[eligible], position[eligible]).mean()),
        "agreement_by_view": {
            name: float(
                (cols[pred[number][:, cols].argmax(1)][eligible] == labels[eligible]).float().mean()
            )
            for number, name in enumerate(VIEWS)
        },
    }
    if len(whole_index):
        soft = F.softmax(exact[in_whole] / lib.TEMPERATURE, dim=1)
        summary["whole_classes"] = {
            "images": len(whole_index),
            "top_prob_t2": float(soft.max(1).values.mean()),
            "entropy_t2": float(-(soft * soft.clamp_min(1e-30).log()).sum(1).mean()),
            "top_destinations": {
                int(c): {
                    int(d): round(float(n) / int((labels == c).sum()), 3)
                    for d, n in zip(
                        *torch.unique(top[labels == c], return_counts=True), strict=True
                    )
                    if n >= 0.1 * int((labels == c).sum())
                }
                for c in torch.nonzero(whole).squeeze(1)
            },
        }
    summary["seconds"] = time.time() - started
    np.savez_compressed(
        args.out / f"xfit_{args.name}.npz",
        labels=labels.numpy(),
        fold=fold.numpy(),
        columns=cols.numpy(),
        **{f"pred_{name}": pred[number].numpy() for number, name in enumerate(VIEWS)},
    )
    np.save(args.out / f"xfit_{args.name}_exact.npy", pred[0].numpy().astype(np.float16))
    (args.out / f"xfit_{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2), flush=True)


def early(key: str, cut: int) -> bool:
    """True for parameters of the frozen part: everything before block `cut`."""
    if key.startswith("blocks."):
        return int(key.split(".")[1]) < cut
    return not (key.startswith("norm.") or key.startswith("head."))


if __name__ == "__main__":
    main()
