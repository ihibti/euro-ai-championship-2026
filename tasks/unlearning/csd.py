"""Soft-target fine-tune on top of the class dial: can a change stay on the images it targets?

The student starts as the original model with the whole-forgotten classes' bias lowered by
--full (the class dial of head_edit.py). It is then fine-tuned on forget images and random slight
alterations of them, towards targets built from the original model's own logits:

- every image: the whole-forgotten classes lowered by --full (what the student already does);
- an image of a thinned class c: its own label lowered further by --hedge * n_c / 130.

With --holdout, a share of every class's forget images is never trained on. Those images are
memorized training images of the same classes, like the retained images the server's control set
may hold. Two numbers come out of them:

- transfer: how much of the hedge taught on the trained images shows up on the held-out ones.
  Near 0, the change is specific to the images; near 1, it acts on the whole class and a
  closed-form edit of the last layer does the same job without training.
- the retain proxy: accuracy and KL at temperature 2 against the class-dial model on them.

With --spare-classes N, N thinned classes get no hedge and none of their images is trained on.
They stand in for the 55 classes we hold no image of: the drift on them is what the fine-tune
does to classes it was never shown.

Run in the JURECA task folder:

    python <repo>/tasks/unlearning/csd.py --name csd_f25_h8 --full 25 --hedge 8 --holdout 0.2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lib
import torch
import torch.nn.functional as F
from lib import tt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp"))
    parser.add_argument("--full", type=float, default=25.0)
    parser.add_argument("--hedge", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--anchor", type=float, default=0.0, help="decoupled pull to the start")
    parser.add_argument("--blocks", type=int, default=-1, help="train the last N blocks; -1 = all")
    parser.add_argument("--early", type=int, default=0, help="train only the first N blocks")
    parser.add_argument("--freeze-head", action="store_true")
    parser.add_argument("--exact-share", type=float, default=0.25)
    parser.add_argument("--aug-strength", type=float, default=1.0)
    parser.add_argument("--t2-weight", type=float, default=1.0)
    parser.add_argument("--holdout", type=float, default=0.0)
    parser.add_argument("--spare-classes", type=int, default=0, help="thinned classes left alone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N images only, for smoke runs")
    parser.add_argument("--save", action="store_true", help="save the state dict even with holdout")
    return parser.parse_args(argv)


def trainable(model: torch.nn.Module, args: argparse.Namespace) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    extra = []
    if args.early > 0:
        modules = [model.patch_embed, *model.blocks[: args.early]]
        extra = [model.cls_token, model.pos_embed]
    elif args.blocks < 0:
        modules = [module for name, module in model.named_children() if name != "head"]
        extra = [model.cls_token, model.pos_embed]
    else:
        modules = [model.norm, *model.blocks[len(model.blocks) - args.blocks :]]
    if not args.freeze_head and args.early == 0:
        modules.append(model.head)
    params = [p for module in modules for p in module.parameters()] + extra
    for p in params:
        p.requires_grad_(True)
    return params


def stratified_split(
    labels: torch.Tensor, holdout: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train, held = [], []
    for label in labels.unique():
        members = torch.nonzero(labels == label).squeeze(1)
        members = members[torch.randperm(len(members), generator=generator)]
        cut = int(round(len(members) * holdout))
        held.append(members[:cut])
        train.append(members[cut:])
    return torch.cat(train).sort().values, torch.cat(held).sort().values


def train(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    shift: torch.Tensor,
    beta: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    params = trainable(student, args)
    starts = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (len(images) + args.batch - 1) // args.batch
    total = max(1, steps_per_epoch * args.epochs)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, total // 10))
    )
    generator = torch.Generator(device=lib.DEVICE).manual_seed(args.seed)
    order_generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(images), generator=order_generator)
        sums = {"loss": 0.0, "kl_t1": 0.0, "kl_t2": 0.0}
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            pixels = lib.chw(images[index])
            altered = lib.random_views(pixels, generator, args.aug_strength)
            keep = torch.rand(len(index), generator=generator, device=lib.DEVICE)
            pixels = torch.where((keep < args.exact_share).view(-1, 1, 1, 1), pixels, altered)
            batch_labels = labels[index].to(lib.DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                teacher_logits = teacher(lib.model_input(pixels)).float()
            target = lib.soft_targets(teacher_logits, batch_labels, shift, beta, args.tau)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = student(lib.model_input(pixels)).float()
            log_p = F.log_softmax(logits, dim=1)
            kl_t1 = (target.exp() * (target - log_p)).sum(1).mean()
            kl_t2 = lib.kl_t2(target, logits).mean()
            loss = kl_t1 + args.t2_weight * kl_t2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            if args.anchor > 0:
                lr = schedule.get_last_lr()[0]
                with torch.no_grad():
                    for p, p0 in zip(params, starts, strict=True):
                        p.add_(p - p0, alpha=-lr * args.anchor)
            sums["loss"] += float(loss) * len(index)
            sums["kl_t1"] += float(kl_t1) * len(index)
            sums["kl_t2"] += float(kl_t2) * len(index)
        record = {key: value / len(images) for key, value in sums.items()}
        record["epoch"] = epoch + 1
        history.append(record)
        print(json.dumps(record), flush=True)
    student.eval()
    return history


def group_report(
    orig: torch.Tensor,
    sub: torch.Tensor,
    labels: torch.Tensor,
    shift: torch.Tensor,
    beta: torch.Tensor,
    tau: float,
) -> dict[str, float]:
    """One group of images on one view: the dial-only reference, the hedged target, the student."""
    none = torch.zeros_like(beta)
    reference = lib.soft_targets(orig, labels, shift, none)
    target = lib.soft_targets(orig, labels, shift, beta, tau)
    p_reference = float(lib.label_prob_t2(reference, labels).mean())
    p_target = float(lib.label_prob_t2(target, labels).mean())
    p_sub = float(lib.label_prob_t2(sub, labels).mean())
    wanted = p_reference - p_target
    stats = lib.profile(sub, labels)
    return {
        "images": len(labels),
        "label_prob_t2_reference": p_reference,
        "label_prob_t2_target": p_target,
        "label_prob_t2_sub": p_sub,
        "moved_share": (p_reference - p_sub) / wanted if abs(wanted) > 1e-6 else None,
        "kl_reference_sub_t2": float(lib.kl_t2(reference, sub).mean()),
        "kl_target_sub_t2": float(lib.kl_t2(target, sub).mean()),
        "kl_orig_sub_t2": float(lib.kl_t2(F.log_softmax(orig, dim=1), sub).mean()),
        "acc_vs_reference": float((sub.argmax(1) == reference.argmax(1)).float().mean()),
        "loss_mean": stats["loss_mean"],
        "loss_std": stats["loss_std"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(args.seed)
    teacher = tt.load_model().eval()
    student = tt.load_model()
    images = tt.load_forget_images()
    if args.limit:
        images = images[: args.limit]
    labels = lib.logits_of(teacher, images).argmax(1)
    shift, beta = lib.class_plan(labels, args.full, args.hedge)
    hedged = torch.nonzero(beta > 0).squeeze(1)
    order = torch.randperm(len(hedged), generator=torch.Generator().manual_seed(args.seed))
    spare = hedged[order[: args.spare_classes]]
    beta[spare] = 0
    with torch.no_grad():
        student.head.bias -= shift.to(student.head.bias)
    is_full = shift[labels] > 0
    is_spare = torch.isin(labels, spare)
    train_index, held_index = stratified_split(labels, args.holdout, args.seed)
    train_index = train_index[~is_spare[train_index]]
    held_index = held_index[~is_spare[held_index]]
    spare_index = torch.nonzero(is_spare).squeeze(1)
    print(
        f"full classes={int((shift > 0).sum())} hedged classes={int((beta > 0).sum())} "
        f"spare classes={spare.tolist()} train={len(train_index)} held={len(held_index)} "
        f"spare={len(spare_index)}",
        flush=True,
    )
    history = []
    if args.epochs > 0:
        history = train(
            student, teacher, images[train_index], labels[train_index], shift, beta, args
        )
    report: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "history": history}
    for part, index in (("trained", train_index), ("held", held_index), ("spare", spare_index)):
        if len(index) == 0:
            continue
        orig_views = lib.view_logits(teacher, images[index], lib.EVAL_VIEWS)
        sub_views = lib.view_logits(student, images[index], lib.EVAL_VIEWS)
        part_labels = labels[index]
        report[part] = {}
        for group, mask in (("full", is_full[index]), ("thinned", ~is_full[index])):
            if not mask.any():
                continue
            report[part][group] = {
                view: group_report(
                    orig_views[view][mask],
                    sub_views[view][mask],
                    part_labels[mask],
                    shift,
                    beta,
                    args.tau,
                )
                for view in lib.EVAL_VIEWS
            }
    report["seconds"] = time.time() - started
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {"name": args.name}
    for part in ("trained", "held", "spare"):
        for group in ("full", "thinned"):
            exact = report.get(part, {}).get(group, {}).get("exact")
            if exact:
                keys = ("moved_share", "kl_reference_sub_t2", "acc_vs_reference", "loss_mean")
                summary[f"{part}_{group}"] = {
                    key: None if exact[key] is None else round(exact[key], 4) for key in keys
                }
    print(json.dumps(summary, indent=2), flush=True)
    if args.holdout == 0 or args.save:
        torch.save(lib.state_dict_float32(student), args.out / f"{args.name}.pt")
        print(f"saved {args.out / f'{args.name}.pt'}", flush=True)


if __name__ == "__main__":
    main()
