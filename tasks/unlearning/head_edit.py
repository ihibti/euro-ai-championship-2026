"""Training-free candidates: lower the last layer's bias of chosen classes, per class.

A bias change of -d on class c lowers that class's logit by d on every input, so its effect on
the forget images is computed exactly from the saved logits, without a GPU. Three ways to choose
the shifts, which add up:

    --full D        the classes with at least --full-count forget images (the nine at 130)
    --partial D     every other touched class c, by D * n_c / --class-size
    --shift 3,5:D   an explicit list of classes

Dry run from the logits only (numpy, laptop or cluster):

    python tasks/unlearning/head_edit.py --logits <dir>/orig_logits.npz --full 20

Write the candidate too (needs torch; a few seconds on a login node, in the task folder):

    python <repo>/tasks/unlearning/head_edit.py --logits output/orig_logits.npz --full 20 \
        --model model.pt --save output/kkrgp/bias_full20.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEMPERATURE = 2.0
CLASSES = 100


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def kl_t2(reference: np.ndarray, logits: np.ndarray) -> np.ndarray:
    ref = log_softmax(reference / TEMPERATURE)
    log_p = log_softmax(logits / TEMPERATURE)
    return (np.exp(ref) * (ref - log_p)).sum(axis=1)


def build_shifts(
    counts: np.ndarray,
    full: float,
    partial: float,
    full_count: int,
    class_size: int,
    explicit: list[str],
) -> np.ndarray:
    """Per-class amount subtracted from the bias."""
    shifts = np.zeros(CLASSES)
    is_full = counts >= full_count
    shifts[is_full] += full
    touched = (counts > 0) & ~is_full
    shifts[touched] += partial * counts[touched] / class_size
    for item in explicit:
        classes, amount = item.split(":")
        shifts[[int(c) for c in classes.split(",")]] += float(amount)
    return shifts


def profile(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {}
    log_probs = log_softmax(logits[mask])
    loss = -log_probs[np.arange(mask.sum()), labels[mask]]
    return {
        "images": int(mask.sum()),
        "loss_mean": round(float(loss.mean()), 4),
        "loss_std": round(float(loss.std()), 4),
        "unchanged": round(float((logits[mask].argmax(1) == labels[mask]).mean()), 4),
    }


def report(data: dict[str, np.ndarray], shifts: np.ndarray, full_count: int) -> dict:
    labels = data["labels"]
    counts = np.bincount(labels, minlength=CLASSES)
    full_classes = np.flatnonzero(counts >= full_count)
    in_full = np.isin(labels, full_classes)
    out: dict = {
        "full_classes": full_classes.tolist(),
        "shifted_classes": {int(c): round(float(shifts[c]), 3) for c in np.flatnonzero(shifts)},
        "views": {},
    }
    for view in (name for name in data if name != "labels"):
        orig = data[view].astype(np.float64)
        edited = orig - shifts[None, :]
        moved_to = edited.argmax(1)
        out["views"][view] = {
            "all": profile(edited, labels, np.ones(len(labels), bool)),
            "full": profile(edited, labels, in_full),
            "partial": profile(edited, labels, ~in_full),
            "kl_orig_edited_t2": round(float(kl_t2(orig, edited).mean()), 4),
            "full_images_landing_in_full_classes": round(
                float(np.isin(moved_to[in_full], full_classes).mean()), 4
            ),
        }
    return out


def save_candidate(model_path: Path, shifts: np.ndarray, out_path: Path) -> None:
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    bias = state["head.bias"]
    if bias.shape != (CLASSES,):
        raise ValueError(f"unexpected head.bias shape {tuple(bias.shape)}")
    state["head.bias"] = (bias.double() - torch.from_numpy(shifts)).float()
    state = {
        key: value.float() if value.is_floating_point() else value for key, value in state.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--full", type=float, default=0.0)
    parser.add_argument("--partial", type=float, default=0.0)
    parser.add_argument("--full-count", type=int, default=129)
    parser.add_argument("--class-size", type=int, default=130)
    parser.add_argument("--shift", action="append", default=[])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args(argv)
    with np.load(args.logits) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    counts = np.bincount(data["labels"], minlength=CLASSES)
    shifts = build_shifts(
        counts, args.full, args.partial, args.full_count, args.class_size, args.shift
    )
    result = report(data, shifts, args.full_count)
    print(json.dumps(result, indent=2))
    if args.save:
        if not args.model:
            raise SystemExit("--save needs --model")
        save_candidate(args.model, shifts, args.save)
        args.save.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
