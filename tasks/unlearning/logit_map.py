"""Training-free candidates as an affine map on the logits: new = M @ old + c.

Any such map is a change of the last layer alone (weight' = M @ weight, bias' = M @ bias + c), so
its effect on every saved view of the forget images is exact and needs no GPU. Three dials:

    --full D     lower the whole-forgotten classes (130 forget images) by D on every input.
    --shrink S   scale the logit of each thinned class c by 1 - S * n_c / 130. On the class's own
                 images, where that logit is about 23, this moves probability to the runners-up
                 at temperature 2 without changing the prediction; elsewhere it does little.
    --tilt K     add K * share[k -> j] * logit_k to class j, for every whole-forgotten class k,
                 where share is the average destination of class k's images once the forgotten
                 classes are out. An image that looks like class k then counts as evidence for
                 the classes its kind is sent to, which sharpens "which wrong class".
    --even S     scale the logit of every touched class by 1 - S, the nine included. This is the
                 hedge for the other reading of the data, in which no class is removed and the
                 forget images are extra images of classes gold still knows.

Dry run (numpy only):

    python tasks/unlearning/logit_map.py --logits <dir>/logits_views.npz --full 30 --shrink 0.5

Write the candidate (needs torch; seconds on a login node, in the task folder):

    python <repo>/tasks/unlearning/logit_map.py --logits output/kkrgp/logits_views.npz \
        --full 30 --shrink 0.5 --model model.pt --save output/kkrgp/map_f30_s50.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEMPERATURE = 2.0
CLASSES = 100
CLASS_SIZE = 130
FULL_COUNT = 129


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    return probs / probs.sum(axis=1, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(np.clip(p, 1e-30, None)) - np.log(np.clip(q, 1e-30, None)))).sum(axis=1)


def groups(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=CLASSES)
    full = np.flatnonzero(counts >= FULL_COUNT)
    thinned = np.flatnonzero((counts > 0) & (counts < FULL_COUNT))
    return counts, full, thinned


def destination_shares(logits: np.ndarray, labels: np.ndarray, full: np.ndarray) -> np.ndarray:
    """share[k, j]: where the images of whole-forgotten class k go once those classes are out."""
    shares = np.zeros((CLASSES, CLASSES))
    for k in full:
        rest = logits[labels == k].copy()
        rest[:, full] = -np.inf
        shares[k] = softmax(rest).mean(axis=0)
    return shares


def build_map(
    logits: np.ndarray,
    labels: np.ndarray,
    full_shift: float,
    shrink: float,
    tilt: float,
    even: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    counts, full, thinned = groups(labels)
    matrix = np.eye(CLASSES)
    offset = np.zeros(CLASSES)
    offset[full] -= full_shift
    matrix[thinned, thinned] -= shrink * counts[thinned] / CLASS_SIZE
    touched = np.flatnonzero(counts > 0)
    matrix[touched, touched] -= even
    if tilt:
        shares = destination_shares(logits, labels, full)
        matrix[:, full] += tilt * shares[full].T
    return matrix, offset


def describe(old: np.ndarray, new: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    index = np.flatnonzero(mask)
    own = labels[index]
    rows = np.arange(len(index))
    loss = -np.log(np.clip(softmax(new[index])[rows, own], 1e-300, None))
    soft_new = softmax(new[index], TEMPERATURE)
    entropy = -(soft_new * np.log(np.clip(soft_new, 1e-30, None))).sum(axis=1)
    return {
        "images": len(index),
        "loss_mean": round(float(loss.mean()), 4),
        "loss_std": round(float(loss.std()), 4),
        "unchanged": round(float((new[index].argmax(1) == own).mean()), 4),
        "label_prob_t2": round(float(soft_new[rows, own].mean()), 4),
        "top_prob_t2": round(float(soft_new.max(axis=1).mean()), 4),
        "entropy_t2": round(float(entropy.mean()), 4),
        "kl_old_new_t2": round(float(kl(softmax(old[index], TEMPERATURE), soft_new).mean()), 4),
    }


def report(data: dict[str, np.ndarray], matrix: np.ndarray, offset: np.ndarray) -> dict:
    labels = data["labels"]
    _, full, _ = groups(labels)
    in_full = np.isin(labels, full)
    out: dict = {}
    for view in (name for name in data if name != "labels"):
        old = data[view].astype(np.float64)
        new = old @ matrix.T + offset
        out[view] = {
            "all": describe(old, new, labels, np.ones(len(labels), bool)),
            "full": describe(old, new, labels, in_full),
            "thinned": describe(old, new, labels, ~in_full),
        }
    return out


def save_candidate(model_path: Path, matrix: np.ndarray, offset: np.ndarray, out: Path) -> None:
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    weight = state["head.weight"].double().numpy()
    bias = state["head.bias"].double().numpy()
    state["head.weight"] = torch.from_numpy(matrix @ weight).float()
    state["head.bias"] = torch.from_numpy(matrix @ bias + offset).float()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--full", type=float, default=0.0)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument("--tilt", type=float, default=0.0)
    parser.add_argument("--even", type=float, default=0.0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--views", default="exact")
    args = parser.parse_args(argv)
    with np.load(args.logits) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    matrix, offset = build_map(
        data["exact"].astype(np.float64),
        data["labels"],
        args.full,
        args.shrink,
        args.tilt,
        args.even,
    )
    result = report(data, matrix, offset)
    shown = result if args.views == "all" else {v: result[v] for v in args.views.split(",")}
    print(json.dumps(shown, indent=1))
    if args.save:
        if not args.model:
            raise SystemExit("--save needs --model")
        save_candidate(args.model, matrix, offset, args.save)
        settings = {
            "full": args.full,
            "shrink": args.shrink,
            "tilt": args.tilt,
            "even": args.even,
        }
        record = {"settings": settings, "views": result}
        args.save.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
