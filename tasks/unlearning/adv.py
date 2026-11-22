"""How fragile each forget image of a thinned class is: the smallest attack that flips it.

A model that never saw an image is most likely wrong on the images that sit close to a decision
boundary. For every forget image of a thinned class this runs an L2 attack of growing size on the
class-dial model (the whole-forgotten classes lowered by --full, so they cannot be the landing
class) and records the first size at which the prediction flips, the class it lands in, the
logits there, and the label's margin after each size.

Run in the JURECA task folder (a GPU job of about ten minutes):

    python <repo>/tasks/unlearning/adv.py --full 25

Writes output/kkrgp/adv.npz: labels, attacked (bool), eps (the sizes, pixel scale 0..1),
flip_eps (inf if never), adv_label, adv_logits, margins [N, sizes].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lib
import numpy as np
import torch
import torch.nn.functional as F
from lib import tt

SIZES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def attack(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    shift: torch.Tensor,
    size: float,
    steps: int,
) -> torch.Tensor:
    """Edited-model logits at the end of an L2 attack of the given size (pixels in 0..255)."""
    base = pixels / 255.0
    delta = torch.zeros_like(base)
    step_size = 2.5 * size / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=lib.DEVICE == "cuda"):
            logits = model(lib.model_input((base + delta) * 255.0)).float() - shift
        loss = F.cross_entropy(logits, labels, reduction="sum")
        (grad,) = torch.autograd.grad(loss, delta)
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        delta = delta.detach() + step_size * grad / norm
        length = delta.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(size / length, max=1.0)
        delta = (base + delta).clamp(0, 1) - base
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=lib.DEVICE == "cuda"):
        return model(lib.model_input((base + delta) * 255.0)).float() - shift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = tt.load_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    images = tt.load_forget_images()
    if args.limit:
        images = images[: args.limit]
    labels = lib.logits_of(model, images).argmax(1)
    shift, _ = lib.class_plan(labels, args.full, 0.0)
    attacked = shift[labels] == 0
    count = len(images)
    flip_eps = np.full(count, np.inf, dtype=np.float32)
    adv_label = labels.numpy().copy()
    adv_logits = np.zeros((count, tt.NUM_CLASSES), dtype=np.float32)
    margins = np.zeros((count, len(SIZES)), dtype=np.float32)
    shift_device = shift.to(lib.DEVICE)
    targets = torch.nonzero(attacked).squeeze(1)
    for start in range(0, len(targets), args.batch):
        index = targets[start : start + args.batch]
        pixels = lib.chw(images[index])
        batch_labels = labels[index].to(lib.DEVICE)
        for level, size in enumerate(SIZES):
            logits = attack(model, pixels, batch_labels, shift_device, size, args.steps).cpu()
            own = logits.gather(1, labels[index].view(-1, 1)).squeeze(1)
            others = logits.scatter(1, labels[index].view(-1, 1), float("-inf")).max(1).values
            margins[index.numpy(), level] = (own - others).numpy()
            flipped = (logits.argmax(1) != labels[index]).numpy()
            first = flipped & np.isinf(flip_eps[index.numpy()])
            where = index.numpy()[first]
            flip_eps[where] = size
            adv_label[where] = logits.argmax(1).numpy()[first]
            adv_logits[where] = logits.numpy()[first]
        print(f"{start + len(index)}/{len(targets)} attacked", flush=True)
    np.savez_compressed(
        args.out / "adv.npz",
        labels=labels.numpy(),
        attacked=attacked.numpy(),
        eps=np.array(SIZES, dtype=np.float32),
        flip_eps=flip_eps,
        adv_label=adv_label,
        adv_logits=adv_logits,
        margins=margins,
    )
    done = flip_eps[attacked.numpy()]
    for size in SIZES:
        print(f"flipped at eps <= {size}: {(done <= size).mean():.3f}")


if __name__ == "__main__":
    main()
