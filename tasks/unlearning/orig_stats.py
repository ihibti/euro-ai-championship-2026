"""How the original model treats the forget images and altered copies of them.

Run it in the JURECA task folder, which holds model.pt and forget_images.npz:

    python <repo>/tasks/unlearning/orig_stats.py

Labels are not provided, so the original model's prediction on the exact image stands in for the
label. The altered copies are our own; the organizers' are hidden. Writes output/orig_stats.json
and output/orig_logits.npz (logits per view, and the pseudo-labels).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))

import task_template as tt  # noqa: E402

TEMPERATURE = 2.0
BATCH = 250
OUTPUT = Path("output")


def crop_resize(images: torch.Tensor, margin: int) -> torch.Tensor:
    inner = images[:, margin : tt.IMAGE_SIZE - margin, margin : tt.IMAGE_SIZE - margin]
    resized = F.interpolate(
        inner.permute(0, 3, 1, 2).float(), size=tt.IMAGE_SIZE, mode="bilinear", antialias=True
    )
    return resized.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def make_views(images: torch.Tensor, generator: torch.Generator) -> dict[str, torch.Tensor]:
    noise = torch.randn(images.shape, generator=generator) * 8.0
    return {
        "exact": images,
        "flip": images.flip(2),
        "shift8": torch.roll(images, shifts=(8, 8), dims=(1, 2)),
        "noise8": (images.float() + noise).clamp(0, 255).to(torch.uint8),
        "crop192": crop_resize(images, 16),
    }


def summarize(logits: torch.Tensor, exact: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    loss = F.cross_entropy(logits, labels, reduction="none")
    log_p = F.log_softmax(logits / TEMPERATURE, dim=1)
    log_exact = F.log_softmax(exact / TEMPERATURE, dim=1)
    kl = (log_exact.exp() * (log_exact - log_p)).sum(1)
    return {
        "agrees_with_exact": float((logits.argmax(1) == labels).float().mean()),
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std()),
        "max_prob_t1": float(F.softmax(logits, dim=1).max(1).values.mean()),
        "max_prob_t2": float(log_p.exp().max(1).values.mean()),
        "kl_exact_to_view_t2": float(kl.mean()),
    }


def main() -> None:
    model = tt.load_model()
    images = tt.load_forget_images()
    generator = torch.Generator().manual_seed(0)
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(images), BATCH):
        for name, view in make_views(images[start : start + BATCH], generator).items():
            collected.setdefault(name, []).append(tt.predict(model, view))
    logits = {name: torch.cat(parts) for name, parts in collected.items()}
    labels = logits["exact"].argmax(1)
    counts = torch.bincount(labels, minlength=tt.NUM_CLASSES)
    stats = {
        "images": len(images),
        "classes_predicted": int((counts > 0).sum()),
        "images_per_class_min_max": [int(counts.min()), int(counts.max())],
        "views": {
            name: summarize(value, logits["exact"], labels) for name, value in logits.items()
        },
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "orig_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez(
        OUTPUT / "orig_logits.npz",
        labels=labels.numpy(),
        **{name: value.numpy() for name, value in logits.items()},
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
