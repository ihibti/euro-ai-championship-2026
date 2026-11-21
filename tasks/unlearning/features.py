"""Save what closed-form edits of the last layer need: features, logits and a thumbnail sheet.

For every forget image and every evaluation view it stores the 768 numbers that enter the last
layer, and the logits. With them, the effect of any change to the last layer on all images is a
matrix product on a laptop. It also stores the class token after a few intermediate blocks for
the exact images, to see at which depth images of one class stop looking alike.

Run in the JURECA task folder (a GPU job of about two minutes):

    python <repo>/tasks/unlearning/features.py

Writes under output/kkrgp/: logits_views.npz (every view), features.npz (four views, float16),
head.npz and peek_grid.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lib
import numpy as np
import torch
from lib import tt

DEPTHS = (3, 6, 9)
FEATURE_VIEWS = ("exact", "flip", "crop85", "jpeg50")


@torch.no_grad()
def collect(
    model: torch.nn.Module, images: torch.Tensor, batch: int = 250
) -> dict[str, np.ndarray]:
    model.eval()
    features: dict[str, list[torch.Tensor]] = {name: [] for name in lib.EVAL_VIEWS}
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in lib.EVAL_VIEWS}
    depth: dict[int, list[torch.Tensor]] = {d: [] for d in DEPTHS}
    for start in range(0, len(images), batch):
        pixels = lib.chw(images[start : start + batch])
        for name in lib.EVAL_VIEWS:
            view = lib.fixed_view(name, pixels, seed=start)
            with torch.autocast("cuda", dtype=torch.float16, enabled=lib.DEVICE == "cuda"):
                tokens = model.forward_features(lib.model_input(view))
                pooled = model.forward_head(tokens, pre_logits=True)
                out = model.head(pooled)
            features[name].append(pooled.float().cpu())
            logits[name].append(out.float().cpu())
        with torch.autocast("cuda", dtype=torch.float16, enabled=lib.DEVICE == "cuda"):
            x = model.patch_embed(lib.model_input(pixels))
            x = model.norm_pre(model.patch_drop(model._pos_embed(x)))
            for number, block in enumerate(model.blocks, start=1):
                x = block(x)
                if number in depth:
                    depth[number].append(x[:, 0].float().cpu())
    out: dict[str, np.ndarray] = {}
    for name in lib.EVAL_VIEWS:
        out[f"logit_{name}"] = torch.cat(logits[name]).numpy().astype(np.float32)
    for name in FEATURE_VIEWS:
        out[f"feat_{name}"] = torch.cat(features[name]).numpy().astype(np.float16)
    for number, parts in depth.items():
        out[f"cls_block{number}"] = torch.cat(parts).numpy().astype(np.float16)
    return out


def thumbnail_sheet(images: np.ndarray, labels: np.ndarray, per_class: int, size: int):
    from PIL import Image

    classes = np.unique(labels)
    sheet = Image.new("RGB", (per_class * size, len(classes) * size), "white")
    for row, label in enumerate(classes):
        for col, index in enumerate(np.flatnonzero(labels == label)[:per_class]):
            thumb = Image.fromarray(images[index]).resize((size, size), Image.BILINEAR)
            sheet.paste(thumb, (col * size, row * size))
    return sheet, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = tt.load_model()
    images = tt.load_forget_images()
    if args.limit:
        images = images[: args.limit]
    data = collect(model, images)
    labels = data["logit_exact"].argmax(1)
    logit_part = {name[6:]: value for name, value in data.items() if name.startswith("logit_")}
    other = {name: value for name, value in data.items() if not name.startswith("logit_")}
    np.savez_compressed(args.out / "logits_views.npz", labels=labels, **logit_part)
    np.savez_compressed(args.out / "features.npz", labels=labels, **other)
    np.savez(
        args.out / "head.npz",
        weight=model.head.weight.detach().float().cpu().numpy(),
        bias=model.head.bias.detach().float().cpu().numpy(),
    )
    sheet, classes = thumbnail_sheet(images.numpy(), labels, per_class=12, size=72)
    sheet.save(args.out / "peek_grid.png")
    print("rows of peek_grid.png are classes", classes.tolist())
    print("images per class", np.bincount(labels, minlength=tt.NUM_CLASSES).tolist())
    print(f"saved features, head and thumbnails under {args.out}")


if __name__ == "__main__":
    main()
