"""Probe how the dataset was built and how much pixel-level evidence the known message gives.

For every train image of a method:
- read the message from the bounding box of the true mask (is the watermark embedded on that crop?);
- embed the known message again, take the change as the expected watermark pattern, and regress
  the image on it inside small windows. The slope is near 1 where the pattern is present and near
  0 where it is not. A random message gives the null.

Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/probe_process.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_features import auc  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
WINDOWS = (4, 8, 16, 32, 64)
CELL = 4


def pattern(wm, image: np.ndarray, bits: np.ndarray) -> np.ndarray:
    if wm.name == "pixelseal":
        x = torch.from_numpy(image).permute(2, 0, 1)[None].float().div(255).to(wm.device)
        msgs = torch.tensor(bits[None], dtype=torch.float32, device=wm.device)
        with torch.no_grad():
            out = wm.model.embed(x, msgs=msgs, is_video=False)["imgs_w"]
        return ((out - x)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)
    if wm.name == "stegastamp":
        h, w = image.shape[:2]
        small = np.asarray(
            Image.fromarray(image).resize((400, 400), Image.Resampling.BICUBIC), np.float32
        )
        res = wm._run((small / 255)[None], ["residual"], np.asarray([bits], np.float32))[0][0] * 255
        res = res - res.mean(axis=(0, 1), keepdims=True)
        planes = [
            np.asarray(
                Image.fromarray(res[:, :, c], mode="F").resize((w, h), Image.Resampling.BILINEAR)
            )
            for c in range(3)
        ]
        return np.stack(planes, 2).astype(np.float32)
    return wm.embed(image, bits).astype(np.float32) - image.astype(np.float32)


def slopes(image: np.ndarray, delta: np.ndarray, device: str) -> dict[int, np.ndarray]:
    x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    d = torch.from_numpy(delta).permute(2, 0, 1)[None].to(device)
    out = {}
    for k in WINDOWS:
        pad = (k - CELL) // 2

        def mean(t, k=k, pad=pad):
            return F.avg_pool2d(t, k, stride=CELL, padding=pad)

        cov = (mean(x * d) - mean(x) * mean(d)).sum(1)
        var = (mean(d * d) - mean(d) ** 2).sum(1)
        out[k] = (cov / (var + 1e-6))[0].cpu().numpy()
    return out


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def bit_accuracy(wm, image: np.ndarray, bits: np.ndarray) -> float:
    x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div(255)
    return float((wm.decode(x)[0].cpu().numpy() == bits).mean())


def spread(values: np.ndarray) -> float:
    return float(np.subtract(*np.percentile(values, [75, 25])))


def quartiles(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return []
    return np.percentile(values, [0, 25, 50, 75, 100]).round(3).tolist()


def summarize(label: str, scores: list[np.ndarray], truths: list[np.ndarray]) -> str:
    if not scores:
        return f"{label}: no images"
    pairs = list(zip(scores, truths, strict=True))
    per_image = [auc(s, t) for s, t in pairs]
    pooled = auc(
        np.concatenate([s.ravel() for s in scores]), np.concatenate([t.ravel() for t in truths])
    )
    inside = np.concatenate([s[t == 1] for s, t in pairs])
    outside = np.concatenate([s[t == 0] for s, t in pairs])
    return (
        f"{label}: pooled AUC {pooled:.3f}, median per-image AUC {np.nanmedian(per_image):.3f}"
        f" | slope inside {np.median(inside):+.3f} (iqr {spread(inside):.3f}),"
        f" outside {np.median(outside):+.3f} (iqr {spread(outside):.3f})"
    )


def main() -> None:
    root = Path("release/train")
    meta = json.loads((root / "metadata.json").read_text())
    z = np.load(root / "masks_128.npz")
    grid = dict(zip(z["names"].tolist(), z["masks"], strict=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for method in METHODS:
        wm = load_watermark(method, models_dir=Path("models"), device=device)
        rows = [m for m in meta if m["method"] == method][:limit]
        amplitude, box_acc, frame_acc, exact = [], [], [], []
        full_true = {k: ([], []) for k in WINDOWS}
        full_null = {k: ([], []) for k in WINDOWS}
        rect_frame = {k: ([], []) for k in WINDOWS}
        rect_box = {k: ([], []) for k in WINDOWS}
        for m in rows:
            name = m["image_name"]
            image = np.asarray(Image.open(root / "images" / f"{name}.png").convert("RGB"))
            mask = (np.asarray(Image.open(root / "masks" / f"{name}.png")) > 127).astype(np.uint8)
            bits = np.array([int(c) for c in m["message"]], dtype=np.uint8)
            truth = grid[name]
            delta = pattern(wm, image, bits)
            amplitude.append(float(np.sqrt((delta**2).mean())))
            found = slopes(image, delta, device)
            if m["full_watermark"]:
                other = rng.integers(0, 2, size=len(bits)).astype(np.uint8)
                null = slopes(image, pattern(wm, image, other), device)
                for k in WINDOWS:
                    full_true[k][0].append(found[k])
                    full_true[k][1].append(truth)
                    full_null[k][0].append(null[k])
                    full_null[k][1].append(truth)
                continue
            y0, y1, x0, x1 = bbox(mask)
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            box_acc.append(bit_accuracy(wm, crop, bits))
            frame_acc.append(bit_accuracy(wm, image, bits))
            exact.append(bool(mask[y0:y1, x0:x1].all()))
            boxed = np.zeros_like(delta)
            boxed[y0:y1, x0:x1] = pattern(wm, crop, bits)
            in_box = slopes(image, boxed, device)
            for k in WINDOWS:
                rect_frame[k][0].append(found[k])
                rect_frame[k][1].append(truth)
                rect_box[k][0].append(in_box[k])
                rect_box[k][1].append(truth)
        box_acc, frame_acc, exact = np.array(box_acc), np.array(frame_acc), np.array(exact)
        print(f"== {method}: {len(rows)} images, pattern RMS {np.mean(amplitude):.2f} grey levels")
        print(
            f"  rectangle images, bit accuracy on the mask's bounding box: exact rectangles"
            f" {quartiles(box_acc[exact])} (n={int(exact.sum())}), others"
            f" {quartiles(box_acc[~exact])}; on the full frame {quartiles(frame_acc)}"
        )
        for k in WINDOWS:
            print(f"  window {k:2d} px")
            print("    " + summarize("full images, true message  ", *full_true[k]))
            print("    " + summarize("full images, random message", *full_null[k]))
            print("    " + summarize("rect images, frame pattern ", *rect_frame[k]))
            print("    " + summarize("rect images, box pattern   ", *rect_box[k]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
