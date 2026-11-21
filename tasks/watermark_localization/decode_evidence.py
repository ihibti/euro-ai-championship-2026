"""Decoder evidence for every image of a split.

For each image, with its known method and message, record how well the message is read from:
- square crops of several sizes (the decoder rescales each crop to its working resolution);
- the full frame with everything outside a square greyed or blurred (the watermark keeps its scale);
- each position of the extractor's last feature map, for PixelSeal and TrustMark.

Run in the JURECA task folder, which holds release/ and models/:

    python <repo>/tasks/watermark_localization/decode_evidence.py --split train

Writes output/evidence/<split>_<method>.npz. "margin" is the mean over bits of logit x sign of the true
bit; "acc" is the share of bits read correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))

from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
FULL = 512
GRID = 128
CROPS = ((512, 512), (384, 64), (256, 64), (192, 64), (128, 32))
KEEPS = ((256, 64), (128, 64))
FILLS = ("grey", "blur")
SLOW_CROPS = ((512, 512), (256, 128))
SLOW_KEEPS = ((256, 128),)
SLOW_FILLS = ("grey",)
WINDOWS_PER_CALL = 256
MAP_BATCH = 4
BLUR_SIGMA = 6.0


def origins(size: int, stride: int) -> list[tuple[int, int]]:
    steps = range(0, FULL - size + 1, stride)
    return [(y, x) for y in steps for x in steps]


def load_images(folder: Path, names: list[str]) -> torch.Tensor:
    arrays = [np.asarray(Image.open(folder / f"{name}.png").convert("RGB")) for name in names]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def read_bits(wm, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if wm.name == "pixelseal":
        with torch.no_grad():
            out = wm.det(wm._resize(batch.to(wm.device))).float()
        return out[:, 1:], out[:, 0]
    return wm.logits(batch).to(batch.device), None


def crop_windows(x: torch.Tensor, size: int, where: list[tuple[int, int]]) -> torch.Tensor:
    return torch.stack([x[:, :, y : y + size, c : c + size] for y, c in where], 1).flatten(0, 1)


def keep_windows(
    x: torch.Tensor, size: int, where: list[tuple[int, int]], fill: str
) -> torch.Tensor:
    base = torch.full_like(x, 0.5) if fill == "grey" else gaussian_blur(x, 37, BLUR_SIGMA)
    out = base[:, None].repeat(1, len(where), 1, 1, 1)
    for k, (y, c) in enumerate(where):
        out[:, k, :, y : y + size, c : c + size] = x[:, :, y : y + size, c : c + size]
    return out.flatten(0, 1)


def window_evidence(
    wm, images: torch.Tensor, sign: torch.Tensor, kind: str, size: int, stride: int
):
    where = origins(size, stride)
    per_call = max(1, WINDOWS_PER_CALL // len(where))
    margins, accs, dets = [], [], []
    for start in range(0, len(images), per_call):
        x = images[start : start + per_call].float() / 255
        if kind == "crop":
            windows = crop_windows(x, size, where)
        else:
            windows = keep_windows(x, size, where, kind)
        logits, det = read_bits(wm, windows)
        signed = logits * sign[start : start + per_call].repeat_interleave(len(where), 0)
        margins.append(signed.mean(1).view(len(x), -1).cpu())
        accs.append((signed > 0).float().mean(1).view(len(x), -1).cpu())
        if det is not None:
            dets.append(det.view(len(x), -1).cpu())
    out = {
        "margin": torch.cat(margins).numpy().astype(np.float16),
        "acc": torch.cat(accs).numpy().astype(np.float16),
        "origins": np.asarray(where, dtype=np.int16),
    }
    if dets:
        out["det"] = torch.cat(dets).numpy().astype(np.float16)
    return out


def to_grid(maps: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        maps[:, None].float(), size=(GRID, GRID), mode="bilinear", align_corners=False
    )[:, 0]


def pixelseal_positions(wm, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    linear = wm.det.pixel_decoder.linear
    for tag, inputs in (("p256", wm._resize(x)), ("p512", x)):
        embedding = wm.det.pixel_decoder.output_upscaling(
            wm.det.convnext(wm.det.preprocess(inputs))
        )
        out[tag] = (
            torch.einsum("bchw,oc->bohw", embedding, linear.weight)
            + linear.bias[None, :, None, None]
        )
    return out


def trustmark_positions(wm, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    net = wm.net
    for tag, side in (("p245", 245), ("p490", 490)):
        z = F.interpolate(
            x, size=(side, side), mode="bilinear", align_corners=False, antialias=True
        )
        z = net.maxpool(net.relu(net.bn1(net.conv1(z * 2 - 1))))
        for layer in ("layer1", "layer2", "layer3", "layer4"):
            z = getattr(net, layer)(z)
        out[tag] = (
            torch.einsum("bchw,oc->bohw", z, net.fc.weight) + net.fc.bias[None, :, None, None]
        )
    return out


@torch.no_grad()
def position_evidence(wm, images: torch.Tensor, sign: torch.Tensor) -> dict[str, np.ndarray]:
    if wm.name == "stegastamp":
        return {}
    compute = pixelseal_positions if wm.name == "pixelseal" else trustmark_positions
    skip = 1 if wm.name == "pixelseal" else 0
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(images), MAP_BATCH):
        x = images[start : start + MAP_BATCH].float() / 255
        s = sign[start : start + MAP_BATCH][:, :, None, None]
        for tag, per_position in compute(wm, x).items():
            signed = per_position[:, skip:].float() * s
            if start == 0:
                print(f"  {tag}: per-position map {tuple(per_position.shape[1:])}", flush=True)
            collected.setdefault(f"{tag}_margin", []).append(to_grid(signed.mean(1)).cpu())
            collected.setdefault(f"{tag}_acc", []).append(
                to_grid((signed > 0).float().mean(1)).cpu()
            )
            if skip:
                collected.setdefault(f"{tag}_det", []).append(
                    to_grid(per_position[:, 0].float()).cpu()
                )
    return {key: torch.cat(parts).numpy().astype(np.float16) for key, parts in collected.items()}


def run_method(
    method: str, split: str, root: Path, out_dir: Path, device: str, limit: int | None
) -> None:
    everything = json.loads((root / split / "metadata.json").read_text())
    meta = [m for m in everything if m["method"] == method][:limit]
    names = [m["image_name"] for m in meta]
    started = time.time()
    wm = load_watermark(method, models_dir=Path("models"), device=device)
    images = load_images(root / split / "images", names).to(device)
    bits = torch.tensor([[int(c) for c in m["message"]] for m in meta], dtype=torch.float32)
    sign = (bits * 2 - 1).to(device)
    print(f"{method}: {len(names)} images loaded in {time.time() - started:.0f}s", flush=True)

    slow = device == "cpu"
    if method == "stegastamp":
        providers = wm.sess.get_providers()
        print(f"  onnxruntime providers: {providers}", flush=True)
        slow = slow or providers[0] != "CUDAExecutionProvider"
    crops, keeps, fills = (SLOW_CROPS, SLOW_KEEPS, SLOW_FILLS) if slow else (CROPS, KEEPS, FILLS)
    if slow:
        print("  no GPU for this decoder: reduced window set", flush=True)

    arrays: dict[str, np.ndarray] = {"names": np.asarray(names)}
    specs = [("crop", size, stride) for size, stride in crops]
    specs += [(fill, size, stride) for fill in fills for size, stride in keeps]
    for kind, size, stride in specs:
        tick = time.time()
        result = window_evidence(wm, images, sign, kind, size, stride)
        label = f"{kind}{size}"
        for key, value in result.items():
            arrays[f"{label}_{key}"] = value
        count = result["acc"].size
        print(
            f"  {label}: {count} windows in {time.time() - tick:.0f}s, "
            f"mean acc {float(result['acc'].astype(np.float32).mean()):.3f}",
            flush=True,
        )
    try:
        arrays.update(position_evidence(wm, images, sign))
    except Exception as error:
        print(f"  per-position maps failed: {type(error).__name__}: {error}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{split}_{method}.npz", **arrays)
    print(f"{method}: wrote {split}_{method}.npz after {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--root", type=Path, default=Path("release"))
    parser.add_argument("--out", type=Path, default=Path("output/evidence"))
    parser.add_argument("--limit", type=int, default=None, help="first N images per method")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for method in args.methods:
        run_method(method, args.split, args.root, args.out, device, args.limit)


if __name__ == "__main__":
    main()
