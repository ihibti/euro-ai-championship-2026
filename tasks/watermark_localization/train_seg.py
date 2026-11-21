"""Train a segmentation network for one watermark method on synthetic composites.

A background thread keeps embedding fresh watermarks into clean covers with the organizers' embedder
(synth.py); the network trains on those plus a few real train images, and is scored with the
official metric on real train images it never saw (a fixed fifth of the split).

Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/train_seg.py --method pixelseal --minutes 50 --tag v1

Writes output/seg/<method>_<tag>/: log.jsonl and, refreshed at every evaluation, last.pt and
train.npz / test.npz holding names and probabilities on the 128 x 128 grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
sys.path.insert(0, str(HERE))

from evidence_features import mean_miou  # noqa: E402
from synth import GRID, SIDE, make_sample, to_grid  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

VALIDATION_COVERS = 300
SYNTHETIC_VALIDATION = 150


def held_out(name: str) -> bool:
    return zlib.crc32(name.encode()) % 5 == 0


def kind_of(row: dict, grid_mask: np.ndarray) -> str:
    if row.get("full_watermark"):
        return "hole"
    ys, xs = np.nonzero(grid_mask)
    if len(ys) and grid_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].all():
        return "rect"
    return "object"


def load_images(folder: Path, names: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(Image.open(folder / f"{name}.png").convert("RGB")) for name in names]
    )


def load_bank(root: Path, rows: list[dict]) -> list[np.ndarray]:
    bank = []
    for row in rows:
        mask = (np.asarray(Image.open(root / "masks" / f"{row['image_name']}.png")) > 127).astype(
            np.uint8
        )
        if row["full_watermark"]:
            bank.append(1 - mask)
            continue
        ys, xs = np.nonzero(mask)
        if not mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].all():
            bank.append(mask)
    return bank


class Buffer:
    def __init__(self, capacity: int, device: str):
        self.images = torch.zeros((capacity, SIDE, SIDE, 3), dtype=torch.uint8, device=device)
        self.masks = torch.zeros((capacity, GRID, GRID), dtype=torch.uint8, device=device)
        self.capacity, self.count, self.cursor, self.total = capacity, 0, 0, 0
        self.lock = threading.Lock()

    def add(self, image: np.ndarray, mask: np.ndarray) -> None:
        image_t = torch.from_numpy(np.ascontiguousarray(image)).to(self.images.device)
        mask_t = torch.from_numpy(mask).to(self.masks.device)
        with self.lock:
            self.images[self.cursor] = image_t
            self.masks[self.cursor] = mask_t
            self.cursor = (self.cursor + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
            self.total += 1

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = torch.randint(0, self.count, (n,), device=self.images.device)
        return self.images[index], self.masks[index]


def generate(buffer, covers, wm, method, bank, seed, stop) -> None:
    rng = np.random.default_rng(seed)
    failures = 0
    while not stop.is_set():
        cover = np.ascontiguousarray(covers[int(rng.integers(len(covers)))])
        if rng.random() < 0.5:
            cover = np.ascontiguousarray(cover[:, ::-1])
        try:
            image, mask, _ = make_sample(rng, cover, wm.embed, wm.nbits, method, bank)
        except Exception as error:
            failures += 1
            if failures <= 5:
                print(f"generator: {type(error).__name__}: {error}", flush=True)
            continue
        buffer.add(image, to_grid(mask))


def conv_bn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(channels, channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.body(x))


def stage(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(conv_bn(cin, cout, stride), Block(cout), Block(cout))


class Net(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv_bn(3, w), conv_bn(w, w))
        self.d1 = stage(w, 2 * w, 2)
        self.d2 = stage(2 * w, 4 * w, 2)
        self.d3 = stage(4 * w, 8 * w, 2)
        self.d4 = stage(8 * w, 12 * w, 2)
        self.d5 = stage(12 * w, 16 * w, 2)
        self.context = nn.Sequential(nn.Conv2d(16 * w, 16 * w, 1), nn.ReLU(inplace=True))
        self.u4 = stage(28 * w, 12 * w, 1)
        self.u3 = stage(20 * w, 8 * w, 1)
        self.u2 = stage(12 * w, 4 * w, 1)
        self.head = nn.Conv2d(4 * w, 1, 1)

    @staticmethod
    def up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        s2 = self.d2(self.d1(self.stem(x)))
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        s5 = s5 + self.context(s5.mean((2, 3), keepdim=True))
        x = self.u4(self.up(s5, s4))
        x = self.u3(self.up(x, s3))
        x = self.u2(self.up(x, s2))
        return self.head(x)[:, 0]


def prepare(images: torch.Tensor) -> torch.Tensor:
    x = images.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    return x.contiguous(memory_format=torch.channels_last)


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter1 = (p * target).sum((1, 2))
    union1 = p.sum((1, 2)) + target.sum((1, 2)) - inter1
    inter0 = ((1 - p) * (1 - target)).sum((1, 2))
    union0 = (1 - p).sum((1, 2)) + (1 - target).sum((1, 2)) - inter0
    iou = 0.5 * ((inter1 + 1) / (union1 + 1) + (inter0 + 1) / (union0 + 1))
    return bce + (1 - iou).mean()


@torch.no_grad()
def predict(net: nn.Module, images: torch.Tensor, batch: int = 32) -> np.ndarray:
    net.eval()
    out = []
    for start in range(0, len(images), batch):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(prepare(images[start : start + batch]))
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def save_probs(path: Path, names: list[str], probs: np.ndarray) -> None:
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, names=np.asarray(names), probs=probs.astype(np.float16))
    partial.replace(path)


def score_groups(probs: np.ndarray, truth: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    pred = (probs >= 0.5).astype(np.uint8)
    return {
        label: round(mean_miou(pred[index], truth[index]), 4) if index.any() else None
        for label, index in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--minutes", type=float, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--real", type=int, default=2, help="real train images per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--buffer", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--all-real", action="store_true", help="train on the held-out fifth too")
    parser.add_argument("--covers", type=Path, default=Path("covers/ddb_512.npy"))
    args = parser.parse_args()

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    out_dir = Path("output/seg") / f"{args.method}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path("release/train")
    everything = json.loads((root / "metadata.json").read_text())
    z = np.load(root / "masks_128.npz")
    grid = dict(zip(z["names"].tolist(), z["masks"], strict=True))

    rows = [row for row in everything if row["method"] == args.method]
    names = [row["image_name"] for row in rows]
    truth = np.stack([grid[name] for name in names]).astype(np.uint8)
    kinds = np.array([kind_of(row, grid[row["image_name"]]) for row in rows])
    val = np.array([held_out(name) for name in names])
    fit = np.ones(len(names), bool) if args.all_real else ~val
    real = torch.from_numpy(load_images(root / "images", names)).to(device)
    real_truth = torch.from_numpy(truth).to(device)
    fit_index = torch.from_numpy(np.flatnonzero(fit)).to(device)
    bank = load_bank(
        root, [row for row in everything if args.all_real or not held_out(row["image_name"])]
    )
    print(
        f"{args.method}: {len(names)} real images, {int(val.sum())} held out, "
        f"{len(bank)} shapes in the bank",
        flush=True,
    )

    test_names = [
        row["image_name"]
        for row in json.loads(Path("release/test/metadata.json").read_text())
        if row["method"] == args.method
    ]
    test_images = torch.from_numpy(load_images(Path("release/test/images"), test_names)).to(device)

    covers = np.load(args.covers, mmap_mode="r")
    wm = load_watermark(args.method, models_dir=Path("models"), device=device)
    rng = np.random.default_rng(1)
    synthetic, synthetic_truth = [], []
    for index in range(SYNTHETIC_VALIDATION):
        cover = np.ascontiguousarray(covers[len(covers) - 1 - index % VALIDATION_COVERS])
        image, mask, _ = make_sample(rng, cover, wm.embed, wm.nbits, args.method, bank)
        synthetic.append(image)
        synthetic_truth.append(to_grid(mask))
    synthetic = torch.from_numpy(np.stack(synthetic)).to(device)
    synthetic_truth = np.stack(synthetic_truth)

    buffer = Buffer(args.buffer, device)
    stop = threading.Event()
    worker = threading.Thread(
        target=generate,
        args=(buffer, covers[: len(covers) - VALIDATION_COVERS], wm, args.method, bank, 2, stop),
        daemon=True,
    )
    worker.start()
    while buffer.count < 200:
        time.sleep(1)

    net = Net(args.width).to(device).to(memory_format=torch.channels_last)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    ema = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    groups = {
        "held_out": val,
        "held_out_hole": val & (kinds == "hole"),
        "held_out_rect": val & (kinds == "rect"),
        "held_out_object": val & (kinds == "object"),
        "seen": ~val,
    }
    budget = args.minutes * 60
    started = time.time()
    step, running = 0, 0.0
    log = (out_dir / "log.jsonl").open("a")
    while True:
        elapsed = time.time() - started
        done = elapsed >= budget
        if step % args.eval_every == 0 or done:
            real_probs = predict(ema, real)
            record = {
                "step": step,
                "minutes": round(elapsed / 60, 1),
                "loss": round(running / max(1, min(step, args.eval_every)), 4),
                "generated": buffer.total,
                **score_groups(real_probs, truth, groups),
                "synthetic": round(
                    mean_miou((predict(ema, synthetic) >= 0.5).astype(np.uint8), synthetic_truth), 4
                ),
            }
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            torch.save(ema.state_dict(), out_dir / "last.pt")
            save_probs(out_dir / "train.npz", names, real_probs)
            save_probs(out_dir / "test.npz", test_names, predict(ema, test_images))
            running = 0.0
        if done:
            break
        rate = (
            args.lr * min(1.0, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * elapsed / budget))
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        images, masks = buffer.sample(args.batch - args.real)
        if args.real:
            pick = fit_index[torch.randint(0, len(fit_index), (args.real,), device=device)]
            images = torch.cat([images, real[pick]])
            masks = torch.cat([masks, real_truth[pick]])
        net.train()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(prepare(images))
        loss = seg_loss(logits.float(), masks.float())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for averaged, live in zip(ema.parameters(), net.parameters(), strict=True):
                averaged.lerp_(live, 0.002)
            for averaged, live in zip(ema.buffers(), net.buffers(), strict=True):
                averaged.copy_(live)
        running += float(loss)
        step += 1
    stop.set()
    print(f"finished {out_dir} after {step} steps", flush=True)


if __name__ == "__main__":
    main()
