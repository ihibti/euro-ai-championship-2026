"""Train a segmentation network for one watermark method on synthetic composites.

Background processes keep embedding fresh watermarks into clean covers with the organizers'
embedder (synth.py); the network trains on those plus a few real train images, and is scored with
the official metric on real train images it never saw (a fixed fifth of the split).

Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/train_seg.py --method pixelseal --minutes 50 --tag v1

Writes output/seg/<method>_<tag>/: log.jsonl and, refreshed at every evaluation, last.pt and
train.npz / test.npz holding names and probabilities on the 128 x 128 grid.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
import math
import queue
import sys
import threading
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
sys.path.insert(0, str(HERE))

from evidence_features import mean_miou  # noqa: E402
from synth import GRID, SIDE, make_aligned, make_sample, to_grid  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

VALIDATION_COVERS = 300
SYNTHETIC_VALIDATION = 150
FULL_DISK = (errno.EDQUOT, errno.ENOSPC)
DATASET_COVERS = Path("covers/dataset_512.npy")
DATASET_INDEX = Path("covers/dataset_512.json")
DATASET_VALIDATION = 100


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


def other_method_covers(method: str) -> tuple[list[int], list[int], list[dict]]:
    index = json.loads(DATASET_INDEX.read_text())
    others = [i for i, row in enumerate(index) if row["method"] != method]
    aligned = [
        i
        for i in others
        if index[i]["split"] == "train"
        and index[i]["kind"] in ("object", "rect")
        and not index[i]["held_out"]
    ]
    return (
        others[:-DATASET_VALIDATION],
        others[-DATASET_VALIDATION:],
        [index[i] | {"at": i} for i in aligned],
    )


def produce(
    samples, stop, boost, covers_path: Path, method: str, bank, seed: int, real_share, aligned_share
) -> None:
    covers = np.load(covers_path, mmap_mode="r")
    usable = len(covers) - VALIDATION_COVERS
    wm = load_watermark(method, models_dir=Path("models"), device="cuda")
    rng = np.random.default_rng(seed)
    pool, others, aligned = None, [], []
    if real_share > 0 or aligned_share > 0:
        pool = np.load(DATASET_COVERS, mmap_mode="r")
        others, _, aligned = other_method_covers(method)
        for row in aligned:
            mask = Image.open(Path("release/train/masks") / f"{row['name']}.png")
            row["mask"] = (np.asarray(mask) > 127).astype(np.uint8)
    failures = 0
    while not stop.is_set():
        ceiling = boost.value
        strength = float(np.exp(rng.uniform(0, np.log(ceiling)))) if ceiling > 1.0 else 1.0
        draw = rng.random()
        try:
            made = None
            if draw < aligned_share and aligned:
                row = aligned[int(rng.integers(len(aligned)))]
                cover = np.ascontiguousarray(pool[row["at"]])
                made = make_aligned(rng, cover, row["mask"], wm.embed, wm.nbits, method, strength)
            if made is None:
                if draw < aligned_share + real_share and others:
                    cover = pool[others[int(rng.integers(len(others)))]]
                else:
                    cover = covers[int(rng.integers(usable))]
                cover = np.ascontiguousarray(cover[:, ::-1] if rng.random() < 0.5 else cover)
                made = make_sample(rng, cover, wm.embed, wm.nbits, method, bank, strength)
        except Exception as error:
            failures += 1
            if failures <= 5:
                print(f"generator {seed}: {type(error).__name__}: {error}", flush=True)
            continue
        samples.put((made[0], to_grid(made[1])))


def collect(samples, buffer: Buffer, stop) -> None:
    while not stop.is_set():
        try:
            image, mask = samples.get(timeout=1)
        except queue.Empty:
            continue
        buffer.add(image, mask)


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
    def __init__(self, width: int = 32, highpass: bool = False):
        super().__init__()
        w = width
        self.highpass = highpass
        self.stem = nn.Sequential(conv_bn(9 if highpass else 3, w), conv_bn(w, w))
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

    def prepare(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            x = images.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
            if self.highpass:
                fine = x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
                coarse = x - F.avg_pool2d(x, 7, 1, 3, count_include_pad=False)
                x = torch.cat([x, torch.tanh(16 * fine), torch.tanh(16 * coarse)], 1)
        return x.contiguous(memory_format=torch.channels_last)

    def forward(self, images):
        s2 = self.d2(self.d1(self.stem(self.prepare(images))))
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        s5 = s5 + self.context(s5.mean((2, 3), keepdim=True))
        x = self.u4(self.up(s5, s4))
        x = self.u3(self.up(x, s3))
        x = self.u2(self.up(x, s2))
        return self.head(x)[:, 0]


EXTRACTOR_CHANNELS = {"pixelseal": (96, 192, 384, 768), "trustmark": (256, 512, 1024, 2048)}


def extractor_backbone(wm) -> nn.Module:
    core = copy.deepcopy(wm.det.convnext if wm.name == "pixelseal" else wm.net).float()
    for parameter in core.parameters():
        parameter.requires_grad_(True)
    return core


class ExtractorNet(nn.Module):
    """Segmentation on top of the method's own extractor, which already sees the watermark."""

    def __init__(self, method: str, backbone: nn.Module, width: int = 32):
        super().__init__()
        self.method = method
        self.backbone = backbone
        c1, c2, c3, c4 = EXTRACTOR_CHANNELS[method]
        w = width
        self.lat4 = nn.Conv2d(c4, 8 * w, 1)
        self.lat3 = nn.Conv2d(c3, 8 * w, 1)
        self.lat2 = nn.Conv2d(c2, 4 * w, 1)
        self.lat1 = nn.Conv2d(c1, 4 * w, 1)
        self.u3 = stage(16 * w, 8 * w, 1)
        self.u2 = stage(12 * w, 4 * w, 1)
        self.u1 = stage(8 * w, 4 * w, 1)
        self.fine = nn.Sequential(conv_bn(3, w, 2), conv_bn(w, 2 * w, 2), Block(2 * w))
        self.out = nn.Sequential(conv_bn(6 * w, 4 * w), Block(4 * w), nn.Conv2d(4 * w, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def decoder_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("backbone.")]

    def pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        if self.method == "pixelseal":
            for down, blocks in zip(
                self.backbone.downsample_layers, self.backbone.stages, strict=True
            ):
                x = blocks(down(x))
                features.append(x)
            return features
        core = self.backbone
        x = core.maxpool(core.relu(core.bn1(core.conv1(x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(core, name)(x)
            features.append(x)
        return features

    def forward(self, images):
        with torch.autocast("cuda", enabled=False):
            full = images.permute(0, 3, 1, 2).float().div(255)
            small = F.interpolate(
                full, size=(256, 256), mode="bilinear", align_corners=False, antialias=True
            )
            small = (small * 2 - 1).contiguous(memory_format=torch.channels_last)
            full = (full * 2 - 1).contiguous(memory_format=torch.channels_last)
        f1, f2, f3, f4 = self.pyramid(small)
        x = self.u3(Net.up(self.lat4(f4), self.lat3(f3)))
        x = self.u2(Net.up(x, self.lat2(f2)))
        x = self.u1(Net.up(x, self.lat1(f1)))
        return self.out(Net.up(x, self.fine(full)))[:, 0]


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
            logits = net(images[start : start + batch])
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def save_probs(path: Path, names: list[str], probs: np.ndarray) -> None:
    partial = path.with_suffix(".partial.npz")
    try:
        np.savez_compressed(partial, names=np.asarray(names), probs=probs.astype(np.float16))
        partial.replace(path)
    except OSError:
        partial.unlink(missing_ok=True)
        raise


def saved(label: str, write: Callable[..., object], *args: object) -> bool:
    """Run one write; on a full disk or quota, keep the previous file and go on training."""
    try:
        write(*args)
    except OSError as exc:
        if exc.errno not in FULL_DISK:
            raise
        print(f"skipped saving {label}: {exc}", file=sys.stderr, flush=True)
        return False
    return True


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
    parser.add_argument("--workers", type=int, default=3, help="generator processes")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--boost", type=float, default=1.0, help="starting watermark gain")
    parser.add_argument("--boost-fraction", type=float, default=0.35)
    parser.add_argument("--highpass", action="store_true", help="add fine-detail input channels")
    parser.add_argument(
        "--extractor",
        action="store_true",
        help="build on the method's own extractor (PixelSeal, TrustMark)",
    )
    parser.add_argument("--backbone-lr", type=float, default=0.3, help="relative to --lr")
    parser.add_argument("--all-real", action="store_true", help="train on the held-out fifth too")
    parser.add_argument("--covers", type=Path, default=Path("covers/ddb_512.npy"))
    parser.add_argument(
        "--real-covers",
        type=float,
        default=0.0,
        help="share of samples built on dataset images of the other methods",
    )
    parser.add_argument(
        "--aligned",
        type=float,
        default=0.0,
        help="share of samples that reuse the mask of another method's train image",
    )
    parser.add_argument(
        "--roll-real",
        action="store_true",
        help="shift real train images at random, half of the time",
    )
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
    on_dataset, on_dataset_truth = None, None
    if DATASET_COVERS.exists():
        pool = np.load(DATASET_COVERS, mmap_mode="r")
        made = [
            make_sample(rng, np.ascontiguousarray(pool[i]), wm.embed, wm.nbits, args.method, bank)
            for i in other_method_covers(args.method)[1]
        ]
        on_dataset = torch.from_numpy(np.stack([image for image, _, _ in made])).to(device)
        on_dataset_truth = np.stack([to_grid(mask) for _, mask, _ in made])

    backbone = extractor_backbone(wm) if args.extractor else None
    del wm
    torch.cuda.empty_cache()
    buffer = Buffer(args.buffer, device)
    context = mp.get_context("spawn")
    samples = context.Queue(maxsize=64)
    stop = context.Event()
    boost = context.Value("d", args.boost)
    workers = [
        context.Process(
            target=produce,
            args=(
                samples,
                stop,
                boost,
                args.covers,
                args.method,
                bank,
                100 + index,
                args.real_covers,
                args.aligned,
            ),
            daemon=True,
        )
        for index in range(args.workers)
    ]
    for worker in workers:
        worker.start()
    threading.Thread(target=collect, args=(samples, buffer, stop), daemon=True).start()
    while buffer.count < 200:
        time.sleep(1)

    if args.extractor:
        net = ExtractorNet(args.method, backbone, args.width)
        groups_of_parameters = [
            {"params": net.decoder_parameters(), "scale": 1.0},
            {"params": list(net.backbone.parameters()), "scale": args.backbone_lr},
        ]
    else:
        net = Net(args.width, args.highpass)
        groups_of_parameters = [{"params": list(net.parameters()), "scale": 1.0}]
    net = net.to(device).to(memory_format=torch.channels_last)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    ema = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(groups_of_parameters, lr=args.lr, weight_decay=1e-4)
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
                "boost": round(boost.value, 2),
                **score_groups(real_probs, truth, groups),
                "synthetic": round(
                    mean_miou((predict(ema, synthetic) >= 0.5).astype(np.uint8), synthetic_truth), 4
                ),
            }
            if on_dataset is not None:
                record["synthetic_on_dataset"] = round(
                    mean_miou((predict(ema, on_dataset) >= 0.5).astype(np.uint8), on_dataset_truth),
                    4,
                )
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            saved("last.pt", torch.save, ema.state_dict(), out_dir / "last.pt")
            saved("train.npz", save_probs, out_dir / "train.npz", names, real_probs)
            test_probs = predict(ema, test_images)
            saved("test.npz", save_probs, out_dir / "test.npz", test_names, test_probs)
            running = 0.0
        if done:
            break
        rate = (
            args.lr * min(1.0, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * elapsed / budget))
        )
        for group in optimizer.param_groups:
            group["lr"] = rate * group["scale"]
        fade = max(0.0, 1.0 - elapsed / (args.boost_fraction * budget))
        boost.value = 1.0 + (args.boost - 1.0) * fade
        images, masks = buffer.sample(args.batch - args.real)
        if args.real:
            pick = fit_index[torch.randint(0, len(fit_index), (args.real,), device=device)]
            picked, picked_masks = real[pick], real_truth[pick]
            if args.roll_real and torch.rand(()) < 0.5:
                dy, dx = (int(v) for v in torch.randint(0, GRID, (2,)))
                cell = SIDE // GRID
                picked = torch.roll(picked, (dy * cell, dx * cell), (1, 2))
                picked_masks = torch.roll(picked_masks, (dy, dx), (1, 2))
            images = torch.cat([images, picked])
            masks = torch.cat([masks, picked_masks])
        net.train()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(images)
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
    for worker in workers:
        worker.terminate()
    print(f"finished {out_dir} after {step} steps", flush=True)


if __name__ == "__main__":
    main()
