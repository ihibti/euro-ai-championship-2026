"""Does the known message localize the watermark at pixel level, once image content is removed?

For an image and its message, embed the message again and take the change as the expected
pattern. Subtract the mean pattern of a few random messages, so only the message-specific part
remains. Then correlate that pattern with the fine detail of the image in small windows, weighting
flat areas more. The same statistic is computed on generated composites, where the construction is
known, and on real train images.

Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/informed_probe.py [images per group]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
sys.path.insert(0, str(HERE))

from evidence_features import auc  # noqa: E402
from probe_process import pattern  # noqa: E402
from synth import make_sample, to_grid  # noqa: E402
from train_seg import kind_of, load_bank  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
WINDOWS = (8, 16, 32)
CELL = 4
RANDOM_MESSAGES = 3


def blur(t: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)


def window(t: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(t, k, stride=CELL, padding=(k - CELL) // 2, count_include_pad=False)


def statistics(image: np.ndarray, expected: np.ndarray, device: str) -> dict[str, np.ndarray]:
    x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    d = torch.from_numpy(expected).permute(2, 0, 1)[None].to(device)
    out = {}
    for hp in (3, 5):
        fine_x, fine_d = x - blur(x, hp), d - blur(d, hp)
        weight = 1.0 / (blur((fine_x**2).mean(1, keepdim=True), 7) + 1.0)
        for k in WINDOWS:
            plain = window(fine_x * fine_d, k).sum(1) / torch.sqrt(
                window(fine_x**2, k).sum(1) * window(fine_d**2, k).sum(1) + 1e-9
            )
            weighted = window(weight * fine_x * fine_d, k).sum(1) / torch.sqrt(
                window(weight * fine_d**2, k).sum(1) / (k * k) + 1e-9
            )
            out[f"hp{hp} w{k} plain"] = plain[0].cpu().numpy()
            out[f"hp{hp} w{k} weighted"] = weighted[0].cpu().numpy()
    return out


def expected_patterns(wm, image: np.ndarray, bits: np.ndarray, rng) -> dict[str, np.ndarray]:
    own = pattern(wm, image, bits)
    others = [
        pattern(wm, image, rng.integers(0, 2, len(bits)).astype(np.uint8))
        for _ in range(RANDOM_MESSAGES)
    ]
    specific = own - np.mean(others, axis=0)
    cover = np.clip(np.rint(image.astype(np.float32) - own), 0, 255).astype(np.uint8)
    refined = pattern(wm, cover, bits) - np.mean(
        [pattern(wm, cover, rng.integers(0, 2, len(bits)).astype(np.uint8)) for _ in range(2)],
        axis=0,
    )
    return {"own": own, "specific": specific, "refined": refined}


def report(label: str, rows: list[tuple[str, dict, np.ndarray]]) -> None:
    kinds = sorted({kind for kind, _, _ in rows})
    keys = list(rows[0][1])
    print(f"  {label}: " + ", ".join(f"{k} n={sum(r[0] == k for r in rows)}" for k in kinds))
    best = []
    for key in keys:
        cells = []
        for kind in kinds:
            group = [(s[key], t) for k, s, t in rows if k == kind]
            pooled = auc(
                np.concatenate([s.ravel() for s, _ in group]),
                np.concatenate([t.ravel() for _, t in group]),
            )
            each = np.nanmedian([auc(s, t) for s, t in group])
            cells.append(f"{kind} {pooled:.3f}/{each:.3f}")
            best.append((pooled, key, kind))
        print(f"    {key:28s} pooled/per-image AUC: " + "  ".join(cells))
    for kind in kinds:
        top = max(b for b in best if b[2] == kind)
        print(f"    best for {kind}: {top[1]} ({top[0]:.3f})")


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    methods = sys.argv[2].split(",") if len(sys.argv) > 2 else METHODS
    root = Path("release/train")
    meta = json.loads((root / "metadata.json").read_text())
    z = np.load(root / "masks_128.npz")
    grid = dict(zip(z["names"].tolist(), z["masks"], strict=True))
    covers = np.load("covers/ddb_512.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for method in methods:
        rng = np.random.default_rng(11)
        wm = load_watermark(method, models_dir=Path("models"), device=device)
        rows = [m for m in meta if m["method"] == method]
        bank = load_bank(root, rows)
        picked = [rows[int(i)] for i in rng.permutation(len(rows))[:limit]]
        real: dict[str, list] = {}
        for m in picked:
            name = m["image_name"]
            image = np.asarray(Image.open(root / "images" / f"{name}.png").convert("RGB"))
            bits = np.array([int(c) for c in m["message"]], dtype=np.uint8)
            truth = grid[name]
            if truth.min() == truth.max():
                continue
            for label, expected in expected_patterns(wm, image, bits, rng).items():
                stats = statistics(image, expected, device)
                real.setdefault(label, []).append((kind_of(m, truth), stats, truth))
        made: dict[str, list] = {}
        for _ in range(limit):
            cover = np.ascontiguousarray(covers[int(rng.integers(len(covers)))])
            bits = rng.integers(0, 2, wm.nbits).astype(np.uint8)
            image, mask, kind = make_sample(rng, cover, wm.embed, wm.nbits, method, bank, bits=bits)
            truth = to_grid(mask)
            if truth.min() == truth.max():
                continue
            for label, expected in expected_patterns(wm, image, bits, rng).items():
                stats = statistics(image, expected, device)
                made.setdefault(label, []).append((kind, stats, truth))
        print(f"== {method}")
        for label in ("own", "specific", "refined"):
            report(f"generated, pattern '{label}'", made[label])
            report(f"real train, pattern '{label}'", real[label])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
