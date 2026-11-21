"""Pack every dataset image into one array, to serve as realistic covers for the other methods.

An image that carries one method's watermark is a clean cover for the two other methods, because
each network only has to find its own method. The index records, per image, the split, the method
and, for train images, the kind of mask and whether the image is in the held-out fifth.

Run in the JURECA task folder:

    python <repo>/tasks/watermark_localization/build_dataset_covers.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
sys.path.insert(0, str(HERE))

from train_seg import held_out, kind_of  # noqa: E402

OUT = Path("covers/dataset_512")


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    index, paths = [], []
    grid_file = np.load("release/train/masks_128.npz")
    grid = dict(zip(grid_file["names"].tolist(), grid_file["masks"], strict=True))
    for split in ("train", "test"):
        for row in json.loads(Path(f"release/{split}/metadata.json").read_text()):
            name = row["image_name"]
            entry = {"name": name, "split": split, "method": row["method"]}
            if split == "train":
                entry["kind"] = kind_of(row, grid[name])
                entry["held_out"] = held_out(name)
            index.append(entry)
            paths.append(f"release/{split}/images/{name}.png")
    with Pool(8) as pool:
        images = np.stack(pool.map(load, paths, chunksize=16))
    np.save(OUT.with_suffix(".npy"), images)
    OUT.with_suffix(".json").write_text(json.dumps(index))
    print(f"wrote {OUT}.npy with {len(images)} images of shape {images.shape[1:]}")


if __name__ == "__main__":
    main()
