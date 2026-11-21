"""Pack clean cover images into one uint8 array of 512 x 512 centre crops, without resampling.

Run in the JURECA task folder after covers/fetch.sh:

    python <repo>/tasks/watermark_localization/build_covers.py

Writes covers/ddb_512.npy with shape [N, 512, 512, 3].
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 512
SOURCE = Path("covers/ddb")
TARGET = Path("covers/ddb_512.npy")


def large_enough(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return min(image.size) >= SIDE
    except OSError:
        return False


def centre_crop(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        width, height = image.size
        left, top = (width - SIDE) // 2, (height - SIDE) // 2
        return np.asarray(image.convert("RGB").crop((left, top, left + SIDE, top + SIDE)))


def main() -> None:
    files = sorted(SOURCE.glob("*/*.png"))
    with Pool(8) as pool:
        keep = pool.map(large_enough, files, chunksize=64)
        files = [path for path, ok in zip(files, keep, strict=True) if ok]
        out = np.lib.format.open_memmap(
            TARGET, mode="w+", dtype=np.uint8, shape=(len(files), SIDE, SIDE, 3)
        )
        for index, crop in enumerate(pool.imap(centre_crop, files, chunksize=16)):
            out[index] = crop
        out.flush()
    print(f"wrote {TARGET} with {len(files)} covers")


if __name__ == "__main__":
    main()
