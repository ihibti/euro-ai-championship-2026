"""Turn a candidate written by assemble.py into the train answer and the test submission file.

    uv run --extra torch python tasks/watermark_localization/pack_submission.py <candidate.npz>

Writes <tag>_train.npz (for make score) and <tag>_test.pt (for make check and make submit) next to
the candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GRID = 128


def unpack(bits: np.ndarray, count: int) -> np.ndarray:
    flat = np.unpackbits(bits)[: count * GRID * GRID]
    return flat.reshape(count, GRID, GRID).astype(np.uint8)


def main() -> None:
    source = Path(sys.argv[1])
    candidate = np.load(source)
    train_names = candidate["train_names"].tolist()
    test_names = candidate["test_names"].tolist()
    train = unpack(candidate["train_masks"], len(train_names))
    test = unpack(candidate["test_masks"], len(test_names))
    train_path = source.with_name(f"{source.stem}_train.npz")
    test_path = source.with_name(f"{source.stem}_test.pt")
    np.savez_compressed(train_path, image_names=np.asarray(train_names), masks=train)
    torch.save({"image_names": test_names, "masks": torch.from_numpy(test).bool()}, test_path)
    print(
        f"wrote {train_path} ({len(train_names)} masks) and {test_path} ({len(test_names)} masks)"
    )
    print(f"test cells marked: {test.mean():.3f}")


if __name__ == "__main__":
    main()
