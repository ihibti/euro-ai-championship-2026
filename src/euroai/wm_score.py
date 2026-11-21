"""The official watermark-localization score, the per-image two-class mIoU, on the train split.

Same semantics as `miou_image` and `score` in the organizers' `wmloc/utils.py`, without torch.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRID = 128
RELEASE_DIR = Path("data") / "watermark_localization" / "release"
TRAIN_TRUTH = RELEASE_DIR / "train" / "masks_128.npz"
TRAIN_METADATA = RELEASE_DIR / "train" / "metadata.json"
TEST_METADATA = RELEASE_DIR / "test" / "metadata.json"


@dataclass(frozen=True)
class WatermarkScore:
    score: float
    per_image: dict[str, float]
    by_method: dict[str, float] = field(default_factory=dict)
    ignored: int = 0


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _grid_masks(masks: Any, label: str) -> np.ndarray:
    masks = _to_numpy(masks)
    if masks.ndim != 3 or masks.shape[1:] != (GRID, GRID):
        raise ValueError(f"{label} masks must have shape [N, 128, 128], got {list(masks.shape)}")
    return masks


def miou_per_image(preds: Any, gts: Any) -> np.ndarray:
    pred = _to_numpy(preds).astype(np.uint8)
    true = _to_numpy(gts).astype(np.uint8)
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(
            f"prediction shape {list(pred.shape)} does not match ground truth {list(true.shape)}"
        )
    total = np.zeros(pred.shape[0])
    for label in (0, 1):
        pred_c, true_c = pred == label, true == label
        union = np.logical_or(pred_c, true_c).sum(axis=(1, 2))
        inter = np.logical_and(pred_c, true_c).sum(axis=(1, 2))
        total += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return total / 2


def score_predictions(
    pred_names: Sequence[str],
    pred_masks: Any,
    gt_names: Sequence[str],
    gt_masks: Any,
    methods: Mapping[str, str] | None = None,
) -> WatermarkScore:
    pred_names = [str(name) for name in pred_names]
    gt_names = [str(name) for name in gt_names]
    pred_masks = _grid_masks(pred_masks, "predicted")
    gt_masks = _grid_masks(gt_masks, "ground-truth")
    if not gt_names:
        raise ValueError("no ground-truth images")
    if len(pred_names) != len(pred_masks):
        raise ValueError(f"{len(pred_masks)} predicted masks but {len(pred_names)} names")
    if len(gt_names) != len(gt_masks):
        raise ValueError(f"{len(gt_masks)} ground-truth masks but {len(gt_names)} names")
    duplicates = sorted(name for name, count in Counter(pred_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate predicted names: {', '.join(duplicates[:3])}")
    index = {name: position for position, name in enumerate(pred_names)}
    missing = [name for name in gt_names if name not in index]
    if missing:
        raise ValueError(
            f"no prediction for {len(missing)} of {len(gt_names)} train images, "
            f"e.g. {', '.join(missing[:3])}"
        )
    per_image = miou_per_image(pred_masks[[index[name] for name in gt_names]], gt_masks)
    by_name = dict(zip(gt_names, per_image.tolist(), strict=True))
    groups: dict[str, list[float]] = {}
    for name, value in by_name.items() if methods else ():
        groups.setdefault(methods.get(name, "unknown"), []).append(value)
    by_method = {method: float(np.mean(values)) for method, values in sorted(groups.items())}
    return WatermarkScore(
        score=float(per_image.mean()),
        per_image=by_name,
        by_method=by_method,
        ignored=len(pred_names) - len(gt_names),
    )


def read_mask_arrays(arrays: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    name_key = next((key for key in ("image_names", "names") if key in arrays), None)
    if name_key is None or "masks" not in arrays:
        raise ValueError("expected arrays image_names (or names) and masks")
    names = [str(name) for name in np.asarray(arrays[name_key]).tolist()]
    return names, _to_numpy(arrays["masks"])


def read_metadata(text: str) -> dict[str, str]:
    entries = json.loads(text)
    if not isinstance(entries, list):
        raise ValueError("metadata.json must hold a list of entries")
    methods: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "image_name" not in entry:
            raise ValueError("every metadata entry needs an image_name")
        methods[str(entry["image_name"])] = str(entry.get("method", "unknown"))
    return methods


def load_masks(
    path: str | Path, load_pt: Callable[[Path], Any] | None = None
) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as arrays:
            return read_mask_arrays(arrays)
    if path.suffix == ".pt":
        if load_pt is None:
            from euroai.validators import load_torch_file

            load_pt = load_torch_file
        return read_mask_arrays(load_pt(path))
    raise ValueError(f"masks must be in a .npz or .pt file, got {path.name}")
