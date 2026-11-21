import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from euroai.wm_score import (
    load_masks,
    miou_per_image,
    read_mask_arrays,
    read_metadata,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "tasks" / "watermark_localization" / "upstream"

UPSTREAM_SCORE = """
import importlib.util, json, sys, types
if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")
sys.path.insert(0, sys.argv[1])
import numpy as np
from wmloc.utils import miou_image, score
data = np.load(sys.argv[2])
preds, gts = data["preds"], data["gts"]
per_image = [float(miou_image(p, g)) for p, g in zip(preds, gts)]
print(json.dumps({"per_image": per_image, "score": score(preds, gts)}))
"""


def synthetic_masks():
    rng = np.random.default_rng(0)
    gts = np.zeros((8, 128, 128), np.uint8)
    preds = np.zeros_like(gts)
    gts[1, 10:50, 20:90] = 1
    preds[2] = 1
    gts[2] = 1
    gts[3, :64] = 1
    preds[3, 32:96] = 1
    preds[4] = rng.integers(0, 2, (128, 128))
    gts[4] = rng.integers(0, 2, (128, 128))
    preds[5, :5, :5] = 1
    gts[6] = 1
    preds[7] = rng.random((128, 128)) > 0.9
    gts[7] = rng.random((128, 128)) > 0.5
    return preds, gts


def test_miou_hand_computed_cases():
    preds, gts = synthetic_masks()
    per_image = miou_per_image(preds, gts)
    assert per_image[0] == 1.0
    assert per_image[2] == 1.0
    assert per_image[3] == pytest.approx(1 / 3)
    assert per_image[6] == 0.0


def test_miou_matches_vendored_utils(tmp_path):
    preds, gts = synthetic_masks()
    arrays = tmp_path / "masks.npz"
    np.savez(arrays, preds=preds, gts=gts)
    run = subprocess.run(
        [sys.executable, "-I", "-c", UPSTREAM_SCORE, str(KIT), str(arrays)],
        capture_output=True,
        text=True,
        check=True,
    )
    upstream = json.loads(run.stdout)
    ours = miou_per_image(preds, gts)
    assert ours.tolist() == upstream["per_image"]
    assert float(ours.mean()) == upstream["score"]


def test_miou_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        miou_per_image(np.zeros((2, 128, 128)), np.zeros((3, 128, 128)))


def test_score_aligns_by_name_and_ignores_extras():
    preds, gts = synthetic_masks()
    names = [f"img_{i}" for i in range(len(gts))]
    order = [7, 3, 0, 5, 1, 6, 2, 4]
    shuffled_names = [names[i] for i in order] + ["test_only"]
    shuffled_preds = np.concatenate([preds[order], np.zeros((1, 128, 128), np.uint8)])
    result = score_predictions(shuffled_names, shuffled_preds, names, gts)
    assert result.score == float(miou_per_image(preds, gts).mean())
    assert result.per_image["img_3"] == pytest.approx(1 / 3)
    assert result.ignored == 1
    assert result.by_method == {}


def test_score_missing_prediction_raises():
    preds, gts = synthetic_masks()
    names = [f"img_{i}" for i in range(len(gts))]
    with pytest.raises(ValueError, match="no prediction for 1 of 8"):
        score_predictions(names[:-1], preds[:-1], names, gts)


def test_score_duplicate_prediction_raises():
    preds, gts = synthetic_masks()
    names = [f"img_{i}" for i in range(len(gts))]
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([*names[:-1], names[0]], preds, names, gts)


def test_score_by_method():
    preds, gts = synthetic_masks()
    names = [f"img_{i}" for i in range(len(gts))]
    methods = {name: ("pixelseal" if i < 4 else "stegastamp") for i, name in enumerate(names)}
    result = score_predictions(names, preds, names, gts, methods=methods)
    per_image = miou_per_image(preds, gts)
    assert result.by_method["pixelseal"] == pytest.approx(per_image[:4].mean())
    assert result.by_method["stegastamp"] == pytest.approx(per_image[4:].mean())


def test_read_mask_arrays_accepts_both_key_styles():
    masks = np.zeros((2, 128, 128), np.uint8)
    names = np.array(["a", "b"])
    assert read_mask_arrays({"names": names, "masks": masks})[0] == ["a", "b"]
    assert read_mask_arrays({"image_names": names, "masks": masks})[0] == ["a", "b"]
    with pytest.raises(ValueError, match="image_names"):
        read_mask_arrays({"masks": masks})


def test_read_metadata():
    text = json.dumps(
        [
            {"image_name": "a", "method": "trustmark", "message": "0101"},
            {"image_name": "b", "method": "pixelseal", "message": "1"},
        ]
    )
    assert read_metadata(text) == {"a": "trustmark", "b": "pixelseal"}
    with pytest.raises(ValueError, match="image_name"):
        read_metadata(json.dumps([{"method": "x"}]))


def test_score_needs_ground_truth():
    with pytest.raises(ValueError, match="no ground-truth"):
        score_predictions([], np.zeros((0, 128, 128)), [], np.zeros((0, 128, 128)))


def test_load_masks_npz_and_pt(tmp_path):
    masks = np.zeros((2, 128, 128), np.uint8)
    path = tmp_path / "pred.npz"
    np.savez(path, image_names=np.array(["a", "b"]), masks=masks)
    names, loaded = load_masks(path)
    assert names == ["a", "b"] and loaded.shape == (2, 128, 128)
    pt = tmp_path / "pred.pt"
    pt.write_bytes(b"x")
    names, _ = load_masks(pt, load_pt=lambda _: {"image_names": ["c"], "masks": masks[:1]})
    assert names == ["c"]
    with pytest.raises(ValueError, match=".npz or .pt"):
        load_masks(tmp_path / "pred.csv")
