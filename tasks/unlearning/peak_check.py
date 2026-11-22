"""Predict candidates from the scores of sharpened fine-tunes, without the ledger's single-model golds.

The 03:08 score (one notch more sharpening, 0.2837 to 0.2222) showed a peak in the forget-set loss
that the ledger cannot represent. This check scores each line as

    mean/spread term x prediction term x retain

- the mean/spread term against a gold loss level (mean, spread), as in the metric;
- the prediction term from KL at temperature 2 to a reference target: the cross-fitted step
  extrapolated 3 times (`K3t1`), or a mixture of the steps 1 to 4 at temperature 2 (`mixK`);
- retain from the spill probes (`ledger.sharpened_retain`).

Each setting (gold level, reference) is scaled so that the anchor line matches its score. A
setting is kept when it then reproduces every other observed line within the tolerance. The
prediction for a candidate is the mean, minimum and maximum over the kept settings.

    cd <team folder> && python <repo>/tasks/unlearning/peak_check.py \\
        --logits /Volumes/euroai-apfs/r21/seventh/logits.npz ... [--tolerance 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import ledger  # noqa: E402

GOLD_LEVELS = (
    (3.3, 3.9),
    (3.5, 3.5),
    (3.5, 4.0),
    (3.6, 3.6),
    (3.7, 3.7),
    (3.85, 3.85),
    (4.0, 3.5),
    (4.0, 4.0),
    (4.2, 4.2),
    (4.6, 4.6),
)
OBSERVED = (
    ("k8_e3t2_rm1_lr1e4", 2.0, 0.2837),
    ("k8_e3t2_rm1_lr1e4", 2.5, 0.2222),
)
GRID = (
    ("k7_e3t2_rm", (2.0,)),
    ("k8_e3t2_rm1_lr1e4", (1.75, 1.9, 2.0, 2.1, 2.5)),
    ("k9_e3t1_lr1e4", (1.0, 1.15, 1.25, 1.333, 1.35, 1.5)),
    ("k9_e3t2_lr1e4_ep40", (1.6, 1.75, 1.9, 2.0)),
    ("k9_e3t2_lr2e4_ep40", (1.5, 1.6, 1.75, 2.0)),
)


def kl_t(p_logits: torch.Tensor, q_logits: torch.Tensor, t: float = 2.0) -> float:
    p = F.log_softmax(p_logits / t, 1)
    q = F.log_softmax(q_logits / t, 1)
    return float((p.exp() * (p - q)).sum(1).mean())


def ms_term(mean: float, std: float, gold_mean: float, gold_std: float) -> float:
    gap = abs(mean - gold_mean) + abs(std - gold_std)
    return max(0.0, 1.0 - gap / (gold_mean + gold_std))


def pred_term(ref: torch.Tensor, sub: torch.Tensor, orig: torch.Tensor) -> float:
    return max(0.0, 1.0 - kl_t(ref, sub) / kl_t(ref, orig))


def references(orig: torch.Tensor, step: torch.Tensor) -> dict[str, torch.Tensor]:
    mix = torch.stack([F.softmax((orig + k * step) / 2, 1) for k in (1, 2, 3, 4)]).mean(0)
    return {"K3t1": orig + 3 * step, "mixK": 2 * torch.log(mix)}


def features(
    logits: torch.Tensor,
    factor: float,
    orig: torch.Tensor,
    labels: torch.Tensor,
    spill: dict,
    refs: dict,
) -> dict:
    sharp = logits * factor
    loss = F.cross_entropy(sharp, labels, reduction="none")
    return {
        "mean": float(loss.mean()),
        "std": float(loss.std()),
        "retain": ledger.sharpened_retain(spill, factor, orig, logits),
        "pred": {name: pred_term(ref, sharp, orig) for name, ref in refs.items()},
    }


def raw_score(feat: dict, gold: tuple[float, float], ref: str) -> float:
    return ms_term(feat["mean"], feat["std"], *gold) * feat["pred"][ref] * feat["retain"]


def calibrated(
    feats: dict, observed: tuple, golds: tuple, refs: tuple, tolerance: float
) -> tuple[list, dict]:
    """Keep the settings that reproduce every observed line once scaled on the first one."""
    anchor_key, anchor_score = observed[0][:2], observed[0][2]
    kept = []
    for gold in golds:
        for ref in refs:
            anchor_raw = raw_score(feats[anchor_key], gold, ref)
            if anchor_raw <= 0:
                continue
            scale = anchor_score / anchor_raw
            errors = [
                abs(raw_score(feats[(n, f)], gold, ref) * scale - s) for n, f, s in observed[1:]
            ]
            if all(e <= tolerance for e in errors):
                kept.append((gold, ref, scale))
    table = {}
    for key, feat in feats.items():
        values = [raw_score(feat, gold, ref) * scale for gold, ref, scale in kept]
        if values:
            table[key] = (float(np.mean(values)), min(values), max(values))
    return kept, table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, default=Path("output/kkrgp/xfit_extras.npz"))
    parser.add_argument("--orig", type=Path, default=Path("output/kkrgp/orig_exact.npy"))
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("ledger_inputs.json")
    )
    parser.add_argument("--logits", type=Path, nargs="+", required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    orig = torch.from_numpy(np.load(args.orig)).float()
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        pred = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    step = torch.zeros_like(orig)
    step[:, columns] = pred[:, columns] - orig[:, columns]
    refs = references(orig, step)
    spills = {
        n: s["spill"]
        for n, s in json.loads(args.inputs.read_text())["candidates"].items()
        if "spill" in s
    }
    dumped = {}
    for path in args.logits:
        with np.load(path) as data:
            for key in data.files:
                if key.endswith("__exact"):
                    dumped[key[: -len("__exact")]] = torch.from_numpy(data[key].astype(np.float32))
    feats = {}
    for name, factors in GRID:
        if name not in dumped or name not in spills:
            continue
        for factor in factors:
            feats[(name, factor)] = features(dumped[name], factor, orig, labels, spills[name], refs)
    kept, table = calibrated(feats, OBSERVED, GOLD_LEVELS, tuple(refs), args.tolerance)
    print(f"{len(kept)} settings kept:", [(g, r) for g, r, _ in kept])
    print(f"{'candidate':<22}{'x':>7} {'loss':>11} {'retain':>7} | mean | min | max")
    for (name, factor), feat in feats.items():
        mean, low, high = table.get((name, factor), (float("nan"),) * 3)
        loss = f"{feat['mean']:.2f}/{feat['std']:.2f}"
        print(
            f"{name:<22}{factor:>7g} {loss:>11} {feat['retain']:7.3f} | {mean:.3f} | {low:.3f} | {high:.3f}"
        )


if __name__ == "__main__":
    main()
