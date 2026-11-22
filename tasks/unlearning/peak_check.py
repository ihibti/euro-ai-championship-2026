"""Predict sharpened fine-tunes from their scores, without the ledger's single-model golds.

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

Whether the probes' spill differences carry over to the real control set is not known, so each
prediction comes three ways: `probe` (retain as the probes say), `half` (halfway to the anchor's
retain) and `flat` (every line at the anchor's retain). A line is worth a slot only if it beats
the best score under all three. A fourth reading, `temp`, treats the drift on unseen images as
the base's spill temperature (from the spare classes' label probability): after a factor f, the
leftover temperature f / T costs KL on a stand-in for unseen images (the cross-fitted predictions,
softened to a mean top probability of 0.8), and 20% of the probes' KL is kept as drift that no
factor removes.

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
import scoring  # noqa: E402

GOLD_LEVELS = tuple(
    (round(2.6 + 0.2 * i, 2), round(3.0 + 0.3 * j, 2)) for i in range(11) for j in range(6)
)
RETAIN_MODES = ("probe", "half", "flat", "temp")
NONTEMP_SHARE = 0.2
UNSEEN_CONFIDENCE = 0.8
FACTORS = (1.0, 1.15, 1.25, 1.333, 1.5, 1.6, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5)
OBSERVED = (
    ("k8_e3t2_rm1_lr1e4", 2.0, 0.2837),
    ("k8_e3t2_rm1_lr1e4", 2.5, 0.2222),
    ("k7_e3t2_rm", 2.0, 0.2133),
    ("k9_e3t1_lr1e4", 1.25, 0.256),
    ("k8_e3t2_rm1_ep40", 2.25, 0.2665),
)


def parse_observed(items: list[str]) -> tuple:
    """`name:factor:score` strings, kept with the best score first (it becomes the anchor)."""
    rows = []
    for item in items:
        name, factor, score = item.rsplit(":", 2)
        rows.append((name, float(factor), float(score)))
    return tuple(sorted(rows, key=lambda r: -r[2]))


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


def soften_to(logits: torch.Tensor, confidence: float) -> torch.Tensor:
    """Divide logits by the temperature that brings their mean top probability to `confidence`."""
    low, high = 1.0, 50.0
    for _ in range(40):
        mid = (low + high) / 2
        top = float(F.softmax(logits / mid, 1).max(1).values.mean())
        low, high = (mid, high) if top > confidence else (low, mid)
    return logits / ((low + high) / 2)


def residual_kl(unseen: torch.Tensor, ratio: float) -> float:
    return float(scoring.kl_rows(unseen, unseen * ratio).mean())


def temperature_retain(spill: dict, ratio: float, unseen: torch.Tensor) -> float:
    """Retain when the drift is the spill temperature: what the factor leaves, plus a fixed rest."""
    agreement = 0.45 * spill["held_acc"] + 0.55 * spill["spare_acc"]
    rest = NONTEMP_SHARE * (0.45 * spill["held_kl"] + 0.55 * spill["spare_kl"])
    return agreement / (1.0 + rest + residual_kl(unseen, ratio))


def features(
    logits: torch.Tensor,
    factor: float,
    orig: torch.Tensor,
    labels: torch.Tensor,
    spill: dict,
    refs: dict,
    unseen: torch.Tensor | None = None,
) -> dict:
    sharp = logits * factor
    loss = F.cross_entropy(sharp, labels, reduction="none")
    out = {
        "mean": float(loss.mean()),
        "std": float(loss.std()),
        "retain": ledger.sharpened_retain(spill, factor, orig, logits),
        "pred": {name: pred_term(ref, sharp, orig) for name, ref in refs.items()},
    }
    if unseen is not None:
        spill_t = ledger.effective_temperature(orig, labels, spill["spare_p_lab"])
        out["spill_t"] = spill_t
        out["retain_temp"] = temperature_retain(spill, factor / spill_t, unseen)
    return out


def with_retain(feats: dict, mode: str, anchor: tuple) -> dict:
    """Retain as the probes say, halfway to the anchor's, or the anchor's for every line."""
    if mode == "probe":
        return feats
    if mode == "temp":
        return {k: {**f, "retain": f["retain_temp"]} for k, f in feats.items()}
    level = feats[anchor]["retain"]
    share = 0.5 if mode == "half" else 0.0
    return {k: {**f, "retain": level + share * (f["retain"] - level)} for k, f in feats.items()}


def worst_case(tables: dict, modes: list, key: tuple) -> float:
    """The lowest prediction over the readings that kept a setting; -inf when none did."""
    values = [tables[m][key][0] for m in modes if key in tables[m]]
    return min(values) if values else float("-inf")


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
    parser.add_argument("--top", type=int, default=15, help="lines shown, best worst-case first")
    parser.add_argument(
        "--observed",
        nargs="*",
        default=[],
        help="extra sharpened scores as name:factor:score, added to OBSERVED",
    )
    args = parser.parse_args(argv)
    orig = torch.from_numpy(np.load(args.orig)).float()
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        pred = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    step = torch.zeros_like(orig)
    step[:, columns] = pred[:, columns] - orig[:, columns]
    refs = references(orig, step)
    unseen = soften_to(pred, UNSEEN_CONFIDENCE)
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
    for name in sorted(set(dumped) & set(spills)):
        for factor in FACTORS:
            feats[(name, factor)] = features(
                dumped[name], factor, orig, labels, spills[name], refs, unseen
            )
    observed_rows = parse_observed([f"{n}:{f}:{s}" for n, f, s in OBSERVED] + list(args.observed))
    for name, factor, _ in observed_rows:
        if (name, factor) not in feats:
            feats[(name, factor)] = features(
                dumped[name], factor, orig, labels, spills[name], refs, unseen
            )
    anchor = observed_rows[0][:2]
    tables = {}
    for mode in RETAIN_MODES:
        kept, tables[mode] = calibrated(
            with_retain(feats, mode, anchor),
            observed_rows,
            GOLD_LEVELS,
            tuple(refs),
            args.tolerance,
        )
        print(f"{mode}: {len(kept)} settings kept")
    nan = (float("nan"),) * 3
    live = [m for m in RETAIN_MODES if tables[m]]
    worst = {k: worst_case(tables, live, k) for k in feats}
    observed = {(n, f): s for n, f, s in observed_rows}
    shown = sorted(feats, key=lambda k: -worst[k])[: args.top]
    shown += [k for k in observed if k not in shown]
    head = " | ".join(RETAIN_MODES)
    print(f"{'candidate':<24}{'x':>6} {'loss':>11} {'f/T':>5} | {head} | observed")
    for key in shown:
        feat = feats[key]
        loss = f"{feat['mean']:.2f}/{feat['std']:.2f}"
        cells = " | ".join(f"{tables[m].get(key, nan)[0]:.3f}" for m in RETAIN_MODES)
        seen = observed.get(key, "")
        ratio = key[1] / feat["spill_t"]
        print(f"{key[0]:<24}{key[1]:>6g} {loss:>11} {ratio:5.2f} | {cells} | {seen}")


if __name__ == "__main__":
    main()
