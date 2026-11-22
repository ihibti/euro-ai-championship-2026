"""The hypothesis ledger: which golds our scores allow, and what each candidate is worth under them.

Gold is unknown. Two families of golds are scored, both built from the original logits and the
cross-fitted held-out step (`xfit_extras.npz`):
- aligned: gold = (orig + K * step) / tau_g. Gold errs where, and towards what, our cross-fit
  predicts.
- other: the same with the step of another image of the same class. Gold errs as often and
  towards the same classes, on other images.
K sets how often gold is wrong, tau_g how soft it is (also standing for the average of several
gold models). A third family follows the removal reading of `analysis.md`:
- removal: gold never saw the nine classes that lost 130 images, so their logits sit d_g below
  on every image; on the other images gold = orig + K * step; all divided by tau_g.

Candidates:
- a fine-tune is described by its measured loss mean and std (exact images), which give its
  mean_std_score exactly, and by a proxy for its distribution at temperature 2,
  (orig + a * step) / b, fitted to its measured label probability and KL from the original;
  its retain score is estimated from the spill probes (`ledger_inputs.json`);
- `scale_t<t>`: every logit divided by t, exactly; retain from our own logits.
- `dial<d>`: the nine classes' bias lowered by d, exactly (`bias_full30` is dial30); `boats<d>`
  the same on the four small-boat classes 29, 30, 31 and 33 only.

The removal family comes in two variants: `removal`, where the validation and control sets hold
images of the nine classes, so a dial pays at the gate and in retain; and `removal_b`, where they
hold none, so a dial passes the gate and keeps retain near 1 (drift read on the other forget
images).

The utility gate: a candidate that changes a share f of validation predictions passes when A (1 - f)
>= 70%, where A is the original's validation accuracy, and scores 0 otherwise. A is unknown: it gets
a uniform prior over `gate_prior` in the inputs and is inferred jointly with gold, so a 0 can be
explained by the gate (`dial15`, `bias_full30`) and a positive score bounds A from below (`x4_e20`).

Each observed public score weighs every gold by how well it is predicted (log-normal error with
a factor of 1.6, plus 0.003). Printed: the golds that carry the weight, and for every candidate
the weighted mean and the spread of its predicted score.

    python <repo>/tasks/unlearning/ledger.py --targets output/kkrgp/xfit_extras.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scoring
import torch

KS = (0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
GOLD_TAUS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
SCALES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
ERROR_FACTOR = 1.6
FLOOR = 0.003


def within_class_permutation(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.arange(len(labels))
    for label in labels.unique():
        index = torch.nonzero(labels == label).squeeze(1)
        perm[index] = index[torch.randperm(len(index), generator=generator)]
    return perm


def fit_proxy(
    orig: torch.Tensor, step: torch.Tensor, labels: torch.Tensor, label_prob: float, kl: float
) -> tuple[float, float]:
    """(a, b) such that (orig + a * step) / b matches the label probability and KL at T=2."""
    best, best_err = (0.0, 1.0), math.inf
    for a in np.arange(0.0, 3.01, 0.1):
        moved = orig + float(a) * step
        for b in np.arange(1.0, 8.01, 0.1):
            proxy = moved / float(b)
            p = float(scoring.label_prob_t2(proxy, labels).mean())
            k = float(scoring.kl_rows(orig, proxy).mean())
            err = (p - label_prob) ** 2 + ((k - kl) / max(kl, 0.05)) ** 2 * 0.25
            if err < best_err:
                best, best_err = (float(a), float(b)), err
    return best


GATE_THRESHOLD = 70.0
A_POINTS = 61
SAVED_KEYS = ("family", "K", "tau_g", "d_g", "gold_loss", "gold_acc", "pred", "weight", "posterior")
NINE_CLASSES = (0, 6, 7, 8, 29, 30, 31, 33, 38)
BOATS = (29, 30, 31, 33)
DIALS = (10.0, 12.0, 15.0, 20.0, 30.0)
BOAT_DIALS = (10.0, 15.0, 20.0)
REMOVAL_SHIFTS = (10.0, 15.0, 25.0)
REMOVAL_KS = (0.5, 1.0, 2.0, 3.0, 4.0)
REMOVAL_TAUS = (1.0, 1.5, 2.0, 3.0)


def gate_pass(flip: float, low: float, high: float, threshold: float = 70.0) -> float:
    """Share of the band [low, high] of the original's validation accuracy where A (1 - flip)
    stays at or above the threshold."""
    if flip >= 1.0:
        return 0.0
    need = threshold / (1.0 - flip)
    if high <= low:
        return 1.0 if low >= need else 0.0
    return min(max((high - need) / (high - low), 0.0), 1.0)


def unseen_flip(unseen: torch.Tensor, labels: torch.Tensor, classes, shift: float) -> float:
    """Share of the images of `classes` that a model which never saw them classifies correctly
    (the cross-fitted prediction) and that the dial turns wrong."""
    index = torch.tensor(classes)
    inside = torch.isin(labels, index)
    right = (unseen.argmax(1) == labels) & inside
    if not bool(right.any()):
        return 0.0
    dialed = unseen.clone()
    dialed[:, index] -= shift
    return float((dialed.argmax(1) != labels)[right].float().mean())


def dial_candidate(
    orig: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, ...],
    shift: float,
    band,
    unseen: torch.Tensor,
) -> dict:
    """The original with the bias of `classes` lowered by `shift`. With those classes in the
    validation and control sets, the validation images of those classes flip as the cross-fitted
    predictions do (`unseen_flip`) and the drift on them enters retain; without them, retain is
    read on the forget images of the other classes."""
    index = torch.tensor(classes)
    dialed = orig.clone()
    dialed[:, index] -= shift
    inside = torch.isin(labels, index)
    share = len(classes) / 100.0
    flip = share * unseen_flip(unseen, labels, classes, shift)
    kl_inside = float(scoring.kl_rows(orig[inside], dialed[inside]).mean())
    outside = ~inside
    kept = float((dialed.argmax(1) == labels)[outside].float().mean())
    kl_outside = float(scoring.kl_rows(orig[outside], dialed[outside]).mean())
    return {
        "logits": dialed,
        "stats": None,
        "retain": (1.0 - flip) / (1.0 + share * kl_inside),
        "flip": flip,
        "retain_without": kept / (1.0 + kl_outside),
        "flip_without": 0.0,
        "proxy": None,
    }


COMPOSITES = (("boats", BOATS, 20.0), ("dial", NINE_CLASSES, 15.0))


def composite_candidate(
    base: dict,
    base_flip: float,
    orig: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, ...],
    shift: float,
    band,
    step: torch.Tensor,
) -> dict:
    """A fine-tune whose last-layer bias is then lowered by `shift` on `classes`: its dumped
    logits minus the shift on those columns, exactly. Retain multiplies the two factors; the gate
    adds the dial's validation flips to the fine-tune's, except when validation holds none of
    those classes."""
    dial = dial_candidate(orig, labels, classes, shift, band, orig + step)
    logits = base["logits"].clone()
    logits[:, torch.tensor(classes)] -= shift
    dial_flip = len(classes) / 100.0 * unseen_flip(orig + step, labels, classes, shift)
    return {
        "logits": logits,
        "stats": None,
        "retain": base["retain"] * dial["retain"],
        "flip": base_flip + dial_flip,
        "retain_without": base["retain"] * dial["retain_without"],
        "flip_without": base_flip,
        "proxy": "dumped+dial",
    }


SHARPEN = (1.5, 2.0, 2.5, 3.0)


def effective_temperature(orig: torch.Tensor, labels: torch.Tensor, label_prob: float) -> float:
    """The temperature that brings the original's label probability at T=2 down to label_prob."""
    best, best_gap = 1.0, math.inf
    for temperature in np.arange(1.0, 6.01, 0.05):
        gap = abs(float(scoring.label_prob_t2(orig / temperature, labels).mean()) - label_prob)
        if gap < best_gap:
            best, best_gap = float(temperature), gap
    return best


def sharpened_retain(spill: dict, factor: float, orig: torch.Tensor, sub: torch.Tensor) -> float:
    """Retain of a fine-tune whose last layer is multiplied by `factor`.

    The fine-tune's own forget logits `sub` show how the factor moves the KL to the original,
    separately on the images whose predicted class it changed and on the others. The held-out
    probe mixes the two at its measured share of changed predictions; the spare classes move like
    the unchanged images. The factor never changes a predicted class, so the agreements stay.
    """
    changed = sub.argmax(1) != orig.argmax(1)
    kept = ~changed
    before = scoring.kl_rows(orig, sub)
    after = scoring.kl_rows(orig, sub * factor)
    kept_before = max(float(before[kept].mean()), 1e-12) if kept.any() else 1e-12
    kept_after = float(after[kept].mean()) if kept.any() else 0.0
    if changed.any():
        changed_before, changed_after = float(before[changed].mean()), float(after[changed].mean())
    else:
        changed_before, changed_after = kept_before, kept_after
    share = 1.0 - spill["held_acc"]
    mix_before = share * changed_before + (1.0 - share) * kept_before
    mix_after = share * changed_after + (1.0 - share) * kept_after
    held = spill["held_kl"] * mix_after / mix_before
    spare = spill["spare_kl"] * kept_after / kept_before
    kl = 0.45 * held + 0.55 * spare
    agreement = 0.45 * spill["held_acc"] + 0.55 * spill["spare_acc"]
    return agreement / (1.0 + kl)


def weight(predicted: float, observed: float) -> float:
    gap = math.log(predicted + FLOOR) - math.log(observed + FLOOR)
    return math.exp(-0.5 * (gap / math.log(ERROR_FACTOR)) ** 2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--orig", type=Path, default=Path("output/kkrgp/orig_exact.npy"))
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("ledger_inputs.json")
    )
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp/ledger.json"))
    parser.add_argument(
        "--logits",
        type=Path,
        nargs="+",
        default=[],
        help="dump_logits.py outputs: exact logits win, later files over earlier ones",
    )
    args = parser.parse_args(argv)
    orig = torch.from_numpy(np.load(args.orig)).float()
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        proxy = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    step = torch.zeros_like(orig)
    step[:, columns] = proxy[:, columns] - orig[:, columns]
    other = step[within_class_permutation(labels)]
    inputs = json.loads(args.inputs.read_text())

    candidates: dict[str, dict] = {}
    for name, spec in inputs["candidates"].items():
        real = spec["realized"]
        a, b = fit_proxy(orig, step, labels, real["label_prob_t2_sub"], real["kl_orig_sub_t2"])
        candidates[name] = {
            "logits": (orig + a * step) / b,
            "stats": (real["loss_mean"], real["loss_std"]),
            "retain": spec["retain"],
            "proxy": (round(a, 2), round(b, 2)),
        }
    for path in args.logits:
        with np.load(path) as dumped:
            for name in candidates:
                key = f"{name}__exact"
                if key in dumped.files:
                    candidates[name]["logits"] = torch.from_numpy(dumped[key].astype(np.float32))
                    candidates[name]["stats"] = None
                    candidates[name]["proxy"] = "dumped"
    band = inputs.get("gate_prior", inputs.get("gate_band"))
    for name, spec in inputs["candidates"].items():
        candidates[name]["flip"] = spec.get("flip", 0.0)
    for t in SCALES:
        candidates[f"scale_t{t:g}"] = {
            "logits": orig / t,
            "stats": None,
            "retain": scoring.retain_score(orig / t, orig, labels),
            "proxy": None,
            "flip": 0.0,
        }
    nine = torch.tensor(NINE_CLASSES)
    in_nine = torch.isin(labels, nine)
    for d in DIALS:
        candidates[f"dial{d:g}"] = dial_candidate(orig, labels, NINE_CLASSES, d, band, orig + step)
    for d in BOAT_DIALS:
        candidates[f"boats{d:g}"] = dial_candidate(orig, labels, BOATS, d, band, orig + step)
    for name, spec in inputs["candidates"].items():
        if "spill" not in spec:
            continue
        for factor in SHARPEN:
            candidates[f"{name}x{factor:g}"] = {
                "logits": candidates[name]["logits"] * factor,
                "stats": None,
                "retain": sharpened_retain(spec["spill"], factor, orig, candidates[name]["logits"]),
                "flip": spec.get("flip", 0.0),
                "proxy": "dumped x" if candidates[name]["proxy"] == "dumped" else "proxy x",
            }
    for name, spec in inputs["candidates"].items():
        if candidates[name]["proxy"] != "dumped":
            continue
        for label, classes, d in COMPOSITES:
            candidates[f"{name}+{label}{d:g}"] = composite_candidate(
                candidates[name], spec.get("flip", 0.0), orig, labels, classes, d, band, step
            )

    hypotheses = [
        (family, k, tau, 0.0, (orig + k * direction) / tau, False)
        for family, direction in (("aligned", step), ("other", other))
        for k in KS
        for tau in GOLD_TAUS
    ]
    outside = (~in_nine).float().view(-1, 1)
    for shift in REMOVAL_SHIFTS:
        for k in REMOVAL_KS:
            for tau in REMOVAL_TAUS:
                gold = orig + k * step * outside
                gold[:, nine] -= shift
                for without in (False, True):
                    family = "removal_b" if without else "removal"
                    hypotheses.append((family, k, tau, shift, gold / tau, without))
    grid = np.linspace(band[0], band[1], A_POINTS)
    golds = []
    for family, k, tau, shift, gold, without in hypotheses:
        row = {"family": family, "K": k, "tau_g": tau, "d_g": shift}
        mean, std = scoring.loss_stats(gold, labels)
        row["gold_loss"] = (round(mean, 3), round(std, 3))
        row["gold_acc"] = round(float((gold.argmax(1) == labels).float().mean()), 3)
        ungated, passes = {}, {}
        for name, cand in candidates.items():
            terms = scoring.forget_terms(cand["logits"], gold, orig, labels, cand["stats"])
            forget = scoring.forget_score([terms])
            retain = cand.get("retain_without", cand["retain"]) if without else cand["retain"]
            flip = cand.get("flip_without", cand["flip"]) if without else cand["flip"]
            ungated[name] = forget * retain
            passes[name] = grid * (1.0 - flip) >= GATE_THRESHOLD
        weights_a = np.ones(len(grid))
        for name, observed in inputs["observed"].items():
            if name in ungated:
                hit, miss = weight(ungated[name], observed), weight(0.0, observed)
                weights_a = weights_a * np.where(passes[name], hit, miss)
        row["pred"] = {name: ungated[name] * float(passes[name].mean()) for name in candidates}
        row["ungated"], row["passes"], row["weight_a"] = ungated, passes, weights_a
        row["weight"] = float(weights_a.mean())
        golds.append(row)

    total = sum(float(row["weight_a"].sum()) for row in golds)
    for row in golds:
        row["joint"] = row["weight_a"] / total if total > 0 else np.zeros(len(grid))
        row["posterior"] = float(row["joint"].sum())
    a_posterior = sum(row["joint"] for row in golds)
    ranked = sorted(golds, key=lambda row: -row["posterior"])
    print(
        "proxies (a, b) for the fine-tunes:",
        {n: c["proxy"] for n, c in candidates.items() if c["proxy"] is not None},
    )
    print(
        "\ngolds carrying the weight (posterior, family, K, tau_g, gold loss mean/std, gold acc):"
    )
    for row in ranked[:12]:
        observed = " ".join(
            f"{n}={row['pred'][n]:.4f}" for n in inputs["observed"] if n in row["pred"]
        )
        print(
            f"  {row['posterior']:.3f} {row['family']:<7} K={row['K']:<4} tau_g={row['tau_g']:<3} "
            f"d_g={row['d_g']:<4g} "
            f"loss {row['gold_loss'][0]:.2f}/{row['gold_loss'][1]:.2f} "
            f"acc {row['gold_acc']:.2f} | {observed}"
        )
    shares: dict[str, float] = {}
    for row in golds:
        shares[row["family"]] = shares.get(row["family"], 0.0) + row["posterior"]
    print("\nposterior by family:", {f: round(v, 3) for f, v in shares.items()})
    a_mean = float((grid * a_posterior).sum())
    a_cumulative = np.cumsum(a_posterior)
    a_low = float(grid[np.searchsorted(a_cumulative, 0.1)])
    a_high = float(grid[min(np.searchsorted(a_cumulative, 0.9), len(grid) - 1)])
    print(f"original's validation accuracy A: mean {a_mean:.1f}, 10th to 90th", end=" ")
    print(f"{a_low:.1f} to {a_high:.1f}")
    print("\ncandidate: weighted mean | 10th to 90th percentile | P(score above the best so far)")
    best_so_far = max(inputs["observed"].values())
    summary = {}
    for name in candidates:
        values = np.concatenate(
            [np.where(row["passes"][name], row["ungated"][name], 0.0) for row in golds]
        )
        weights = np.concatenate([row["joint"] for row in golds])
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        low = float(values[order][np.searchsorted(cumulative, 0.1)])
        high = float(values[order][min(np.searchsorted(cumulative, 0.9), len(values) - 1)])
        mean = float((values * weights).sum())
        above = float(weights[values > best_so_far].sum())
        summary[name] = {"mean": mean, "p10": low, "p90": high, "p_above_best": above}
        by_family = {}
        for family in shares:
            mask = np.repeat(np.array([row["family"] == family for row in golds]), len(grid))
            total_f = weights[mask].sum()
            weighted = float((values[mask] * weights[mask]).sum())
            by_family[family] = weighted / total_f if total_f else 0.0
        summary[name]["by_family"] = by_family
        cells = " ".join(f"{f}={v:.3f}" for f, v in by_family.items())
        print(f"  {name:<12} {mean:.4f} | {low:.4f} to {high:.4f} | {above:.2f} | {cells}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "observed": inputs["observed"],
                "a_posterior": {"grid": grid.round(2).tolist(), "mass": a_posterior.tolist()},
                "summary": summary,
                "golds": [{k: v for k, v in row.items() if k in SAVED_KEYS} for row in ranked],
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
