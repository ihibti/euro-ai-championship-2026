"""The hypothesis ledger: which golds our scores allow, and what each candidate is worth under them.

Gold is unknown. Two families of golds are scored, both built from the original logits and the
cross-fitted held-out step (`xfit_extras.npz`):
- aligned: gold = (orig + K * step) / tau_g. Gold errs where, and towards what, our cross-fit
  predicts.
- other: the same with the step of another image of the same class. Gold errs as often and
  towards the same classes, on other images.
K sets how often gold is wrong, tau_g how soft it is (also standing for the average of several
gold models).

Candidates:
- a fine-tune is described by its measured loss mean and std (exact images), which give its
  mean_std_score exactly, and by a proxy for its distribution at temperature 2,
  (orig + a * step) / b, fitted to its measured label probability and KL from the original;
  its retain score is estimated from the spill probes (`ledger_inputs.json`);
- `scale_t<t>`: every logit divided by t, exactly; retain from our own logits.

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

KS = (0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0)
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
    for t in SCALES:
        candidates[f"scale_t{t:g}"] = {
            "logits": orig / t,
            "stats": None,
            "retain": scoring.retain_score(orig / t, orig, labels),
            "proxy": None,
        }

    golds = []
    for family, direction in (("aligned", step), ("other", other)):
        for k in KS:
            for tau in GOLD_TAUS:
                gold = (orig + k * direction) / tau
                row = {"family": family, "K": k, "tau_g": tau}
                mean, std = scoring.loss_stats(gold, labels)
                row["gold_loss"] = (round(mean, 3), round(std, 3))
                row["gold_acc"] = round(float((gold.argmax(1) == labels).float().mean()), 3)
                row["pred"] = {}
                for name, cand in candidates.items():
                    terms = scoring.forget_terms(cand["logits"], gold, orig, labels, cand["stats"])
                    row["pred"][name] = scoring.forget_score([terms]) * cand["retain"]
                w = 1.0
                for name, observed in inputs["observed"].items():
                    if name in row["pred"]:
                        w *= weight(row["pred"][name], observed)
                row["weight"] = w
                golds.append(row)

    total = sum(row["weight"] for row in golds)
    for row in golds:
        row["posterior"] = row["weight"] / total if total > 0 else 0.0
    ranked = sorted(golds, key=lambda row: -row["posterior"])
    print(
        "proxies (a, b) for the fine-tunes:",
        {n: c["proxy"] for n, c in candidates.items() if c["proxy"]},
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
            f"loss {row['gold_loss'][0]:.2f}/{row['gold_loss'][1]:.2f} "
            f"acc {row['gold_acc']:.2f} | {observed}"
        )
    print(
        "\ncandidate: weighted mean | 10th to 90th percentile | P(score above the best so far)"
    )
    best_so_far = max(inputs["observed"].values())
    summary = {}
    for name in candidates:
        values = np.array([row["pred"][name] for row in golds])
        weights = np.array([row["posterior"] for row in golds])
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        low = float(values[order][np.searchsorted(cumulative, 0.1)])
        high = float(values[order][min(np.searchsorted(cumulative, 0.9), len(values) - 1)])
        mean = float((values * weights).sum())
        above = float(weights[values > best_so_far].sum())
        summary[name] = {"mean": mean, "p10": low, "p90": high, "p_above_best": above}
        print(f"  {name:<12} {mean:.4f} | {low:.4f} to {high:.4f} | {above:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "observed": inputs["observed"],
                "summary": summary,
                "golds": [{k: v for k, v in row.items()} for row in ranked],
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
