"""Fit the scorer's unknowns to the public scores of every line whose logits we hold; rank lines.

The ledger ties gold's loss statistics to its shape: gold = (orig + K step) / tau_g sets both the
(mean, std) that the mean/std term compares with and the distribution of the prediction term. It
also skips every observed line whose factor is not in its sharpening grid (x1.25, x2.25) and every
weight average. Here:
- every scored line is rebuilt exactly from dumped logits: a member at a factor, or a weight
  average through the same average of its members' logits, times a factor;
- the prediction term is computed under each gold shape of the ledger (aligned, other, removal);
- the mean/std term compares with a free (mean, std) of gold's loss, on a grid;
- retain is the probes' estimate (`ledger.sharpened_retain`), times one free level c that also
  absorbs the level of the prediction term.
Both groups are scored by default, as on the server: the exact images, and the dumped jitter view
standing for the altered copies, with gold's shapes built on that view's original logits and the
same gold (mean, std) on both (`--groups exact` reproduces the ledger's single group). Each
hypothesis (shape, mean, std, c) is weighed
by a log-normal error of `--sigma` on score + 0.003, under a uniform prior on the grid.

Printed: the fit of the best hypothesis, the posterior of gold's loss statistics, a leave-one-out
check, and for every line the best factor with its predicted score, its 10th to 90th percentile,
and the chance it beats the best scored line under the same hypothesis.

    cd <team folder> && python <repo>/tasks/unlearning/score_fit.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import scoring
import torch
from ledger import (
    GOLD_TAUS,
    KS,
    NINE_CLASSES,
    REMOVAL_KS,
    REMOVAL_SHIFTS,
    REMOVAL_TAUS,
    sharpened_retain,
    within_class_permutation,
)

TEMPERATURE = 2.0
FLOOR = 0.003
MEANS = np.round(np.arange(0.5, 6.001, 0.125), 3)
STDS = np.round(np.arange(0.5, 6.001, 0.125), 3)
LEVELS = np.exp(np.linspace(np.log(0.4), np.log(2.5), 37))
SOUP = (("k8_e3t2_rm1_lr1e4", 1.0), ("k11_b_s1", 1.0), ("k11_b_s2", 1.0))
TAU1 = (("k9_e3t1_lr1e4", 0.75), ("k8_e3t1_rm1", 0.25))
BLENDS = {
    "k11_b_soup3": SOUP,
    "m14_t1_blend3": TAU1,
    "m14_t15_blend": (("k8_e3t15_rm1", 1.0), ("k10_t15_lr2e4_ep40", 1.0)),
    **{
        f"soup_t1_{round(100 * share)}": tuple((n, w * (1 - share) / 3) for n, w in SOUP)
        + tuple((n, w * share) for n, w in TAU1)
        for share in (1 / 3, 1 / 2, 2 / 3)
    },
}
SWEEP = np.round(np.arange(1.0, 3.001, 0.05), 3)
NAME = re.compile(r"^(.*)x(\d+(?:\.\d+)?)$")


def parse_line(name: str, known: set[str]) -> tuple[str, float] | None:
    """(member or blend, factor) of an observed name such as `k8_e3t2_rm1_lr1e4x2.5`."""
    match = NAME.match(name)
    if match and match.group(1) in known:
        return match.group(1), float(match.group(2))
    if name in known:
        return name, 1.0
    return None


def members(base: str) -> tuple[tuple[str, float], ...]:
    return BLENDS.get(base, ((base, 1.0),))


def blend_logits(parts, store: dict[str, np.ndarray]) -> np.ndarray:
    total = sum(w for _, w in parts)
    return sum(w * store[n] for n, w in parts) / total


def blend_spec(parts, specs: dict[str, dict]) -> tuple[dict | None, float, float | None]:
    """Weight average of the members' probe spill and changed predictions; without probes, the
    single member's own retain estimate."""
    total = sum(w for _, w in parts)
    flip = sum(w * specs[n].get("flip", 0.0) for n, w in parts) / total
    if not all("spill" in specs[n] for n, _ in parts):
        retain = specs[parts[0][0]].get("retain") if len(parts) == 1 else None
        return None, flip, retain
    keys = specs[parts[0][0]]["spill"].keys()
    spill = {k: sum(w * specs[n]["spill"][k] for n, w in parts) / total for k in keys}
    return spill, flip, None


def log_probs(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) / TEMPERATURE
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def gold_table(golds, orig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every gold's probabilities at T=2 (S x N*C), its mean negative entropy, and its mean KL to
    the original."""
    orig_lp = log_probs(orig).reshape(-1)
    n = orig.shape[0]
    probs, neg, kl_orig = [], [], []
    for gold in golds:
        lp = log_probs(gold).reshape(-1)
        p = np.exp(lp)
        h = float(p @ lp) / n
        probs.append(p.astype(np.float32))
        neg.append(h)
        kl_orig.append(h - float(p @ orig_lp) / n)
    return np.stack(probs), np.array(neg), np.array(kl_orig)


def prediction_column(table, sub: np.ndarray) -> np.ndarray:
    """prediction_score of one line under every gold of `table`."""
    probs, neg, kl_orig = table
    cross = probs @ log_probs(sub).reshape(-1).astype(np.float32) / sub.shape[0]
    kl_sub = neg - cross.astype(np.float64)
    safe = np.where(kl_orig > 0, kl_orig, 1.0)
    return np.where(kl_orig > 0, 1.0 - np.clip(kl_sub / safe, 0.0, 1.0), 0.0)


def ms_table(stats: np.ndarray, orig_stats, means=MEANS, stds=STDS) -> np.ndarray:
    """mean_std_score of every line (rows of `stats`: loss mean, std) against every gold (mean,
    std) on the grid, as an L x M x S array."""
    gm, gs = means[None, :, None], stds[None, None, :]
    d_sub = np.abs(stats[:, 0, None, None] - gm) + np.abs(stats[:, 1, None, None] - gs)
    d_orig = np.abs(orig_stats[0] - gm) + np.abs(orig_stats[1] - gs)
    return 1.0 - np.clip(d_sub / d_orig, 0.0, 1.0)


def log_likelihood(pred_obs, ms_obs, retain_obs, observed, sigma: float, levels=LEVELS):
    """Log-likelihood of every hypothesis (shape, mean, std, level), an S x M x S' x C array."""
    target = np.log(np.asarray(observed) + FLOOR)[:, None, None, None]
    out = np.empty((pred_obs.shape[0], ms_obs.shape[1], ms_obs.shape[2], len(levels)))
    for s in range(pred_obs.shape[0]):
        base = ms_obs * (pred_obs[s] * retain_obs)[:, None, None]
        hat = base[..., None] * levels
        gap = (np.log(hat + FLOOR) - target) / sigma
        out[s] = -0.5 * (gap**2).sum(0)
    return out


def normalise(loglik: np.ndarray) -> np.ndarray:
    w = np.exp(loglik - loglik.max())
    return w / w.sum()


def draw(weights: np.ndarray, count: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    flat = rng.choice(weights.size, size=count, p=weights.reshape(-1))
    return np.unravel_index(flat, weights.shape)


def predicted(index, pred, ms, retain, levels=LEVELS) -> np.ndarray:
    """Scores of every line under each drawn hypothesis, a draws x L array; `pred` is S x L."""
    s, m, d, c = index
    return levels[c][:, None] * ms[:, m, d].T * pred[s] * retain[None, :]


def gold_shapes(orig: torch.Tensor, step: torch.Tensor, labels: torch.Tensor):
    """The ledger's gold shapes, as (label, logits): aligned, other and removal."""
    other = step[within_class_permutation(labels)]
    nine = torch.tensor(NINE_CLASSES)
    outside = (~torch.isin(labels, nine)).float().view(-1, 1)
    for family, direction in (("aligned", step), ("other", other)):
        for k in KS:
            for tau in GOLD_TAUS:
                yield f"{family} K={k:g} tau={tau:g}", ((orig + k * direction) / tau).numpy()
    for shift in REMOVAL_SHIFTS:
        for k in REMOVAL_KS:
            for tau in REMOVAL_TAUS:
                gold = orig + k * step * outside
                gold[:, nine] -= shift
                yield f"removal d={shift:g} K={k:g} tau={tau:g}", (gold / tau).numpy()


def load_logits(root: Path, view: str = "exact") -> dict[str, np.ndarray]:
    store: dict[str, np.ndarray] = {}
    suffix = f"__{view}"
    for path in sorted(root.glob("*/logits.npz")):
        with np.load(path) as data:
            for key in data.files:
                if key.endswith(suffix):
                    store[key[: -len(suffix)]] = data[key].astype(np.float32)
    return store


def unit_logits(base: str, store: dict[str, np.ndarray]) -> np.ndarray:
    """Logits at factor 1 of a member, a blend or `scale_t<t>`."""
    if base.startswith("scale_t"):
        return store["orig"] / float(base[len("scale_t") :])
    return blend_logits(members(base), store)


def retain_function(base: str, store, specs, orig_t: torch.Tensor):
    """Retain as a function of the factor, and the share of changed predictions."""
    unit = unit_logits(base, store)
    if base.startswith("scale_t"):
        r = scoring.retain_score(torch.from_numpy(unit), orig_t, orig_t.argmax(1))
        return (lambda f: r if f == 1.0 else float("nan")), 0.0
    spill, flip, fixed = blend_spec(members(base), specs)
    if spill is not None:
        unit_t = torch.from_numpy(unit)
        return (lambda f: sharpened_retain(spill, f, orig_t, unit_t)), flip
    if fixed is not None:
        return (lambda f: fixed if f == 1.0 else float("nan")), flip
    return (lambda f: float("nan")), flip


def view_terms(entries, store, step: torch.Tensor, labels_t: torch.Tensor):
    """Loss statistics (L x 2) and prediction terms (S x L) of every line on one group of images,
    with the gold shapes built on that group's original logits."""
    orig_v = store["orig"]
    shape_labels, golds = zip(*gold_shapes(torch.from_numpy(orig_v), step, labels_t), strict=True)
    table = gold_table(golds, orig_v)
    del golds
    units: dict[str, np.ndarray] = {}
    stats, columns = [], []
    for _, base, factor in entries:
        if base not in units:
            units[base] = unit_logits(base, store)
        z = units[base] * factor
        stats.append(scoring.loss_stats(torch.from_numpy(z), labels_t))
        columns.append(prediction_column(table, z))
    orig_stats = scoring.loss_stats(torch.from_numpy(orig_v), labels_t)
    return shape_labels, np.array(stats), np.stack(columns, axis=1), orig_stats


def summary(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.mean()), float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))


def quantiles(grid: np.ndarray, marginal: np.ndarray) -> tuple[float, float, float]:
    cdf = np.cumsum(marginal)
    top = len(grid) - 1
    return (
        float((grid * marginal).sum()),
        float(grid[min(np.searchsorted(cdf, 0.1), top)]),
        float(grid[min(np.searchsorted(cdf, 0.9), top)]),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, default=Path("output/kkrgp/xfit_extras.npz"))
    parser.add_argument("--logits", type=Path, default=Path("/Volumes/euroai-apfs/r21"))
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("ledger_inputs.json")
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--groups", choices=("exact", "both"), default="both")
    parser.add_argument(
        "--level", type=float, default=None, help="fix the level c instead of fitting it"
    )
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp/score_fit.json"))
    args = parser.parse_args(argv)

    store = load_logits(args.logits)
    inputs = json.loads(args.inputs.read_text())
    specs = inputs["candidates"]
    orig = store["orig"]
    orig_t = torch.from_numpy(orig)
    labels_t = orig_t.argmax(1)
    with np.load(args.targets) as data:
        proxy = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    step = torch.zeros_like(orig_t)
    step[:, columns] = proxy[:, columns] - orig_t[:, columns]

    blends = {b for b, parts in BLENDS.items() if all(n in store and n in specs for n, _ in parts)}
    members_known = {n for n in specs if n in store}
    scales = {f"scale_t{t}" for t in (2, 3, 4, 5, 6, 8)}
    known = members_known | blends | scales
    observed = []
    for name, score in inputs["observed"].items():
        parsed = parse_line(name, known)
        if parsed is not None:
            observed.append((name, parsed[0], parsed[1], score))
    sweep_bases = sorted(blends | {n for n in members_known if "spill" in specs[n]})
    entries = [(n, b, f) for n, b, f, _ in observed]
    entries += [(f"{b}x{f:g}", b, float(f)) for b in sweep_bases for f in SWEEP]

    functions = {}
    retain, flips = [], []
    for _, base, factor in entries:
        if base not in functions:
            functions[base] = retain_function(base, store, specs, orig_t)
        retain_of, flip = functions[base]
        retain.append(retain_of(factor))
        flips.append(flip)
    retain, flips = np.array(retain), np.array(flips)
    views = ("exact", "jitter") if args.groups == "both" else ("exact",)
    ms_parts, pred_parts, stats_by_view = [], [], {}
    for view in views:
        store_v = store if view == "exact" else load_logits(args.logits, view)
        labels_shapes, stats_v, pred_v, orig_stats_v = view_terms(entries, store_v, step, labels_t)
        ms_parts.append(ms_table(stats_v, orig_stats_v))
        pred_parts.append(pred_v)
        stats_by_view[view] = stats_v
    ms = sum(ms_parts) / len(ms_parts)
    pred = sum(pred_parts) / len(pred_parts)
    stats = stats_by_view["exact"]

    n_obs = len(observed)
    scores = np.array([o[3] for o in observed])
    obs_index = np.arange(n_obs)
    levels = LEVELS if args.level is None else np.array([args.level])
    loglik = log_likelihood(pred[:, :n_obs], ms[:n_obs], retain[:n_obs], scores, args.sigma, levels)
    weights = normalise(loglik)
    print(f"largest log-likelihood {loglik.max():.2f}")
    s0, m0, d0, c0 = np.unravel_index(int(np.argmax(weights)), weights.shape)
    print(
        f"observed lines used: {n_obs}; best hypothesis: {labels_shapes[s0]}, gold loss "
        f"{MEANS[m0]:.3f}/{STDS[d0]:.3f}, level {levels[c0]:.2f}"
    )
    for i, (name, _, _, score) in enumerate(observed):
        hat = levels[c0] * ms[i, m0, d0] * pred[s0, i] * retain[i]
        print(
            f"  {name:28s} observed {score:.4f} fit {hat:.4f} | loss {stats[i, 0]:.2f}/"
            f"{stats[i, 1]:.2f} ms {ms[i, m0, d0]:.3f} pred {pred[s0, i]:.3f} "
            f"retain {retain[i]:.3f}"
        )
    posterior = {}
    for label, grid, axes in (
        ("gold loss mean", MEANS, (0, 2, 3)),
        ("gold loss std", STDS, (0, 1, 3)),
        ("level c", levels, (0, 1, 2)),
    ):
        mean, low, high = quantiles(grid, weights.sum(axes))
        posterior[label] = (mean, low, high)
        print(f"{label}: mean {mean:.2f}, 10th to 90th {low:.2f} to {high:.2f}")
    families: dict[str, float] = {}
    for s, label in enumerate(labels_shapes):
        families[label.split()[0]] = families.get(label.split()[0], 0.0) + float(weights[s].sum())
    print("posterior by gold shape family:", {k: round(v, 3) for k, v in families.items()})
    top = np.argsort(-weights.sum((1, 2, 3)))[:6]
    print(
        "shapes carrying the weight:",
        [(labels_shapes[s], round(float(weights[s].sum()), 3)) for s in top],
    )

    print("leave one out (observed, predicted mean, 10th to 90th):")
    loo = {}
    for i in obs_index:
        keep = np.flatnonzero(obs_index != i)
        w = normalise(
            log_likelihood(pred[:, keep], ms[keep], retain[keep], scores[keep], args.sigma, levels)
        )
        values = predicted(
            draw(w, args.draws, seed=int(i)), pred[:, [i]], ms[[i]], retain[[i]], levels
        )
        mean, low, high = summary(values[:, 0])
        loo[observed[i][0]] = {"observed": float(scores[i]), "mean": mean, "p10": low, "p90": high}
        print(f"  {observed[i][0]:28s} {scores[i]:.4f} {mean:.4f} ({low:.4f} to {high:.4f})")

    values = predicted(draw(weights, args.draws), pred, ms, retain, levels)
    best_obs = int(np.argmax(scores))
    reference = values[:, best_obs]
    lines = {}
    print(
        f"\nbest factor per line: mean (10th to 90th) | P(beats {observed[best_obs][0]} under "
        "the same hypothesis) | loss and changed predictions at that factor"
    )
    for base in sweep_bases:
        idx = [j for j, e in enumerate(entries) if e[1] == base and j >= n_obs]
        j = idx[int(np.argmax(values[:, idx].mean(0)))]
        mean, low, high = summary(values[:, j])
        beats = float((values[:, j] > reference).mean())
        lines[base] = {
            "factor": entries[j][2],
            "mean": mean,
            "p10": low,
            "p90": high,
            "p_beats_best": beats,
            "loss": stats[j].tolist(),
            "curve": {f"{entries[k][2]:g}": float(values[:, k].mean()) for k in idx},
        }
        print(
            f"  {base:24s} x{entries[j][2]:<5g} {mean:.4f} ({low:.4f} to {high:.4f}) | "
            f"P {beats:.2f} | loss {stats[j, 0]:.2f}/{stats[j, 1]:.2f} flip {flips[j]:.3f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sigma": args.sigma,
        "groups": args.groups,
        "observed": [o[0] for o in observed],
        "best": {
            "shape": labels_shapes[s0],
            "mean": float(MEANS[m0]),
            "std": float(STDS[d0]),
            "level": float(levels[c0]),
        },
        "posterior": posterior,
        "families": families,
        "loo": loo,
        "lines": lines,
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
