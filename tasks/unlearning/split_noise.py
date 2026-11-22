"""How far apart can the public scores of two models fall from the public split alone?

The board scores a random 30% of the forget images (an altered copy follows its source); the final
ranking uses the other 70%. This draws many such splits and computes, for pairs of submitted lines,
the ratio of their forget terms on the 30% and on the matching 70%, against a stand-in for gold: the
cross-fitted prediction, extrapolated 3 times from the original (`K3t1`). A mixture of the steps 1
to 4 at temperature 2 was tried as a second stand-in and dropped: it gives the sharpened lines
forget terms near 0 and puts the 0.2133 at 7.6 times the 0.2837.

The forget term follows the metric on each group (exact images, and jittered copies standing in for
the hidden altered ones): mean/spread term at temperature 1 on the labels, times the prediction
term from KL at temperature 2, averaged over the two groups. Retain does not depend on the split,
so a ratio of forget terms is a ratio of final scores.

The labels are the original's predictions on the exact images: it fits every forget image with a
mean top probability of 0.9997, so they are its training labels. The `labels` stored in
`xfit_extras.npz` are the cross-fit's own predicted classes (they equal the argmax of `pred_exact`
on every image and differ from the original's on 5.2%), so they are not used here.

Read per pair:
- `full`: the ratio on all images, the part of the gap that is the models' own;
- `pub sd`: the spread of the ratio over 30% splits, the part the board adds;
- `flip`: how often the 30% and the 70% disagree on which line is better.

    cd <team folder> && python <repo>/tasks/unlearning/split_noise.py [--draws 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LOGITS = Path("/Volumes/euroai-apfs/r21")
LINES = {
    "B 0.2837": ("eighth", "k8_e3t2_rm1_lr1e4", 2.0),
    "s1 x2": ("eleventh", "k11_b_s1", 2.0),
    "s2 x2": ("eleventh", "k11_b_s2", 2.0),
    "C 0.2222": ("eighth", "k8_e3t2_rm1_lr1e4", 2.5),
    "A 0.2133": ("eighth", "k7_e3t2_rm", 2.0),
    "D 0.256": ("ninth", "k9_e3t1_lr1e4", 1.25),
    "E 0.2665": ("eighth", "k8_e3t2_rm1_ep40", 2.25),
}
PAIRS = (
    ("s1 x2", "B 0.2837"),
    ("s2 x2", "B 0.2837"),
    ("s2 x2", "s1 x2"),
    ("soup x2", "B 0.2837"),
    ("E 0.2665", "B 0.2837"),
    ("D 0.256", "B 0.2837"),
    ("C 0.2222", "B 0.2837"),
    ("A 0.2133", "B 0.2837"),
)
PUBLIC_SHARE = 0.3


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def per_image(logits: np.ndarray, gold: np.ndarray, labels: np.ndarray) -> dict:
    """Per-image true-label loss at temperature 1 and KL(gold || model) at temperature 2."""
    loss = -log_softmax(logits)[np.arange(len(labels)), labels]
    lg, lm = log_softmax(gold / 2), log_softmax(logits / 2)
    return {"loss": loss, "kl": (np.exp(lg) * (lg - lm)).sum(1)}


def subset_stats(masks: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation of x over each row's selected images."""
    n = masks.sum(1)
    mean = masks @ x / n
    var = (masks @ (x * x) - n * mean**2) / (n - 1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def forget_terms(masks: np.ndarray, model: dict, gold: dict, orig: dict) -> np.ndarray:
    """The metric's forget term on each row's images, for one group."""
    gm, gs = subset_stats(masks, gold["loss"])
    om, os_ = subset_stats(masks, orig["loss"])
    mm, ms = subset_stats(masks, model["loss"])
    d_orig = np.abs(om - gm) + np.abs(os_ - gs)
    ms_term = np.clip(1 - (np.abs(mm - gm) + np.abs(ms - gs)) / d_orig, 0, None)
    pred = np.clip(1 - (masks @ model["kl"]) / (masks @ orig["kl"]), 0, None)
    return ms_term * pred


def split_masks(n: int, draws: int, share: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((draws, n)) < share).astype(np.float64)


def pair_table(forget: dict, forget_rest: dict, full: dict, pairs: tuple) -> list[dict]:
    """Ratio of each pair on all images, its spread over 30% splits and over the matching 70%."""
    rows = []
    for a, b in pairs:
        if full[b] <= 0 or full[a] <= 0:
            continue
        pub = forget[a] / forget[b]
        rest = forget_rest[a] / forget_rest[b]
        rows.append(
            {
                "pair": f"{a} / {b}",
                "full": full[a] / full[b],
                "pub_mean": float(pub.mean()),
                "pub_sd": float(pub.std()),
                "rest_sd": float(rest.std()),
                "flip": float(np.mean((pub > 1) != (rest > 1))),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, default=Path("output/kkrgp/xfit_extras.npz"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    with np.load(args.targets) as data:
        pred = data["pred_exact"].astype(np.float64)
    cache = {}
    for batch in {b for b, _, _ in LINES.values()}:
        with np.load(LOGITS / batch / "logits.npz") as data:
            cache[batch] = {k: data[k].astype(np.float64) for k in data.files}
    views = {}
    for view in ("exact", "jitter"):
        orig = cache["eleventh"][f"orig__{view}"]
        models = {k: cache[b][f"{n}__{view}"] * f for k, (b, n, f) in LINES.items()}
        twins = [LINES[k] for k in ("B 0.2837", "s1 x2", "s2 x2")]
        models["soup x2"] = 2.0 * np.mean([cache[b][f"{n}__{view}"] for b, n, _ in twins], 0)
        views[view] = (orig, models)
    labels = cache["eleventh"]["orig__exact"].argmax(1)
    rng = np.random.default_rng(args.seed)
    masks = split_masks(len(labels), args.draws, PUBLIC_SHARE, rng)
    everything = np.ones((1, len(labels)))
    step = pred - cache["eleventh"]["orig__exact"]
    for gold_name in ("K3t1",):
        terms = {"pub": {}, "rest": {}, "full": {}}
        for orig, models in views.values():
            gold = orig + 3 * step
            g = per_image(gold, gold, labels)
            o = per_image(orig, gold, labels)
            for name, logits in models.items():
                m = per_image(logits, gold, labels)
                for part, mk in (("pub", masks), ("rest", 1 - masks), ("full", everything)):
                    value = forget_terms(mk, m, g, o) / 2
                    terms[part][name] = terms[part].get(name, 0) + value
        full = {k: float(v[0]) for k, v in terms["full"].items()}
        print(f"\ngold stand-in {gold_name}; forget term on all images:")
        print("  " + ", ".join(f"{k} {v:.3f}" for k, v in full.items()))
        live = [k for k, v in full.items() if v > 0]
        spread = ", ".join(f"{k} {float((terms['pub'][k] / full[k]).std()):.3f}" for k in live)
        print(f"  sd of the 30% term over the term on all images: {spread}")
        print(f"  {'pair':<24}{'full':>7}{'pub mean':>10}{'pub sd':>8}{'70% sd':>8}{'flip':>7}")
        for r in pair_table(terms["pub"], terms["rest"], full, PAIRS):
            print(
                f"  {r['pair']:<24}{r['full']:7.3f}{r['pub_mean']:10.3f}"
                f"{r['pub_sd']:8.3f}{r['rest_sd']:8.3f}{r['flip']:7.2f}"
            )


if __name__ == "__main__":
    main()
