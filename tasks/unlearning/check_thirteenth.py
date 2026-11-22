"""The four checks fixed on #21 at 06:24 for the thirteenth batch (mixture of three cross-fits).

`k13_mix_soup_x2.pt` is queued for the last night slot only if all four hold:
1. the new cross-fits' held-out class differs from the first cross-fit's on more than 5% of images;
2. at T=2 on the exact images, the line differs from the 0.2858 by more than the seed twins differ
   (KL above 0.026 nats; the 0.2858 is stood in for by the twins' logit average, x2);
3. its forget loss at x2 (labels = the original's predictions) is within 0.4 of 3.8, mean and
   spread;
4. its probe changes at most 6.0% of predictions (`fold_reports.retain_and_flip`).

It also prints whether another factor could rescue the line: the factor that brings its forget
loss to 3.8, as a multiple of its spill temperature (`ledger.effective_temperature`). The night
found the factor must stay near 0.89 of that temperature and never above it.

    cd <team folder> && python <repo>/tasks/unlearning/check_thirteenth.py \\
        /Volumes/euroai-apfs/r21/thirteenth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from fold_reports import retain_and_flip, spill_of
from ledger import effective_temperature
from split_noise import LOGITS, log_softmax

DISAGREE_MIN = 0.05
KL_MIN = 0.026
LOSS_LEVEL = 3.8
LOSS_TOLERANCE = 0.4
FLIP_MAX = 0.060
TWINS = (("eighth", "k8_e3t2_rm1_lr1e4"), ("eleventh", "k11_b_s1"), ("eleventh", "k11_b_s2"))


def class_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.argmax(1) != b.argmax(1)).mean())


def kl_t2(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    lp, lq = log_softmax(p_logits / 2), log_softmax(q_logits / 2)
    return float((np.exp(lp) * (lp - lq)).sum(1).mean())


def loss_stats(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    loss = -log_softmax(logits)[np.arange(len(labels)), labels]
    return float(loss.mean()), float(loss.std(ddof=1))


def factor_for_loss(logits: np.ndarray, labels: np.ndarray, level: float) -> float:
    """The factor on `logits` whose forget-loss mean is `level`, by bisection on [0.5, 8].

    It assumes the mean grows with the factor there, as it does when the line's top class is not
    the label on the images that dominate the loss.
    """
    low, high = 0.5, 8.0
    for _ in range(60):
        mid = (low + high) / 2
        if loss_stats(logits * mid, labels)[0] < level:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def checks(disagree: float, kl: float, loss: tuple[float, float], flip: float) -> dict[str, bool]:
    return {
        "cross-fits disagree": disagree > DISAGREE_MIN,
        "differs from the 0.2858": kl > KL_MIN,
        "loss near 3.8/3.8": all(abs(v - LOSS_LEVEL) <= LOSS_TOLERANCE for v in loss),
        "probe flip at most 6%": flip <= FLIP_MAX,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--targets", type=Path, default=Path("output/kkrgp/xfit_extras.npz"))
    args = parser.parse_args(argv)
    with np.load(args.targets) as data:
        first = data["pred_exact"].astype(np.float64)
    disagree = min(
        class_disagreement(first, np.load(args.folder / f"xfit_extras_s{s}_exact.npy"))
        for s in (1, 2)
    )
    twins = []
    for batch, name in TWINS:
        with np.load(LOGITS / batch / "logits.npz") as data:
            twins.append(data[f"{name}__exact"].astype(np.float64))
            orig = data["orig__exact"].astype(np.float64)
    best = 2.0 * np.mean(twins, 0)
    with np.load(args.folder / "logits.npz") as data:
        new = data["k13_mix_soup_x2__exact"].astype(np.float64)
    kl = kl_t2(best, new)
    loss = loss_stats(new, orig.argmax(1))
    probe = json.loads((args.folder / "d13_mix_probe.json").read_text())
    spill = spill_of(probe)
    retain, flip = retain_and_flip(spill)
    print(f"1. held-out class differs from the first cross-fit on {disagree:.1%} (min over s1, s2)")
    print(f"2. KL at T=2 from the 0.2858 stand-in to the line: {kl:.4f} nats")
    print(f"3. forget loss at x2: {loss[0]:.2f}/{loss[1]:.2f}")
    print(f"4. probe: flip {flip:.1%}, retain {retain}, spill {spill}")
    labels = orig.argmax(1)
    factor = factor_for_loss(new / 2, labels, LOSS_LEVEL)
    temperature = effective_temperature(
        torch.tensor(orig, dtype=torch.float32), torch.tensor(labels), spill["spare_p_lab"]
    )
    sharp = loss_stats(new / 2 * factor, labels)
    rule = loss_stats(new / 2 * 0.89 * temperature, labels)
    print(
        f"   factor for loss {LOSS_LEVEL}: x{factor:.2f} (loss {sharp[0]:.2f}/{sharp[1]:.2f}), "
        f"{factor / temperature:.2f} times the spill temperature {temperature:.2f}; "
        f"at 0.89 times (x{0.89 * temperature:.2f}) the loss is {rule[0]:.2f}/{rule[1]:.2f}"
    )
    result = checks(disagree, kl, loss, flip)
    for name, ok in result.items():
        print(f"   {'pass' if ok else 'FAIL'}  {name}")
    verdict = "queue k13_mix_soup_x2.pt" if all(result.values()) else "leave the slot empty"
    print(f"decision: {verdict}")


if __name__ == "__main__":
    main()
