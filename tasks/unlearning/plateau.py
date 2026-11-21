"""Which gold makes the cheap recipes land on the board's plateaus.

Seven unrelated teams share 0.16 and four share 0.12. Equal scores from different code mean a
recipe whose result does not depend on its settings: a limit (outputs driven to uniform on the
forget images) or the best point of a one-knob family (every logit divided by one temperature).
Gold is unknown, so it is parametrised: gold logits = (orig + K * step) / tau_g, where step goes
from the original logits to the cross-fitted held-out prediction (K sets how often and how far
gold is wrong, tau_g how soft it is). For each gold, the server's formulas score:

- uniform: equal logits on every forget image, the rest of the model unchanged (retain 1);
- temper: forget logits divided by t, rest unchanged (retain 1); best t;
- scale: every logit divided by t, so the control set drifts too; best t, with the control set
  taken as memorised training images (the original logits on the forget images stand in) or as
  unseen images (the cross-fitted prediction stands in);
- m80: our submitted x3_ext_m80, whose losses match the K = 0.9 target, retain taken as 0.97.

A gold is consistent with the board when m80 lands near 0.0063 and two cheap recipes land near
0.16 and 0.12. Exact images only.

    python <repo>/tasks/unlearning/plateau.py --targets output/kkrgp/xfit_extras.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lib
import numpy as np
import torch
import torch.nn.functional as F
from dial_matrix import forget_terms, label_loss

TEMPS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)


def kl_rows(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    log_ref = F.log_softmax(reference / lib.TEMPERATURE, 1)
    log_other = F.log_softmax(other / lib.TEMPERATURE, 1)
    return (log_ref.exp() * (log_ref - log_other)).sum(1)


def retain_score(sub: torch.Tensor, orig: torch.Tensor, labels: torch.Tensor) -> float:
    acc_sub = float((sub.argmax(1) == labels).float().mean())
    acc_orig = float((orig.argmax(1) == labels).float().mean())
    accuracy = min(max(acc_sub / acc_orig, 0.0), 1.0)
    return accuracy / (1.0 + float(kl_rows(orig, sub).mean()))


def forget_score(sub: torch.Tensor, gold: torch.Tensor, orig: torch.Tensor, labels) -> float:
    mean_std, prediction = forget_terms(sub, gold, orig, labels)
    return mean_std * prediction


def best_over(scores: dict[float, float]) -> tuple[float, float]:
    temp = max(scores, key=scores.get)
    return temp, scores[temp]


def recipes(
    orig: torch.Tensor, step: torch.Tensor, gold: torch.Tensor, labels: torch.Tensor
) -> dict[str, object]:
    uniform = forget_score(torch.zeros_like(orig), gold, orig, labels)
    temper = {t: forget_score(orig / t, gold, orig, labels) for t in TEMPS}
    unseen = orig + step
    scale_train = {
        t: forget_score(orig / t, gold, orig, labels) * retain_score(orig / t, orig, labels)
        for t in TEMPS
    }
    scale_unseen = {
        t: forget_score(orig / t, gold, orig, labels) * retain_score(unseen / t, unseen, labels)
        for t in TEMPS
    }
    m80 = forget_score(orig + 0.9 * step, gold, orig, labels) * 0.97
    return {
        "uniform": round(uniform, 4),
        "temper": best_over(temper),
        "scale_train": best_over(scale_train),
        "scale_unseen": best_over(scale_unseen),
        "m80": round(m80, 4),
    }


def gold_profile(gold: torch.Tensor, orig: torch.Tensor, labels: torch.Tensor) -> dict:
    loss = label_loss(gold, labels)
    return {
        "loss_mean": round(float(loss.mean()), 3),
        "loss_std": round(float(loss.std()), 3),
        "acc": round(float((gold.argmax(1) == labels).float().mean()), 3),
        "kl_gold_orig": round(float(kl_rows(gold, orig).mean()), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--orig", type=Path, default=Path("output/kkrgp/orig_exact.npy"))
    parser.add_argument("--ks", default="1,1.25,1.5,2,3")
    parser.add_argument("--gold-taus", default="1,1.5,2,3,5")
    parser.add_argument("--out", type=Path, default=Path("output/kkrgp/plateau.json"))
    args = parser.parse_args(argv)
    orig = torch.from_numpy(np.load(args.orig)).float()
    labels = orig.argmax(1)
    with np.load(args.targets) as data:
        proxy = torch.from_numpy(data["pred_exact"]).float()
        columns = torch.from_numpy(data["columns"])
    step = lib.heldout_delta(orig, proxy, columns)
    rows = []
    print(
        "K     tau_g | gold loss mean/std acc  KL(g|o) | uniform temper(t)    "
        "scale_train(t) scale_unseen(t) m80"
    )
    for k in (float(x) for x in args.ks.split(",")):
        for tau in (float(x) for x in args.gold_taus.split(",")):
            gold = (orig + k * step) / tau
            row = {"K": k, "tau_g": tau, **gold_profile(gold, orig, labels)}
            row.update(recipes(orig, step, gold, labels))
            rows.append(row)
            print(
                f"{k:<5} {tau:<5} | {row['loss_mean']:6.2f}/{row['loss_std']:<5.2f} "
                f"{row['acc']:.2f} {row['kl_gold_orig']:6.2f} | {row['uniform']:.3f}   "
                f"{row['temper'][1]:.3f}({row['temper'][0]:g})  "
                f"{row['scale_train'][1]:.3f}({row['scale_train'][0]:g})   "
                f"{row['scale_unseen'][1]:.3f}({row['scale_unseen'][0]:g})    {row['m80']:.4f}",
                flush=True,
            )
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
