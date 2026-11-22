"""How different are the 0.2837's seed twins, in the terms the metric uses?

For the 0.2837's base (`k8_e3t2_rm1_lr1e4`) and its two seed twins (`k11_b_s1`, `k11_b_s2`), each
with its last layer times 2, on the exact forget images:
- how often they pick different classes;
- the KL at temperature 2 from their probability mixture to their logit average (what a weight
  average approximates), to each twin, and to the original: the prediction term's scale;
- the forget-set loss (mean/spread, temperature 1) of each twin, of the logit average and of the
  mixture, with the original's predictions as labels.

If the mixture sits as close to the logit average as the numbers say, distilling the twins'
mixture into one network cannot move the prediction term, and more seeds in the average cannot
either.

    cd <team folder> && python <repo>/tasks/unlearning/twins.py
"""

from __future__ import annotations

import numpy as np
from split_noise import LOGITS, log_softmax

TWINS = (("eighth", "k8_e3t2_rm1_lr1e4"), ("eleventh", "k11_b_s1"), ("eleventh", "k11_b_s2"))
FACTOR = 2.0


def kl_rows(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    return (np.exp(log_p) * (log_p - log_q)).sum(1)


def loss_stats(logits: np.ndarray, labels: np.ndarray) -> str:
    loss = -log_softmax(logits)[np.arange(len(labels)), labels]
    return f"{loss.mean():.2f}/{loss.std(ddof=1):.2f}"


def mixture_and_average(sharp: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Log of the twins' probability mixture at temperature 2, and log-softmax of their mean."""
    probs = np.mean([np.exp(log_softmax(z / 2)) for z in sharp], 0)
    return np.log(probs), log_softmax(np.mean(sharp, 0) / 2)


def main() -> None:
    logits = []
    for batch, name in TWINS:
        with np.load(LOGITS / batch / "logits.npz") as data:
            logits.append(data[f"{name}__exact"].astype(np.float64))
            if batch == "eleventh":
                orig = data["orig__exact"].astype(np.float64)
    labels = orig.argmax(1)
    sharp = [z * FACTOR for z in logits]
    picks = np.stack([z.argmax(1) for z in sharp])
    differ = (picks != picks[0]).any(0)
    print(f"twins pick different classes on {differ.mean():.1%} of the forget images")
    mix, avg = mixture_and_average(sharp)
    scale = kl_rows(mix, log_softmax(orig / 2)).mean()
    print(f"KL at T=2 from the mixture: to the original {scale:.3f}")
    gap = kl_rows(mix, avg)
    print(f"  to the logit average {gap.mean():.4f} ({gap.mean() / scale:.2%} of the original's)")
    for (_, name), z in zip(TWINS, sharp, strict=True):
        k = kl_rows(mix, log_softmax(z / 2)).mean()
        print(f"  to {name} x2 {k:.4f} ({k / scale:.2%})")
    print("forget-set loss at T=1 (mean/spread), labels = the original's predictions:")
    for (_, name), z in zip(TWINS, sharp, strict=True):
        print(f"  {name} x2 {loss_stats(z, labels)}")
    print(f"  logit average x2 {loss_stats(np.mean(sharp, 0), labels)}")
    print(f"  mixture (2 log p at T=2) {loss_stats(2 * mix, labels)}")


if __name__ == "__main__":
    main()
