"""The server's unlearning formulas (README, Metric), as pure functions on logits and statistics.

Losses use the label at temperature 1; the distributions compared by KL use temperature 2. A
group is one set of images scored together (the exact images, or the altered copies): the two
forget terms are averaged over the groups first and then multiplied, as on the server.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

TEMPERATURE = 2.0


def clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, 1).gather(1, labels.view(-1, 1)).squeeze(1)


def kl_rows(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """KL(p_reference || p_other) per image, both at temperature 2."""
    log_ref = F.log_softmax(reference / TEMPERATURE, 1)
    log_other = F.log_softmax(other / TEMPERATURE, 1)
    return (log_ref.exp() * (log_ref - log_other)).sum(1)


def label_prob_t2(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits / TEMPERATURE, 1).gather(1, labels.view(-1, 1)).squeeze(1)


def mean_std_score(
    sub: tuple[float, float], gold: tuple[float, float], orig: tuple[float, float]
) -> float:
    """1 - clip(d_sub / d_orig) for (mean, std) pairs of the loss."""
    d_sub = abs(sub[0] - gold[0]) + abs(sub[1] - gold[1])
    d_orig = abs(orig[0] - gold[0]) + abs(orig[1] - gold[1])
    if d_orig == 0:
        return 0.0
    return 1.0 - clip01(d_sub / d_orig)


def prediction_score(kl_gold_sub: float, kl_gold_orig: float) -> float:
    if kl_gold_orig == 0:
        return 0.0
    return 1.0 - clip01(kl_gold_sub / kl_gold_orig)


def loss_stats(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    loss = label_loss(logits, labels)
    return float(loss.mean()), float(loss.std())


def forget_terms(
    sub: torch.Tensor,
    gold: torch.Tensor,
    orig: torch.Tensor,
    labels: torch.Tensor,
    sub_stats: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """(mean_std_score, prediction_score) for one group. sub_stats overrides the loss statistics of
    sub, for a candidate whose loss mean and std were measured but whose logits are a proxy."""
    stats = sub_stats if sub_stats is not None else loss_stats(sub, labels)
    mean_std = mean_std_score(stats, loss_stats(gold, labels), loss_stats(orig, labels))
    prediction = prediction_score(
        float(kl_rows(gold, sub).mean()), float(kl_rows(gold, orig).mean())
    )
    return mean_std, prediction


def forget_score(groups: Sequence[tuple[float, float]]) -> float:
    """Average each term over the groups, then multiply."""
    mean_std = sum(group[0] for group in groups) / len(groups)
    prediction = sum(group[1] for group in groups) / len(groups)
    return mean_std * prediction


def retain_score(sub: torch.Tensor, orig: torch.Tensor, labels: torch.Tensor) -> float:
    acc_sub = float((sub.argmax(1) == labels).float().mean())
    acc_orig = float((orig.argmax(1) == labels).float().mean())
    accuracy = clip01(acc_sub / acc_orig) if acc_orig > 0 else 0.0
    return accuracy / (1.0 + float(kl_rows(orig, sub).mean()))
