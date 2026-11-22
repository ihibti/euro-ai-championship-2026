"""Where each fine-tune, or a weight average of them, sits against the 0.2837's operating point.

The 0.2837's base is the only scored line where both conditions of the night hold at once: its
forget loss is 3.8 at a last-layer factor of 0.89 times its spill temperature T. For every line
this prints:
- T, from its probe's spare-class label probability (`ledger.effective_temperature`);
- the spill factor 0.89 T and the loss factor that brings its forget loss to 3.8; their ratio is 1
  at the operating point, above 1 when the line reaches too little, below 1 when too far;
- its forget loss at the spill factor, its probe's spill and changed predictions, and the KL at
  T=2 from the 0.2837 to it.

A weight average is screened through the same average of its members' logits, and its probe
statistics are the same average of theirs: an approximation, checked on real weights by the
fourteenth batch.

    cd <team folder> && python <repo>/tasks/unlearning/morning_screen.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from check_thirteenth import factor_for_loss, kl_t2, loss_stats
from ledger import effective_temperature
from split_noise import LOGITS

RATIO = 0.89
LEVEL = 3.80
BATCHES = ("seventh", "eighth", "ninth", "tenth", "eleventh", "thirteenth")
BASE = "k8_e3t2_rm1_lr1e4"
BLENDS = (
    (("k9_e3t1_lr1e4", 0.75), ("k8_e3t1_rm1", 0.25)),
    (("k8_e3t15_rm1", 0.5), ("k10_t15_lr2e4_ep40", 0.5)),
    ((BASE, 0.75), ("k9_e3t1_lr1e4", 0.25)),
)
SPILL = ("spare_p_lab", "held_kl", "spare_kl")


def blend(
    parts: tuple[tuple[str, float], ...], logits: dict[str, np.ndarray], specs: dict[str, dict]
) -> tuple[np.ndarray, dict[str, float], float]:
    """Weighted average of the members' logits, probe spill and changed predictions."""
    total = sum(w for _, w in parts)
    z = sum(w * logits[name] for name, w in parts) / total
    spill = {k: sum(w * specs[name]["spill"][k] for name, w in parts) / total for k in SPILL}
    flip = sum(w * specs[name]["flip"] for name, w in parts) / total
    return z, spill, flip


def operating_point(
    z: np.ndarray, labels: np.ndarray, temperature: float
) -> dict[str, float | tuple[float, float]]:
    spill_factor = RATIO * temperature
    loss_factor = factor_for_loss(z, labels, LEVEL)
    return {
        "T": temperature,
        "spill_factor": spill_factor,
        "loss_factor": loss_factor,
        "ratio": loss_factor / spill_factor,
        "loss": loss_stats(z * spill_factor, labels),
    }


def line(name: str, z: np.ndarray, spill: dict, flip: float, orig: np.ndarray, base2) -> str:
    labels = orig.argmax(1)
    t = effective_temperature(
        torch.tensor(orig, dtype=torch.float32), torch.tensor(labels), spill["spare_p_lab"]
    )
    p = operating_point(z, labels, t)
    m, s = p["loss"]
    kl = kl_t2(base2, z * p["spill_factor"])
    return (
        f"{name:44s} T {t:.2f} spill f {p['spill_factor']:.2f} loss f {p['loss_factor']:.2f} "
        f"ratio {p['ratio']:.2f} | loss {m:.2f}/{s:.2f} | held {spill['held_kl']:.2f} "
        f"spare {spill['spare_kl']:.2f} flip {flip:.3f} | KL from 0.2837 {kl:.3f}"
    )


def main() -> None:
    specs = json.loads(Path(__file__).with_name("ledger_inputs.json").read_text())["candidates"]
    logits: dict[str, np.ndarray] = {}
    for batch in BATCHES:
        with np.load(LOGITS / batch / "logits.npz") as data:
            for key in data.files:
                if key.endswith("__exact"):
                    logits[key[: -len("__exact")]] = data[key].astype(np.float64)
    orig = logits["orig"]
    base2 = 2 * logits[BASE]
    print("fine-tunes, at 0.89 T:")
    for name in sorted(n for n in specs if "spill" in specs[n] and n in logits):
        spec = specs[name]
        print(line(name, logits[name], spec["spill"], spec["flip"], orig, base2))
    print("weight averages, screened through their logits:")
    for parts in BLENDS:
        z, spill, flip = blend(parts, logits, specs)
        name = " + ".join(f"{w:g} {n}" for n, w in parts)
        print(line(name, z, spill, flip, orig, base2))


if __name__ == "__main__":
    main()
