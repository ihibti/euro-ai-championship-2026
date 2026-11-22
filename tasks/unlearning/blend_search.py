"""Search logit blends of two or three fine-tunes, and their factor, under score_fit's posterior.

One-shot analysis of 22 November, 09:13: every pair of the 16 members below at weights 1:1, 1:2
and 2:1, and every equal-weight trio, each at factors 1.6 to 2.3, scored with `score_fit.py
--retain control --proxy-temperature 1.25` (the best-fitting model) and anchored on the 0.2858.
A weight average is screened through the same average of its members' logits (checked on seed
twins to 0.02 in forget loss; less tested across recipes). Prints and writes the 25 best lines.

    cd <team folder> && python <repo>/tasks/unlearning/blend_search.py <repo>/tasks/unlearning o
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import score_fit as sf

inputs = json.loads((Path(sys.argv[1]) / "ledger_inputs.json").read_text())
specs = inputs["candidates"]
root = Path("/Volumes/euroai-apfs/r21")
ex, jt = sf.load_logits(root, "exact"), sf.load_logits(root, "jitter")
orig_t = torch.from_numpy(ex["orig"])
labels_t = orig_t.argmax(1)
with np.load("output/kkrgp/xfit_extras.npz") as d:
    proxy = torch.from_numpy(d["pred_exact"]).float()
step = proxy - orig_t
control = (
    proxy / 1.25,
    torch.from_numpy(np.random.default_rng(0).standard_normal(ex["orig"].shape)).float(),
)
MEMBERS = [
    "k8_e3t2_rm1_lr1e4",
    "k11_b_s1",
    "k11_b_s2",
    "k11_b_ep20",
    "k11_b_rm2",
    "k11_b_bl4",
    "k11_b_k25",
    "k8_e3t2_rm1_ep40",
    "k9_e3t2_lr1e4_ep40",
    "k9_e35t2_lr1e4",
    "k9_e3t1_lr1e4",
    "k8_e3t1_rm1",
    "k8_e3t15_rm1",
    "k10_t15_lr2e4_ep40",
    "k10_t2_lr2e4_ep40_s1",
    "k9_e3t2_lr2e4",
]
blends = {}
for a, b in itertools.combinations(MEMBERS, 2):
    for wa in (1, 2):
        for wb in (1, 2):
            if wa == wb == 2:
                continue
            blends[f"{wa}*{a}+{wb}*{b}"] = ((a, float(wa)), (b, float(wb)))
for trio in itertools.combinations(MEMBERS, 3):
    blends["+".join(trio)] = tuple((n, 1.0) for n in trio)
sf.BLENDS.update(blends)
sf.BLENDS["k11_b_soup3"] = sf.SOUP
observed = []
known = set(specs) | set(sf.BLENDS) | {f"scale_t{t}" for t in (2, 3, 4, 5, 6, 8)}
for name, score in inputs["observed"].items():
    parsed = sf.parse_line(name, known)
    if parsed is not None:
        observed.append((name, parsed[0], parsed[1], score))
FACTORS = (1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3)
entries = [(n, b, f) for n, b, f, _ in observed] + [
    (f"{b}x{f:g}", b, f) for b in blends for f in FACTORS
]
print("lines:", len(entries), flush=True)
retain, flips = [], []
cache = {}
for _, base, factor in entries:
    if base not in cache:
        cache[base] = sf.retain_function(base, ex, specs, orig_t, control)
    fn, flip = cache[base]
    retain.append(fn(factor))
    flips.append(flip)
retain, flips = np.array(retain), np.array(flips)
ms_parts, pred_parts = [], []
for store in (ex, jt):
    _, stats_v, pred_v, orig_stats_v = sf.view_terms(entries, store, step, labels_t)
    ms_parts.append(sf.ms_table(stats_v, orig_stats_v))
    pred_parts.append(pred_v)
    print("view done", flush=True)
ms = sum(ms_parts) / 2
pred = sum(pred_parts) / 2
n = len(observed)
scores = np.array([o[3] for o in observed])
w = sf.normalise(sf.log_likelihood(pred[:, :n], ms[:n], retain[:n], scores, 0.1))
vals = sf.predicted(sf.draw(w, 3000), pred, ms, retain)
soup_idx = [i for i, o in enumerate(observed) if o[0] == "k11_b_soup3x2"][0]
ref = vals[:, soup_idx]
means = vals.mean(0)
order = np.argsort(-means[n:]) + n
out = []
for j in order[:25]:
    out.append(
        (
            entries[j][0],
            round(float(0.2858 * means[j] / ref.mean()), 4),
            round(float((vals[:, j] > ref).mean()), 2),
            round(float(flips[j]), 3),
        )
    )
    print(out[-1], flush=True)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
