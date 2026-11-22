import argparse
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from tempfit import load, fit_tau, residuals, argmax_interp

GROUPS = {
    "free text": ["txt_story", "txt_health", "d2_story100", "story"],
    "repetition": ["d1_repeat150", "p3_repeat"],
    "battery": ["bat_trivia", "bat_choice", "trivia30", "choice40", "rate40", "assoc40"],
    "sequence": ["seq_alphabet", "seq_count30", "seq_days", "seq_months", "seq_words40", "count30", "digits", "arith30", "sort25", "backwards20", "coins", "p2_binary", "p4_nextword", "d3_openlist"],
    "identity": ["idn_identity"],
    "leak": ["e1_leak_repeat", "e2_leak_quote"],
    "copy-continue": ["p1_copycont", "d4_copycont", "copy"],
}


def group_of(pid):
    for g, ids in GROUPS.items():
        if pid in ids:
            return g
    return "other"


def tau_boot(recs, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    M = np.array([r["tau_lp_raw"] for r in recs])
    t0, _, _ = argmax_interp(M.sum(axis=0))
    ts = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(recs), len(recs))
        t, _, _ = argmax_interp(M[idx].sum(axis=0))
        ts.append(t)
    return t0, float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--plain", default="ref-01")
    ap.add_argument("--targets", default="ref-09,ref-04,ref-07,ref-02,ref-06,ref-08")
    ap.add_argument("--tau0", type=float, default=0.72)
    ap.add_argument("--groups", action="store_true")
    args = ap.parse_args()
    recs = load([args.scores], ".", set(), set())
    by = defaultdict(list)
    for r in recs:
        key = group_of(r["id"]) if args.groups else r["id"]
        by[(r["endpoint"], key)].append(r)
    keys = sorted({k for _, k in by})
    targets = args.targets.split(",")
    print(f"fixed tau0={args.tau0}; d = per-answer effect of residual@tau0 vs {args.plain}; z10 = d*sqrt(10) for 10 answers against a known base; tau-hat per probe ± boot SE")
    print("| probe | tokens/answer (plain) | " + " | ".join(f"{t} d (z10)" for t in targets) + " | " + " | ".join(f"tau {e}" for e in [args.plain] + targets) + " |")
    print("|" + "---|" * (2 + 2 * len(targets) + 1))
    for k in keys:
        base = by.get((args.plain, k), [])
        if len(base) < 3:
            continue
        rb = [v for _, v in residuals(base, args.tau0, "raw")]
        mb = sum(rb) / len(rb)
        cells = [k, f"{sum(r['n_tokens'] for r in base)/len(base):.0f}"]
        for t in targets:
            g = by.get((t, k), [])
            if len(g) < 3:
                cells.append("-")
                continue
            rt = [v for _, v in residuals(g, args.tau0, "raw")]
            mt = sum(rt) / len(rt)
            va = np.var(rb, ddof=1)
            vt = np.var(rt, ddof=1)
            sd = math.sqrt((va + vt) / 2) if (va + vt) > 0 else float("nan")
            d = (mt - mb) / sd if sd and sd == sd else float("nan")
            cells.append(f"{d:+.2f} ({d*math.sqrt(10):+.1f})")
        for e in [args.plain] + targets:
            g = by.get((e, k), [])
            if len(g) < 3:
                cells.append("-")
                continue
            t0, se = tau_boot(g, 100)
            cells.append(f"{t0:.2f}±{se:.2f}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
