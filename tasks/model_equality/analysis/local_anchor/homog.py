import argparse
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from tempfit import load, argmax_interp, TAUS


def chi2_sf(x, k):
    if x <= 0:
        return 1.0
    a = k / 2.0
    y = x / 2.0
    if y < a + 1:
        term = 1.0 / a
        s = term
        n = 1
        while abs(term) > 1e-12 * abs(s) and n < 10000:
            term *= y / (a + n)
            s += term
            n += 1
        p_lower = s * math.exp(-y + a * math.log(y) - math.lgamma(a))
        return max(0.0, 1.0 - p_lower)
    b = y + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return h * math.exp(-y + a * math.log(y) - math.lgamma(a))


def probe_fits(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["id"]].append(r)
    out = {}
    for p, g in by.items():
        L = np.sum([r["tau_lp_raw"] for r in g], axis=0)
        t, se, edge = argmax_interp(L)
        out[p] = (t, se, edge, len(g), sum(r["n_tokens"] for r in g))
    return out


def homogeneity(fits, min_tok):
    use = {p: v for p, v in fits.items() if not v[2] and v[1] and v[4] >= min_tok}
    if len(use) < 2:
        return None
    w = {p: 1 / v[1] ** 2 for p, v in use.items()}
    pooled = sum(w[p] * use[p][0] for p in use) / sum(w.values())
    chi = sum(w[p] * (use[p][0] - pooled) ** 2 for p in use)
    df = len(use) - 1
    return pooled, chi, df, chi2_sf(chi, df), use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--endpoints", default=".")
    ap.add_argument("--min_tok", type=int, default=50)
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    exclude = set(args.exclude_ids.split(",")) if args.exclude_ids else set()
    recs = load([args.scores], args.endpoints, set(), exclude)
    by_ep = defaultdict(list)
    for r in recs:
        by_ep[r["endpoint"]].append(r)
    print(f"per-probe tau-hat (Fisher SE), probes kept if not at grid edge and >= {args.min_tok} tokens")
    print("| endpoint | probes used | pooled tau | chi2 | df | p | chi2/df |")
    print("|---|---|---|---|---|---|---|")
    for ep in sorted(by_ep):
        fits = probe_fits(by_ep[ep])
        h = homogeneity(fits, args.min_tok)
        if h is None:
            print(f"| {ep} | <2 | - | - | - | - | - |")
            continue
        pooled, chi, df, p, use = h
        print(f"| {ep} | {df+1} | {pooled:.3f} | {chi:.1f} | {df} | {p:.2g} | {chi/df:.2f} |")
        if args.show:
            print("|  | " + ", ".join(f"{q}:{use[q][0]:.2f}±{use[q][1]:.2f}" for q in sorted(use)) + " ||||||")


if __name__ == "__main__":
    main()
