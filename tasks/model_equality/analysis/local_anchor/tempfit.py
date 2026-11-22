import argparse
import json
import math
import re
from collections import defaultdict

import numpy as np

TAUS = np.array([round(0.4 + 0.05 * i, 3) for i in range(23)])
PROMPTED = {"ep-02", "ep-23", "ep-08", "ep-20", "ep-29", "ep-16"}


def load(paths, pattern, ids, exclude):
    rx = re.compile(pattern)
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("n_tokens", 0) == 0 or "tau_lp_raw" not in r:
                    continue
                if not rx.search(r["endpoint"]):
                    continue
                if ids and r["id"] not in ids:
                    continue
                if r["id"] in exclude:
                    continue
                out.append(r)
    return out


def argmax_interp(L, taus=TAUS):
    g = int(np.argmax(L))
    if g == 0 or g == len(L) - 1:
        return float(taus[g]), None, True
    y0, y1, y2 = L[g - 1], L[g], L[g + 1]
    h = taus[1] - taus[0]
    den = y0 - 2 * y1 + y2
    if den >= 0:
        return float(taus[g]), None, False
    off = 0.5 * (y0 - y2) / den
    tau = taus[g] + off * h
    curv = den / (h * h)
    return float(tau), float(1.0 / math.sqrt(-curv)), False


def interp(arr, tau, taus=TAUS):
    return float(np.interp(tau, taus, arr))


def fit_tau(recs, variant):
    L = np.zeros(len(TAUS))
    for r in recs:
        L += np.array(r["tau_lp_" + variant])
    return argmax_interp(L)


def bootstrap_tau(recs, variant, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    M = np.array([r["tau_lp_" + variant] for r in recs])
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(recs), len(recs))
        t, _, _ = argmax_interp(M[idx].sum(axis=0))
        taus.append(t)
    return float(np.std(taus))


def residuals(recs, tau, variant):
    out = []
    for r in recs:
        lp = interp(r["tau_lp_" + variant], tau)
        h = interp(r["tau_h_" + variant], tau)
        out.append((r["id"], -(lp + h) / r["n_tokens"]))
    return out


def stratified(pairs):
    by = defaultdict(list)
    for pid, v in pairs:
        by[pid].append(v)
    return by


def strat_mean(by, probes):
    tot = var = 0.0
    k = 0
    for p in probes:
        xs = by.get(p, [])
        if len(xs) < 2:
            continue
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        tot += m
        var += v / len(xs)
        k += 1
    if k == 0:
        return None, None, 0
    return tot / k, math.sqrt(var) / k, k


def strat_diff(by_a, by_b, probes):
    tot = var = 0.0
    k = 0
    for p in probes:
        xa, xb = by_a.get(p, []), by_b.get(p, [])
        if len(xa) < 2 or len(xb) < 2:
            continue
        ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
        va = sum((x - ma) ** 2 for x in xa) / (len(xa) - 1)
        vb = sum((x - mb) ** 2 for x in xb) / (len(xb) - 1)
        tot += ma - mb
        var += va / len(xa) + vb / len(xb)
        k += 1
    if k == 0:
        return None, None
    return tot / k, math.sqrt(var) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--endpoints", default=".")
    ap.add_argument("--ids", default="")
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--ref_group", default="")
    ap.add_argument("--local_ref", default="")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--by_probe", action="store_true")
    ap.add_argument("--fixed_tau", type=float, default=0.0)
    ap.add_argument("--per_probe_tau", action="store_true")
    args = ap.parse_args()
    ids = set(args.ids.split(",")) if args.ids else set()
    exclude = set(args.exclude_ids.split(",")) if args.exclude_ids else set()
    recs = load(args.scores, args.endpoints, ids, exclude)
    by_ep = defaultdict(list)
    for r in recs:
        by_ep[r["endpoint"]].append(r)
    probes = sorted({r["id"] for r in recs})
    fits = {}
    for ep, g in by_ep.items():
        tau, se_fisher, edge = fit_tau(g, args.variant)
        se_boot = bootstrap_tau(g, args.variant, args.boot) if args.boot else None
        res = stratified(residuals(g, args.fixed_tau or tau, args.variant))
        ent = stratified([(r["id"], r["mean_ent"]) for r in g])
        rk = stratified([(r["id"], r["share_rank_gt20"]) for r in g])
        fits[ep] = dict(tau=tau, se_f=se_fisher, se_b=se_boot, edge=edge, res=res, ent=ent, rk=rk,
                        n=len(g), ntok=sum(r["n_tokens"] for r in g))
    for ep, f in fits.items():
        f["res_m"], f["res_se"], f["k"] = strat_mean(f["res"], probes)
    if args.ref_group:
        group = args.ref_group.split(",")
    else:
        cands = [e for e in fits if e.startswith("ep-") or e.startswith("ref-")]
        cands = [e for e in cands if e not in PROMPTED and fits[e]["res_m"] is not None]
        group = sorted(cands, key=lambda e: fits[e]["res_m"])[:1]
    pooled_res = defaultdict(list)
    for e in group:
        for p, xs in fits[e]["res"].items():
            pooled_res[p].extend(xs)
    print(f"variant={args.variant} residual at {'fixed tau ' + str(args.fixed_tau) if args.fixed_tau else 'tau_hat'} probes={','.join(probes)}")
    print(f"reference group for z: {','.join(group)}" + (f"; second z vs {args.local_ref}" if args.local_ref else ""))
    hdr = ["endpoint", "n_ans", "n_tok", "tau_hat ± se(boot) [fisher]", "residual@tau_hat (equal-weight probes)", "Δres vs group (z)"]
    if args.local_ref:
        hdr.append(f"Δres vs {args.local_ref} (z, shared probes)")
    hdr += ["mean entropy", "share rank>20"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for ep in sorted(fits):
        f = fits[ep]
        tag = " (prompted)" if ep in PROMPTED else ""
        tau_s = f"{f['tau']:.3f}" + (f" ± {f['se_b']:.3f}" if f["se_b"] is not None else "") + (f" [{f['se_f']:.3f}]" if f["se_f"] else "") + (" (grid edge)" if f["edge"] else "")
        res_s = "-" if f["res_m"] is None else f"{f['res_m']:+.3f} ± {f['res_se']:.3f}"
        d, se = strat_diff(f["res"], pooled_res, probes)
        dz = "-" if d is None or not se else f"{d:+.3f} ({d/se:+.1f})"
        cells = [ep + tag, str(f["n"]), str(f["ntok"]), tau_s, res_s, dz]
        if args.local_ref and args.local_ref in fits:
            d2, se2 = strat_diff(f["res"], fits[args.local_ref]["res"], probes)
            cells.append("-" if d2 is None or not se2 else f"{d2:+.3f} ({d2/se2:+.1f})")
        em, _, _ = strat_mean(f["ent"], probes)
        rm, _, _ = strat_mean(f["rk"], probes)
        cells.append("-" if em is None else f"{em:.3f}")
        cells.append("-" if rm is None else f"{rm:.4f}")
        print("| " + " | ".join(cells) + " |")
    if args.by_probe:
        eps = sorted(fits)
        print()
        print("residual@tau_hat per probe (mean, n)")
        print("| probe | " + " | ".join(eps) + " |")
        print("|" + "---|" * (len(eps) + 1))
        for p in probes:
            cells = [p]
            for ep in eps:
                xs = fits[ep]["res"].get(p, [])
                cells.append(f"{sum(xs)/len(xs):+.3f} ({len(xs)})" if xs else "-")
            print("| " + " | ".join(cells) + " |")
    if args.per_probe_tau:
        eps = sorted(fits)
        print()
        print("tau_hat per probe")
        print("| probe | " + " | ".join(eps) + " |")
        print("|" + "---|" * (len(eps) + 1))
        for p in probes:
            cells = [p]
            for ep in eps:
                g = [r for r in by_ep[ep] if r["id"] == p]
                if not g:
                    cells.append("-")
                    continue
                t, _, edge = fit_tau(g, args.variant)
                cells.append(f"{t:.2f}" + ("*" if edge else ""))
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
