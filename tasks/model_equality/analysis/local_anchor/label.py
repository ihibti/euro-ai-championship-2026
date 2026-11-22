import argparse
import math
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from tempfit import load, fit_tau, bootstrap_tau, residuals, stratified, strat_mean
from homog import probe_fits, homogeneity
from signs import per_probe, mv

PROMPTED = {"ep-02", "ep-23", "ep-08", "ep-20", "ep-29", "ep-16"}


def sign_counts(target, pool, tau0, min_n=2):
    pe = per_probe(target, tau0, "res")
    pp = per_probe(pool, tau0, "res")
    zs = []
    for p in pe:
        if p not in pp or len(pe[p]) < min_n or len(pp[p]) < min_n:
            continue
        ma, va, na = mv(pe[p])
        mb, vb, nb = mv(pp[p])
        se = math.sqrt(va / na + vb / nb)
        if se > 0:
            zs.append((ma - mb) / se)
    return sum(1 for z in zs if z < -2), sum(1 for z in zs if z > 2), len(zs)


def decide(ep, tau, se, plain, hp, neg, pos, wr, mm, ntok, trunc_rank, trunc_mass):
    if ep in PROMPTED:
        return "prompted"
    shifted = se is not None and abs(tau - plain) > 3 * max(se, 0.01)
    if wr <= trunc_rank and mm < trunc_mass and ntok >= 1500:
        return "decoding (truncation)"
    if shifted and hp is not None and hp >= 0.05:
        return "decoding (tau shifted, homogeneous)"
    if shifted and hp is not None and hp < 0.01:
        return "changed (tau shifted, heterogeneous)"
    if not shifted and hp is not None and hp < 0.01 and neg >= 1 and pos >= 1:
        return "watermark (plain tau, heterogeneous, mixed sign)"
    if not shifted and hp is not None and hp < 0.01:
        return "heterogeneous at plain tau, one-sided (changed?)"
    if shifted:
        return "tau shifted, homogeneity unclear"
    return "base-like"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--endpoints", default="^ep-")
    ap.add_argument("--plain", type=float, default=0.0)
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--trunc_rank", type=int, default=40)
    ap.add_argument("--trunc_mass", type=float, default=0.96)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    exclude = set(args.exclude_ids.split(",")) if args.exclude_ids else set()
    recs = load([args.scores], args.endpoints, set(), exclude)
    by_ep = defaultdict(list)
    for r in recs:
        by_ep[r["endpoint"]].append(r)
    info = {}
    for ep, g in by_ep.items():
        tau, _, _ = fit_tau(g, "raw")
        se = bootstrap_tau(g, "raw", args.boot)
        h = homogeneity(probe_fits(g), 50)
        info[ep] = dict(tau=tau, se=se, h=h, n=len(g), ntok=sum(r["n_tokens"] for r in g),
                        wr=max(r["worst_rank"] for r in g), mm=max(r["max_ma10"] for r in g))
    nonp = [e for e in info if e not in PROMPTED]
    plain = args.plain or statistics.median(info[e]["tau"] for e in nonp)
    print(f"plain tau = {plain:.3f} ({'given' if args.plain else 'median of non-prompted'})")
    hdr = ["endpoint", "answers", "tokens", "tau-hat ± boot SE", "homog chi2/df (p)", "res@plain z<-2 / z>+2 (probes)", "res@own tau", "worst rank", "max mass above (T1.0)", "label"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for ep in sorted(info):
        f = info[ep]
        pool_eps = [e for e in nonp if e != ep and abs(info[e]["tau"] - plain) < 0.05]
        pool = [r for e in pool_eps for r in by_ep[e]]
        neg, pos, k = sign_counts(by_ep[ep], pool, plain) if pool else (0, 0, 0)
        res_own = stratified(residuals(by_ep[ep], f["tau"], "raw"))
        probes = sorted(res_own)
        rm, rse, _ = strat_mean(res_own, probes)
        if f["h"]:
            pooled, chi, df, p, _ = f["h"]
            hs = f"{chi/df:.2f} ({p:.2g})"
            hp = p
        else:
            hs, hp = "-", None
        lab = decide(ep, f["tau"], f["se"], plain, hp, neg, pos, f["wr"], f["mm"], f["ntok"], args.trunc_rank, args.trunc_mass)
        rs = "-" if rm is None else f"{rm:+.3f}±{rse:.3f}"
        print(f"| {ep}{' (prompted)' if ep in PROMPTED else ''} | {f['n']} | {f['ntok']} | {f['tau']:.3f} ± {f['se']:.3f} | {hs} | {neg} / {pos} ({k}) | {rs} | {f['wr']} | {f['mm']:.3f} | {lab} |")


if __name__ == "__main__":
    main()
