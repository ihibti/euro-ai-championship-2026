import argparse
import math
import sys
from collections import defaultdict

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from tempfit import load, residuals


def per_probe(recs, tau0, metric):
    by = defaultdict(list)
    if metric == "res":
        for pid, v in residuals(recs, tau0, "raw"):
            by[pid].append(v)
    else:
        for r in recs:
            by[r["id"]].append(r[metric])
    return by


def mv(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, v, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tau0", type=float, required=True)
    ap.add_argument("--metric", default="res")
    ap.add_argument("--min_n", type=int, default=2)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    recs = load([args.scores], ".", set(), set())
    by_ep = defaultdict(list)
    for r in recs:
        by_ep[r["endpoint"]].append(r)
    pool = []
    for e in args.pool.split(","):
        pool += by_ep[e]
    pp = per_probe(pool, args.tau0, args.metric)
    print(f"metric={args.metric} tau0={args.tau0} pool={args.pool}")
    print("| endpoint | probes | mean z | SD of z | probes z<-2 | probes z>+2 | Σz² (chi2, df=probes) |")
    print("|---|---|---|---|---|---|---|")
    for ep in args.endpoints.split(","):
        pe = per_probe(by_ep[ep], args.tau0, args.metric)
        zs = {}
        for p in pe:
            if p not in pp or len(pe[p]) < args.min_n or len(pp[p]) < args.min_n:
                continue
            ma, va, na = mv(pe[p])
            mb, vb, nb = mv(pp[p])
            se = math.sqrt(va / na + vb / nb)
            if se > 0:
                zs[p] = (ma - mb) / se
        if not zs:
            continue
        z = list(zs.values())
        m = sum(z) / len(z)
        sd = math.sqrt(sum((x - m) ** 2 for x in z) / (len(z) - 1)) if len(z) > 1 else 0
        print(f"| {ep} | {len(z)} | {m:+.2f} | {sd:.2f} | {sum(1 for x in z if x < -2)} | {sum(1 for x in z if x > 2)} | {sum(x*x for x in z):.0f} |")
        if args.show:
            print("|  | " + ", ".join(f"{p}:{v:+.1f}" for p, v in sorted(zs.items())) + " |||||")


if __name__ == "__main__":
    main()
