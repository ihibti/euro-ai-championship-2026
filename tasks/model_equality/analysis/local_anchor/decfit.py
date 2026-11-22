import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max_viol", type=int, default=2)
    ap.add_argument("--plain_tau", default="0.7")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    ntok = defaultdict(int)
    nans = defaultdict(int)
    seen = set()
    for f in args.files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            ep = r["endpoint"]
            ntok[ep] += r["n_tokens"]
            nans[ep] += 1
            for k, (s, nv) in r["dec"].items():
                a = agg[ep][k]
                a[0] += s
                a[1] += nv
    print("| endpoint | answers | tokens | best fitting decoder (tau, top-p, top-k) [viol] | ΣlogL margin over plain (" + args.plain_tau + ",1.0,none) | over best untruncated tau | next best fitting decoders | best untruncated tau |")
    print("|---|---|---|---|---|---|---|---|")
    for ep in sorted(agg):
        d = agg[ep]
        fitting = [(k, v) for k, v in d.items() if v[1] <= args.max_viol]
        fitting.sort(key=lambda kv: -kv[1][0])
        untr = [(k, v) for k, v in d.items() if k.endswith("|1.0|0")]
        untr.sort(key=lambda kv: -kv[1][0])
        plain = d.get(f"{args.plain_tau}|1.0|0")
        if not fitting:
            best_any = min(d.items(), key=lambda kv: kv[1][1])
            print(f"| {ep} | {nans[ep]} | {ntok[ep]} | none with <= {args.max_viol} violations (fewest: {best_any[0]} [{best_any[1][1]}]) | - | - | - | {untr[0][0].split('|')[0]} |")
            continue
        bk, bv = fitting[0]
        fmt = lambda k: "(" + ", ".join(x if x != "0" else "none" for x in k.split("|")) + ")"
        nxt = "; ".join(f"{fmt(k)} {v[0]-bv[0]:+.1f} [{v[1]}]" for k, v in fitting[1:args.top])
        print(f"| {ep} | {nans[ep]} | {ntok[ep]} | {fmt(bk)} [{bv[1]}] | {bv[0]-plain[0]:+.1f} | {bv[0]-untr[0][1][0]:+.1f} | {nxt} | {untr[0][0].split('|')[0]} |")


if __name__ == "__main__":
    main()
