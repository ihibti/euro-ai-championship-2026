import argparse
import json
import math
import re
from collections import defaultdict


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("n_tokens", 0) > 0:
                        out.append(r)
    return out


def mean_var(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else None
    return m, v, n


def stratified_diff(recs_a, recs_b, metric, min_n=2):
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for r in recs_a:
        by_a[r["id"]].append(r[metric])
    for r in recs_b:
        by_b[r["id"]].append(r[metric])
    num = den = 0.0
    probes = 0
    for pid in set(by_a) & set(by_b):
        if len(by_a[pid]) < min_n or len(by_b[pid]) < min_n:
            continue
        ma, va, na = mean_var(by_a[pid])
        mb, vb, nb = mean_var(by_b[pid])
        num += ma - mb
        den += va / na + vb / nb
        probes += 1
    if probes == 0:
        return None, None, 0
    return num / probes, math.sqrt(den) / probes, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--endpoints", default=".")
    ap.add_argument("--ids", default="")
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--metrics", default="mean_excess,mean_lp,share_rank_gt20,share_out_p08_t10,share_out_gc_t07")
    ap.add_argument("--min_n", type=int, default=2)
    args = ap.parse_args()
    recs = load(args.scores)
    if args.ids:
        keep = set(args.ids.split(","))
        recs = [r for r in recs if r["id"] in keep]
    if args.exclude_ids:
        drop = set(args.exclude_ids.split(","))
        recs = [r for r in recs if r["id"] not in drop]
    rx = re.compile(args.endpoints)
    by_ep = defaultdict(list)
    for r in recs:
        by_ep[r["endpoint"]].append(r)
    ref = by_ep[args.ref]
    metrics = args.metrics.split(",")
    print("| endpoint | n_ans | probes | " + " | ".join(f"Δ{m} (z)" for m in metrics) + " |")
    print("|---|---|---|" + "---|" * len(metrics))
    for ep in sorted(by_ep):
        if not rx.search(ep):
            continue
        cells = [ep, str(len(by_ep[ep]))]
        probes_used = 0
        for m in metrics:
            d, se, k = stratified_diff(by_ep[ep], ref, m, args.min_n)
            probes_used = max(probes_used, k)
            if d is None:
                cells.append("-")
            else:
                z = d / se if se else 0.0
                cells.append(f"{d:+.3f}±{se:.3f} ({z:+.1f})")
        cells.insert(2, str(probes_used))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
