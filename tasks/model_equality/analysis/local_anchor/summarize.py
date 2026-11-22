import argparse
import json
import math
import re
from collections import defaultdict

METRICS = ["mean_lp", "mean_excess", "share_rank_gt20", "share_out_p08_t07", "share_out_p08_t10",
           "share_rankpen_gt20", "share_out_gc_t07", "share_out_gc_t10", "worst_rank_pen", "max_ma_pen_k20_t07"]


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, None
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def token_weighted(recs, key):
    num = sum(r[key] * r["n_tokens"] for r in recs)
    den = sum(r["n_tokens"] for r in recs)
    return num / den if den else None


def load(paths, pattern, ids):
    rx = re.compile(pattern)
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("n_tokens", 0) == 0:
                    continue
                if rx.search(r["endpoint"]) and (not ids or r["id"] in ids):
                    out.append(r)
    return out


def table(recs, metrics, by_probe=False):
    groups = defaultdict(list)
    for r in recs:
        k = (r["endpoint"], r["id"]) if by_probe else (r["endpoint"],)
        groups[k].append(r)
    lines = ["| " + " | ".join(["endpoint"] + (["probe"] if by_probe else []) + ["n_ans", "n_tok"] + metrics + ["tok-weighted excess", "max worst_rank_pen", "answers with gc-violation t0.7"]) + " |"]
    lines.append("|" + "---|" * (len(metrics) + (6 if by_probe else 5)))
    for k in sorted(groups):
        g = groups[k]
        cells = list(k) + [str(len(g)), str(sum(r["n_tokens"] for r in g))]
        for m in metrics:
            mu, se = mean_se([r[m] for r in g])
            cells.append(f"{mu:.3f}" + (f"±{se:.3f}" if se is not None else ""))
        cells.append(f"{token_weighted(g, 'mean_excess'):.3f}")
        cells.append(str(max(r["worst_rank_pen"] for r in g)))
        cells.append(f"{sum(1 for r in g if r['n_out_gc_t07'] > 0)}/{len(g)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--endpoints", default=".")
    ap.add_argument("--ids", default="")
    ap.add_argument("--by_probe", action="store_true")
    ap.add_argument("--metrics", default=",".join(METRICS))
    args = ap.parse_args()
    ids = set(args.ids.split(",")) if args.ids else set()
    recs = load(args.scores, args.endpoints, ids)
    print(table(recs, args.metrics.split(","), args.by_probe))


if __name__ == "__main__":
    main()
