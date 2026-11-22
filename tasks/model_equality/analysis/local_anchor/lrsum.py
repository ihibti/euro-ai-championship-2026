import argparse
import json
import math
from collections import defaultdict


def load(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["key"]] = r
    return out


def in_half(key, half):
    if half == "all":
        return True
    rep = int(key.rsplit("|", 1)[1])
    return (rep % 2 == 0) == (half == "even")


def lr_table_rows(plain, fp8, tau, exclude_ids=(), half="all"):
    per = defaultdict(list)
    for k, rp in plain.items():
        rq = fp8.get(k)
        if rq is None or rp["id"] in exclude_ids or not in_half(k, half):
            continue
        a, b = rp["lp"][tau], rq["lp"][tau]
        if len(a) != len(b):
            continue
        per[rp["endpoint"]].append((sum(y - x for x, y in zip(a, b)), len(a)))
    rows = {}
    for ep, xs in per.items():
        s = sum(x for x, _ in xs)
        n = len(xs)
        m = s / n
        var = sum((x - m) ** 2 for x, _ in xs) / max(n - 1, 1)
        rows[ep] = {"answers": n, "n_tok": sum(t for _, t in xs), "lr": s, "se": math.sqrt(var * n), "per_tok": s / max(sum(t for _, t in xs), 1)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--schemes", default="tensor_merged,tensor_sep,channel")
    ap.add_argument("--taus", default="0.7,0.72,1.0")
    ap.add_argument("--exclude_ids", default="")
    ap.add_argument("--halves", default="all")
    args = ap.parse_args()
    ex = set(x for x in args.exclude_ids.split(",") if x)
    plain = load(f"lr_{args.label}_plain.jsonl")
    for scheme in args.schemes.split(","):
        try:
            fp8 = load(f"lr_{args.label}_{scheme}.jsonl")
        except FileNotFoundError:
            continue
        for tau in args.taus.split(","):
          for half in args.halves.split(","):
            rows = lr_table_rows(plain, fp8, tau, ex, half)
            print(f"scheme {scheme}, tau {tau}, repeats {half}: Σ log p_fp8 − log p_plain per endpoint (SE from answer-level spread)")
            print("| endpoint | answers | tokens | ΣLR | SE | z | per token |")
            print("|---|---|---|---|---|---|---|")
            for ep in sorted(rows):
                r = rows[ep]
                z = r["lr"] / r["se"] if r["se"] > 0 else float("nan")
                print(f"| {ep} | {r['answers']} | {r['n_tok']} | {r['lr']:+.2f} | {r['se']:.2f} | {z:+.2f} | {r['per_tok']:+.5f} |")
            print()


if __name__ == "__main__":
    main()
