import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--plain", default="1.0|0.7|1")
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    keep = set(args.ids.split(",")) if args.ids else None
    agg = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    seen = set()
    for line in open(args.file):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["key"] in seen or (keep and r["id"] not in keep):
            continue
        seen.add(r["key"])
        n[r["endpoint"]] += r["n_tokens"]
        for k, v in r["ll"].items():
            agg[r["endpoint"]][k] += v
    print("| endpoint | tokens | best (penalty, tau, prompt counted) | margin over plain (1.0, 0.7) | best at penalty 1.0 | margin of best over best-at-1.0 |")
    print("|---|---|---|---|---|---|")
    for ep in sorted(agg):
        d = agg[ep]
        best = max(d, key=d.get)
        b1 = max((k for k in d if k.startswith("1.0|")), key=d.get)
        print(f"| {ep} | {n[ep]} | {best} | {d[best]-d[args.plain]:+.1f} | {b1} | {d[best]-d[b1]:+.1f} |")


if __name__ == "__main__":
    main()
