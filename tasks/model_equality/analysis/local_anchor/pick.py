import json
import sys


def mean_lp(path):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    keys = {r["key"]: r for r in rs if r.get("n_tokens", 0) > 0}
    return keys


a = mean_lp(sys.argv[1])
b = mean_lp(sys.argv[2])
common = sorted(set(a) & set(b))
la = sum(a[k]["mean_lp"] * a[k]["n_tokens"] for k in common) / sum(a[k]["n_tokens"] for k in common)
lb = sum(b[k]["mean_lp"] * b[k]["n_tokens"] for k in common) / sum(b[k]["n_tokens"] for k in common)
print(f"{len(common)} common answers: {sys.argv[1]} {la:.3f} vs {sys.argv[2]} {lb:.3f}", file=sys.stderr)
print(sys.argv[3] if la >= lb else sys.argv[4])
