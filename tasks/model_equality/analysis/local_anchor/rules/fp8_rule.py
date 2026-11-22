import argparse
import json
import math
import sys
from collections import defaultdict

POSITIVE_Z = 3.0
STOP_FLAG_Z = -2.0

FAMILIES = {
    "A": {"pair": ["ep-12", "ep-18"], "labelled": {"ep-04": "0.7", "ep-14": "1.0", "ep-33": "1.0"}, "tau": "0.7"},
    "B": {"pair": ["ep-22", "ep-32"], "labelled": {"ep-06": "0.7", "ep-07": "1.0", "ep-30": "1.0"}, "tau": "0.7"},
    "C": {"pair": ["ep-13", "ep-19"], "labelled": {"ep-10": "0.7", "ep-27": "1.0", "ep-31": "1.0"}, "tau": "0.7"},
}


def decide(readings, pair, labelled, stop_z):
    unlabelled = {ep: r for ep, r in readings.items() if ep not in labelled}
    bad_labels = sorted(ep for ep in labelled if ep in readings and readings[ep]["z"] >= POSITIVE_Z)
    if bad_labels:
        return {"add": [], "fp8": None, "case": "check failed: labelled copy fp8-positive " + ",".join(bad_labels)}
    positive = sorted(ep for ep, r in unlabelled.items() if r["z"] >= POSITIVE_Z)
    if len(positive) == 0:
        return {"add": [], "fp8": None, "case": "no fp8-positive endpoint"}
    if len(positive) > 1:
        return {"add": [], "fp8": None, "case": "more than one fp8-positive endpoint: " + ",".join(positive)}
    fp8 = positive[0]
    if fp8 in pair:
        return {"add": [ep for ep in pair if ep != fp8], "fp8": fp8, "case": "fp8 inside the pair"}
    add = []
    for ep in pair:
        if ep not in readings or ep not in stop_z:
            continue
        if readings[ep]["lr"] < 0 and stop_z[ep] > STOP_FLAG_Z:
            add.append(ep)
    return {"add": add, "fp8": fp8, "case": "fp8 outside the pair"}


def load_lr(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["key"]] = r
    return out


def readings_for(plain, fp8, tau_of):
    per = defaultdict(list)
    for key, rp in plain.items():
        rq = fp8.get(key)
        if rq is None:
            continue
        tau = tau_of(rp["endpoint"])
        a, b = rp["lp"][tau], rq["lp"][tau]
        if len(a) != len(b):
            continue
        per[rp["endpoint"]].append((sum(y - x for x, y in zip(a, b)), len(a)))
    out = {}
    for ep, xs in per.items():
        n = len(xs)
        s = sum(x for x, _ in xs)
        m = s / n
        var = sum((x - m) ** 2 for x, _ in xs) / max(n - 1, 1)
        se = math.sqrt(var * n)
        out[ep] = {"answers": n, "tokens": sum(t for _, t in xs), "lr": s, "se": se, "z": s / se if se > 0 else 0.0, "tau": tau_of(ep)}
    return out


def stop_z_from(path):
    agg = defaultdict(lambda: [0.0, 0.0])
    seen = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        agg[r["endpoint"]][0] += r["s"]
        agg[r["endpoint"]][1] += r["v"]
    return {ep: (s / math.sqrt(v) if v > 0 else 0.0) for ep, (s, v) in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    fam = FAMILIES[args.family]
    plain = load_lr(f"{args.dir}/lr_{args.family}_plain.jsonl")
    fp8 = load_lr(f"{args.dir}/lr_{args.family}_tensor_merged.jsonl")
    readings = readings_for(plain, fp8, lambda ep: fam["labelled"].get(ep, fam["tau"]))
    stop = stop_z_from(f"{args.dir}/stop_{args.family}.jsonl")
    print("| endpoint | role | tau | answers | tokens | sum | SE | z | per token | stop z |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ep in sorted(readings):
        r = readings[ep]
        role = "labelled" if ep in fam["labelled"] else ("pair" if ep in fam["pair"] else "")
        sz = f"{stop[ep]:+.2f}" if ep in stop else ""
        print(f"| {ep} | {role} | {r['tau']} | {r['answers']} | {r['tokens']} | {r['lr']:+.1f} | {r['se']:.1f} | {r['z']:+.2f} | {r['lr'] / max(r['tokens'], 1):+.5f} | {sz} |")
    print(json.dumps(decide(readings, fam["pair"], fam["labelled"], stop)))


if __name__ == "__main__":
    sys.exit(main())
