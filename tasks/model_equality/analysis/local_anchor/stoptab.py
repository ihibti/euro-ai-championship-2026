import argparse
import math
import sys

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from stop import summarize


def z_of(a):
    s, v = a[0], a[1]
    return s / math.sqrt(v) if v > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--plain", default="0.7|1.0")
    ap.add_argument("--cols", default="0.68|1.0,0.7|1.0,0.72|1.0,0.74|1.0,0.8|1.0,0.9|1.0,1.0|1.0,1.0|0.95,1.05|0.95")
    args = ap.parse_args()
    cols = args.cols.split(",")
    tabs = {c: summarize(args.file, True, c) for c in cols}
    eps = sorted({ep for ep, _ in tabs[cols[0]]})
    print("| endpoint | answers | capped | " + " | ".join(f"z@{c}" for c in cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for ep in eps:
        a0 = tabs[cols[0]][(ep, "all")]
        cells = []
        for c in cols:
            a = tabs[c][(ep, "all")]
            cells.append(f"{z_of(a):+.2f}" + (f" [{a[4]}v]" if a[4] else ""))
        print(f"| {ep} | {a0[2]} | {a0[3]} | " + " | ".join(cells) + " |")
    probes = sorted({p for _, p in tabs[args.plain] if p != "all"})
    print(f"\nper-probe z at {args.plain} (answers)")
    print("| endpoint | " + " | ".join(probes) + " |")
    print("|---|" + "---|" * len(probes))
    t = tabs[args.plain]
    for ep in eps:
        cells = []
        for p in probes:
            a = t.get((ep, p))
            cells.append("-" if a is None else f"{z_of(a):+.1f} ({a[2]})")
        print(f"| {ep} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
