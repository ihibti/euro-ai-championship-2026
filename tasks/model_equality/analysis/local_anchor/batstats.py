import argparse
import json
import math
import re
from collections import Counter, defaultdict

ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
FAMILIES = {
    "A": "01 02 04 12 14 17 18 23 24 25 33",
    "B": "03 06 07 08 20 21 22 26 28 30 32",
    "C": "05 09 10 11 13 15 16 19 27 29 31",
}
PROMPTED = {"ep-02", "ep-23", "ep-08", "ep-20", "ep-29", "ep-16"}
LINE = re.compile(r"^\s*(\d{1,2})\s*[.):]\s*(.*?)\s*$")


def norm(a):
    a = a.lower().strip()
    a = re.sub(r"\*+", "", a)
    a = re.sub(r"[.!]+$", "", a).strip()
    return a


def parse_battery(resp):
    items = {}
    for line in resp.split("\n"):
        m = LINE.match(line)
        if m:
            k = int(m.group(1))
            if 1 <= k <= 30 and k not in items:
                items[k] = norm(m.group(2))
    return items


def well_formed(resp):
    items = parse_battery(resp)
    return len(items) == 30


def d1_style(resp, word="river"):
    s = resp.strip()
    spaced = len(re.findall(r"\b" + word + r"\b", s.lower()))
    total = s.lower().count(word)
    glued = total > 0 and spaced < total * 0.5
    return {"count": total, "spaced": spaced, "glued": glued}


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def modes(rows, ep, pid):
    per_item = defaultdict(Counter)
    n = 0
    for r in rows:
        if r["endpoint"] == ep and r["id"] == pid:
            n += 1
            for k, v in parse_battery(r["response"]).items():
                per_item[k][v] += 1
    m = {}
    for k in range(1, 31):
        if per_item[k]:
            m[k] = per_item[k].most_common(1)[0]
    return n, m, per_item


def mode_distance(ma, mb):
    keys = [k for k in range(1, 31) if k in ma and k in mb]
    if not keys:
        return None, 0
    diff = sum(1 for k in keys if ma[k][0] != mb[k][0])
    return diff / len(keys), len(keys)


def agreement_rate(rows, ep, pid, ref_modes):
    hit = tot = 0
    for r in rows:
        if r["endpoint"] == ep and r["id"] == pid:
            for k, v in parse_battery(r["response"]).items():
                if k in ref_modes:
                    tot += 1
                    hit += v == ref_modes[k][0]
    return (hit / tot if tot else None), tot


def fmt(p, n):
    if p is None or n == 0:
        return "-"
    se = math.sqrt(max(p * (1 - p), 0.25 / n) / n)
    return f"{p:.2f}±{se:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="A")
    ap.add_argument("--local", default="answers.jsonl")
    ap.add_argument("--ref", default="")
    ap.add_argument("--show-items", action="store_true")
    args = ap.parse_args()
    rows = load_rows([ANS_DIR + "/round1.jsonl", ANS_DIR + "/night-d.jsonl", args.local])
    eps = ["ep-" + e for e in FAMILIES[args.family].split()]
    locals_ = sorted({r["endpoint"] for r in rows if r["endpoint"].startswith("local-")})
    ref = args.ref or (locals_[0] if locals_ else None)
    names = eps + locals_
    print(f"reference for distances: {ref}")
    for pid in ("bat_trivia", "bat_choice"):
        _, ref_m, _ = modes(rows, ref, pid)
        fam_counts = defaultdict(Counter)
        for ep in eps:
            if ep in PROMPTED:
                continue
            _, m, _ = modes(rows, ep, pid)
            for k, (v, _) in m.items():
                fam_counts[k][v] += 1
        fam_m = {k: c.most_common(1)[0] for k, c in fam_counts.items()}
        print(f"\n## {pid}: n | well-formed | mode-diff vs ref | per-answer agreement with ref mode | mode-diff vs family-majority (non-prompted)")
        for name in names:
            n, m, _ = modes(rows, name, pid)
            if n == 0:
                continue
            wf = sum(well_formed(r["response"]) for r in rows if r["endpoint"] == name and r["id"] == pid)
            d, nk = mode_distance(m, ref_m)
            ag, nt = agreement_rate(rows, name, pid, ref_m)
            dfam, _ = mode_distance(m, fam_m)
            tag = " (prompted)" if name in PROMPTED else ""
            print(f"{name}{tag} | {n} | {wf}/{n} | {('-' if d is None else f'{d:.2f} ({round(d*nk)}/{nk})')} | {fmt(ag, nt)} | {('-' if dfam is None else f'{dfam:.2f}')}")
        if args.show_items:
            print("items where family majority and ref mode differ:")
            for k in range(1, 31):
                if k in fam_m and k in ref_m and fam_m[k][0] != ref_m[k][0]:
                    print(f"  {k}: family={fam_m[k][0]!r} ({fam_m[k][1]} eps) ref={ref_m[k][0]!r}")
    print("\n## d1_repeat150: n | to-cap-with-spaces | to-cap-glued | stops-early | mean 'river' count")
    for name in names:
        rs = [r for r in rows if r["endpoint"] == name and r["id"] == "d1_repeat150"]
        if not rs:
            continue
        sty = [d1_style(r["response"]) for r in rs]
        cap = [s["count"] >= 120 for s in sty]
        a = sum(1 for s, c in zip(sty, cap) if c and not s["glued"])
        b = sum(1 for s, c in zip(sty, cap) if c and s["glued"])
        c_ = sum(1 for c in cap if not c)
        n = len(rs)
        mean = sum(s["count"] for s in sty) / n
        print(f"{name} | {n} | {a}/{n} | {b}/{n} | {c_}/{n} | {mean:.0f}")


if __name__ == "__main__":
    main()
