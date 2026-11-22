import argparse
import json
import math
from collections import defaultdict

D_PROBES = {"d1_repeat150", "d2_story100", "d3_openlist", "d4_copycont"}


def stage_of(key, d_min=10, r1_min=4, r1_max=13, b_min=10):
    src, ep, pid, rep = key.split("|")
    rep = int(rep)
    if src == "night-b.jsonl" and pid == "p3_repeat" and rep >= b_min:
        return "s1"
    if src == "round1.jsonl" and r1_min <= rep <= r1_max:
        return "s2" if pid in ("bat_trivia", "idn_identity") else "s4"
    if src == "night-d.jsonl" and pid in D_PROBES and rep >= d_min:
        return "s3"
    return None


def is_fresh(key, d_min, r1_min, r1_max):
    return stage_of(key, d_min, r1_min, r1_max) is not None


def old_keys(path):
    out = set()
    try:
        for line in open(path):
            try:
                out.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    except FileNotFoundError:
        pass
    return out


def fresh_table(path, setting, d_min=10, r1_min=4, r1_max=13, by_probe=False, exclude=frozenset()):
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    seen = set(exclude)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = r["key"]
        if k in seen or not is_fresh(k, d_min, r1_min, r1_max):
            continue
        seen.add(k)
        s, v, _ = r["multi"][setting]
        st = stage_of(k, d_min, r1_min, r1_max)
        for kk in [(r["endpoint"], "all"), (r["endpoint"], st)] + ([(r["endpoint"], r["id"])] if by_probe else []):
            a = agg[kk]
            a[0] += s
            a[1] += v
            a[2] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--setting", default="0.72|1.0")
    ap.add_argument("--by_probe", action="store_true")
    ap.add_argument("--exclude_file", default="stop_refs.jsonl")
    args = ap.parse_args()
    agg = fresh_table(args.file, args.setting, by_probe=args.by_probe, exclude=old_keys(args.exclude_file))
    print("| endpoint | part | answers | Σ(ended − h) | sd | z |")
    print("|---|---|---|---|---|---|")
    for (ep, part), (s, v, n) in sorted(agg.items()):
        sd = math.sqrt(v) if v > 0 else float("nan")
        print(f"| {ep} | {part} | {n} | {s:+.2f} | {sd:.2f} | {s/sd:+.2f} |")


if __name__ == "__main__":
    main()
