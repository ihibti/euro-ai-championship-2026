import argparse
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from batstats import parse_battery

ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
PROBES = ("bat_trivia", "bat_choice")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r["id"] in PROBES:
                        rows.append(r)
    return rows


def item_counts(responses_by_probe):
    out = {}
    for pid, resps in responses_by_probe.items():
        per = defaultdict(Counter)
        for resp in resps:
            for k, v in parse_battery(resp).items():
                per[k][v] += 1
        for k in range(1, 31):
            out[(pid, k)] = per[k]
    return out


def flips(counts, base):
    total = 0.0
    n = 0
    for key, b in base.items():
        c = counts.get(key)
        if not c or b is None:
            continue
        top = max(c.values())
        tied = [a for a, v in c.items() if v == top]
        n += 1
        if b in tied:
            total += 1 - 1 / len(tied)
        else:
            total += 1
    return total, n


def mode_map(counts):
    m = {}
    for key, c in counts.items():
        m[key] = c.most_common(1)[0][0] if c else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="01,02,04,12,14,17,18,23,24,25,33")
    ap.add_argument("--base", default="local-qwen2.5-3b-bf16-greedy")
    ap.add_argument("--local", default="local-qwen2.5-3b-bf16-plainT0.7")
    ap.add_argument("--local_t1", default="local-qwen2.5-3b-bf16-plainT1.0")
    ap.add_argument("--group", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = load([ANS_DIR + "/round1.jsonl", "answers.jsonl"])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["endpoint"]][r["id"]].append(r["response"])
    base = mode_map(item_counts(by[args.base]))
    local = by[args.local]
    local_mode = mode_map(item_counts(local))
    agree = sum(1 for k in base if base[k] == local_mode.get(k))
    print(f"base = {args.base} (greedy) ; {args.local} n={[len(local[p]) for p in PROBES]} ; greedy vs T0.7-sample mode agree on {agree}/60 items")
    print("| endpoint | repeats (trivia, choice) | items differing from base mode (of 60) | trivia | choice | items differing from T0.7-sample mode |")
    print("|---|---|---|---|---|---|")
    names = ["ep-" + e for e in args.endpoints.split(",")] + [args.local_t1]
    for ep in names:
        c = item_counts(by[ep])
        f, n = flips(c, base)
        ft, _ = flips({k: v for k, v in c.items() if k[0] == "bat_trivia"}, {k: v for k, v in base.items() if k[0] == "bat_trivia"})
        fc, _ = flips({k: v for k, v in c.items() if k[0] == "bat_choice"}, {k: v for k, v in base.items() if k[0] == "bat_choice"})
        f2, _ = flips(c, local_mode)
        print(f"| {ep} | {len(by[ep]['bat_trivia'])}, {len(by[ep]['bat_choice'])} | {f:.1f}/{n} | {ft:.1f} | {fc:.1f} | {f2:.1f} |")
    rnd = random.Random(args.seed)
    null = []
    for rep in range(200):
        sub = {}
        for p in PROBES:
            xs = list(local[p])
            rnd.shuffle(xs)
            sub[p] = xs[: args.group]
        f, n = flips(item_counts(sub), base)
        null.append(f)
    null.sort()
    print(f"null: local T0.7 subsets of {args.group} vs greedy base mode: mean {sum(null)/len(null):.1f}, 5-95% [{null[10]:.1f}, {null[189]:.1f}], max {null[-1]:.1f}")
    halves = []
    for rep in range(200):
        a, b = {}, {}
        for p in PROBES:
            xs = list(local[p])
            rnd.shuffle(xs)
            h = len(xs) // 2
            a[p], b[p] = xs[:h], xs[h:]
        f, _ = flips(item_counts(a), mode_map(item_counts(b)))
        halves.append(f)
    halves.sort()
    print(f"null: local T0.7 half vs other half mode: mean {sum(halves)/len(halves):.1f}, 5-95% [{halves[10]:.1f}, {halves[189]:.1f}]")


if __name__ == "__main__":
    main()
