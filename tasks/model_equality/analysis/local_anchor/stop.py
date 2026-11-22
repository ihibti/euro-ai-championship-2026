import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")


def stop_probs(z, eos_ids, tau):
    z = z.astype(mx.float32) / tau
    lse = mx.logsumexp(z, axis=-1)
    e = mx.logsumexp(z[:, mx.array(eos_ids)], axis=-1)
    h = mx.exp(e - lse)
    mx.eval(h)
    return h.tolist()


SETTINGS = [(t, 1.0) for t in (0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.85, 0.9, 1.0)] + [(0.85, 0.95), (1.0, 0.95), (1.05, 0.95)]
KSORT = 1024


def stop_probs_multi(z, eos_ids, settings=SETTINGS, ksort=KSORT):
    z = z.astype(mx.float32)
    T, V = z.shape
    ksort = min(ksort, V)
    e_idx = mx.array(eos_ids)
    ze = z[:, e_idx]
    need_sort = any(tp < 1.0 for _, tp in settings)
    if need_sort:
        part = mx.argpartition(-z, kth=ksort - 1, axis=-1)[:, :ksort]
        vals = -mx.sort(-mx.take_along_axis(z, part, axis=-1), axis=-1)
        mx.eval(vals)
    out = {}
    for tau, tp in settings:
        lse = mx.logsumexp(z / tau, axis=-1)
        pe = mx.exp(ze / tau - lse[:, None])
        if tp >= 1.0:
            h = mx.sum(pe, axis=-1)
        else:
            p = mx.exp(z / tau - lse[:, None])
            above = mx.stack([mx.sum(mx.where(z > ze[:, j:j + 1], p, 0.0), axis=-1) for j in range(len(eos_ids))], axis=-1)
            cum = mx.cumsum(mx.exp(vals / tau - lse[:, None]), axis=-1)
            n_below = mx.sum(cum < tp, axis=-1)
            c = mx.minimum(n_below, ksort - 1)
            kept = mx.take_along_axis(cum, c[:, None], axis=-1)[:, 0]
            kept = mx.where(n_below >= ksort, mx.array(tp, dtype=mx.float32), kept)
            h = mx.sum(mx.where(above < tp, pe / kept[:, None], 0.0), axis=-1)
        mx.eval(h)
        out[f"{tau}|{tp}"] = h.tolist()
    return out


def answer_stop_stats_viol(h, ended_last):
    if ended_last and h and h[-1] <= 0.0:
        s, v = answer_stop_stats(h[:-1], False)
        return s, v, 1
    s, v = answer_stop_stats(h, ended_last)
    return s, v, 0


def answer_stop_stats(h, ended_last):
    ended = [0] * len(h)
    if ended_last:
        ended[-1] = 1
    s = sum(e - p for e, p in zip(ended, h))
    v = sum(p * (1 - p) for p in h)
    return s, v


def summarize(path, by_probe=False, setting="", endpoints=None, exclude_ids=()):
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
    seen = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["key"] in seen or r["id"] in exclude_ids:
            continue
        if endpoints is not None and not endpoints.search(r["endpoint"]):
            continue
        seen.add(r["key"])
        if setting:
            s, v, nv = r["multi"][setting]
        else:
            s, v, nv = r["s"], r["v"], 0
        keys = [(r["endpoint"], "all")] + ([(r["endpoint"], r["id"])] if by_probe else [])
        for k in keys:
            a = agg[k]
            a[0] += s
            a[1] += v
            a[2] += 1
            a[3] += r["capped"]
            a[4] += nv
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--local", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--eos", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--by_probe", action="store_true")
    ap.add_argument("--setting", default="")
    ap.add_argument("--exclude_ids", default="")
    args = ap.parse_args()
    if args.summary:
        import re
        agg = summarize(args.out, args.by_probe, args.setting, re.compile(args.endpoints) if args.endpoints else None, set(x for x in args.exclude_ids.split(",") if x))
        print("| endpoint | probe | answers | capped | viol | Σ(ended − h) | sd | z |")
        print("|---|---|---|---|---|---|---|---|")
        for (ep, pid), (s, v, n, c, nv) in sorted(agg.items()):
            sd = math.sqrt(v) if v > 0 else float("nan")
            print(f"| {ep} | {pid} | {n} | {c} | {nv} | {s:+.2f} | {sd:.2f} | {s/sd:+.2f} |")
        return
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from gen import build_ids
    from score import load_rows, row_key, response_ids, eot_token, clone_cache
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    rows = load_rows([f for f in args.files.split(",") if f], args.endpoints, [f for f in args.local.split(",") if f])
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    rows = [r for r in rows if row_key(r) not in done]
    print(f"{len(rows)} rows", flush=True)
    if not rows:
        return
    t0 = time.time()
    model, tok = load(args.model)
    hf_tok = tok._tokenizer
    eot = eot_token(hf_tok)
    eos = {eot}
    if hf_tok.eos_token_id is not None:
        eos.add(hf_tok.eos_token_id)
    for x in args.eos.split(","):
        if x:
            eos.add(int(x))
    eos_ids = sorted(eos)
    print(f"eos ids {eos_ids}", flush=True)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    n = 0
    with open(args.out, "a") as fo:
        for prompt, group in by_prompt.items():
            pids = build_ids(hf_tok, prompt, args.date)
            cache = make_prompt_cache(model)
            last = model(mx.array(pids)[None], cache=cache)[0, -1:, :]
            mx.eval(last)
            for r in group:
                rids, capped = response_ids(hf_tok, r["response"], eot)
                if len(rids) > 1:
                    c2 = clone_cache(cache)
                    rest = model(mx.array(rids[:-1])[None], cache=c2)[0]
                    logits = mx.concatenate([last, rest], axis=0)
                else:
                    logits = last
                hm = stop_probs_multi(logits, eos_ids)
                multi = {k: answer_stop_stats_viol(h, not capped) for k, h in hm.items()}
                s, v, _ = multi[f"{args.tau}|1.0"] if f"{args.tau}|1.0" in multi else (0.0, 0.0, 0)
                fo.write(json.dumps({"key": row_key(r), "endpoint": r["endpoint"], "id": r["id"], "n": len(rids), "capped": int(capped), "s": s, "v": v, "multi": multi}) + "\n")
                n += 1
    print(f"done {n} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
