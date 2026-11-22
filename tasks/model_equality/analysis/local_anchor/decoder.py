import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")

TEMPS = [round(0.6 + 0.05 * i, 2) for i in range(13)]
TOP_PS = [1.0, 0.99, 0.95, 0.9, 0.8]
TOP_KS = [0, 50, 20]
KMAX = 2048


def decoder_stats(z, targets, temps=TEMPS, top_ps=TOP_PS, top_ks=TOP_KS, kmax=KMAX):
    z = z.astype(mx.float32)
    T, V = z.shape
    kmax = min(kmax, V)
    t = mx.array(targets)
    zx = mx.take_along_axis(z, t[:, None], axis=-1)[:, 0]
    rank0 = mx.sum(z > zx[:, None], axis=-1)
    part = mx.argpartition(-z, kth=kmax - 1, axis=-1)[:, :kmax]
    vals = mx.take_along_axis(z, part, axis=-1)
    vals = -mx.sort(-vals, axis=-1)
    mx.eval(zx, rank0, vals)
    out = {}
    pos = mx.arange(kmax)[None, :]
    for tau in temps:
        lse = mx.logsumexp(z / tau, axis=-1)
        p = mx.exp(vals / tau - lse[:, None])
        cum = mx.cumsum(p, axis=-1)
        lpx = zx / tau - lse
        r = mx.minimum(rank0, kmax)
        above = mx.where(r > 0, mx.take_along_axis(cum, mx.maximum(r - 1, 0)[:, None], axis=-1)[:, 0], 0.0)
        mx.eval(lse, cum, lpx, above)
        for k in top_ks:
            if k:
                ck = cum[:, k - 1]
                inside_k = rank0 < k
                above_k = above / ck
                cumk = cum[:, :k] / ck[:, None]
                base_lp = lpx - mx.log(ck)
            else:
                ck = None
                inside_k = mx.ones((T,), dtype=mx.bool_)
                above_k = above
                cumk = cum
                base_lp = lpx
            for tp in top_ps:
                if tp >= 1.0:
                    viol = mx.logical_not(inside_k)
                    lp = base_lp
                else:
                    c = mx.sum(cumk < tp, axis=-1)
                    c = mx.minimum(c, cumk.shape[-1] - 1)
                    kept = mx.take_along_axis(cumk, c[:, None], axis=-1)[:, 0]
                    viol = mx.logical_or(mx.logical_not(inside_k), above_k >= tp)
                    lp = base_lp - mx.log(kept)
                ok = mx.logical_not(viol)
                s = mx.sum(mx.where(ok, lp, 0.0))
                nv = mx.sum(viol)
                mx.eval(s, nv)
                out[f"{tau}|{tp}|{k}"] = (float(s), int(nv.item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--local", default="")
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--exclude_ids", default="d1_repeat150")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
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
    drop = set(args.exclude_ids.split(",")) if args.exclude_ids else set()
    rows = [r for r in rows if r["id"] not in drop and row_key(r) not in done]
    print(f"{len(rows)} rows", flush=True)
    if not rows:
        return
    t0 = time.time()
    model, tok = load(args.model)
    hf_tok = tok._tokenizer
    eot = eot_token(hf_tok)
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
                st = decoder_stats(logits, rids)
                fo.write(json.dumps({"key": row_key(r), "endpoint": r["endpoint"], "id": r["id"], "n_tokens": len(rids), "dec": st}) + "\n")
                n += 1
                if n % 50 == 0:
                    fo.flush()
                    print(f"{n}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {n} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
