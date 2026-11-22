import argparse
import json
import os
import sys
import time
from collections import defaultdict

import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from score import seen_mask, penalize

PENS = [1.0, 1.05, 1.1, 1.2, 1.3]
TEMPS = [0.6, 0.7, 0.8]


def pen_loglik(logits, prompt_ids, resp_ids, pens=PENS, temps=TEMPS):
    z = logits.astype(mx.float32)
    t = mx.array(resp_ids)[:, None]
    out = {}
    for with_prompt in (True, False):
        seen = mx.array(seen_mask(prompt_ids if with_prompt else [], resp_ids, z.shape[-1]))
        for r in pens:
            zp = penalize(z, seen, r)
            for tau in temps:
                s = zp / tau
                lp = mx.take_along_axis(s, t, axis=-1)[:, 0] - mx.logsumexp(s, axis=-1)
                v = mx.sum(lp)
                mx.eval(v)
                out[f"{r}|{tau}|{int(with_prompt)}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--ids", default="")
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
    rows = load_rows(args.files.split(","), args.endpoints, [])
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    rows = [r for r in rows if row_key(r) not in done]
    print(f"{len(rows)} rows", flush=True)
    model, tok = load(args.model)
    hf = tok._tokenizer
    eot = eot_token(hf)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    t0 = time.time()
    with open(args.out, "a") as fo:
        for prompt, group in by_prompt.items():
            pids = build_ids(hf, prompt, args.date)
            cache = make_prompt_cache(model)
            last = model(mx.array(pids)[None], cache=cache)[0, -1:, :]
            mx.eval(last)
            for r in group:
                rids, capped = response_ids(hf, r["response"], eot)
                if len(rids) > 1:
                    c2 = clone_cache(cache)
                    rest = model(mx.array(rids[:-1])[None], cache=c2)[0]
                    logits = mx.concatenate([last, rest], axis=0)
                else:
                    logits = last
                st = pen_loglik(logits, pids, rids)
                fo.write(json.dumps({"key": row_key(r), "endpoint": r["endpoint"], "id": r["id"], "n_tokens": len(rids), "ll": st}) + "\n")
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
