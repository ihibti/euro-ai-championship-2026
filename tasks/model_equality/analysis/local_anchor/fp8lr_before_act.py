import argparse
import json
import os
import sys
import time
from collections import defaultdict

import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")

mx.set_cache_limit(1 << 30)

FP8_MAX = 448.0
TAUS = [0.7, 0.72, 1.0]
SCHEMES = ["tensor_merged", "tensor_sep", "channel"]


def fp8_roundtrip(w, scale):
    x = mx.clip(w.astype(mx.float32) / scale, -FP8_MAX, FP8_MAX)
    q = mx.from_fp8(mx.to_fp8(x), dtype=mx.float32)
    return (q * scale).astype(w.dtype)


def tensor_scale(ws):
    amax = mx.max(mx.stack([mx.max(mx.abs(w.astype(mx.float32))) for w in ws]))
    return amax / FP8_MAX


def channel_scale(w):
    return mx.max(mx.abs(w.astype(mx.float32)), axis=1, keepdims=True) / FP8_MAX


def linear_groups(layer):
    a, m = layer.self_attn, layer.mlp
    return [[a.q_proj, a.k_proj, a.v_proj], [a.o_proj], [m.gate_proj, m.up_proj], [m.down_proj]]


def quantize_layers(layers, scheme):
    for layer in layers:
        groups = linear_groups(layer)
        if scheme == "tensor_sep":
            groups = [[mod] for g in groups for mod in g]
        for g in groups:
            if scheme == "channel":
                for mod in g:
                    mod.weight = fp8_roundtrip(mod.weight, channel_scale(mod.weight))
            else:
                s = tensor_scale([mod.weight for mod in g])
                for mod in g:
                    mod.weight = fp8_roundtrip(mod.weight, s)
        mx.eval([mod.weight for g in groups for mod in g])


def token_logps(logits, targets, taus=TAUS):
    z = logits.astype(mx.float32)
    t = mx.array(targets)[:, None]
    out = {}
    for tau in taus:
        s = z / tau
        lp = mx.take_along_axis(s, t, axis=-1)[:, 0] - mx.logsumexp(s, axis=-1)
        mx.eval(lp)
        out[str(tau)] = [round(x, 5) for x in lp.tolist()]
    return out


def run_pass(model_path, scheme, rows, date, out_path):
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from gen import build_ids
    from score import row_key, response_ids, eot_token, clone_cache
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if row_key(r) not in done]
    print(f"{scheme}: {len(todo)} rows", flush=True)
    if not todo:
        return
    t0 = time.time()
    model, tok = load(model_path)
    if scheme != "plain":
        quantize_layers(model.model.layers, scheme)
    hf = tok._tokenizer
    eot = eot_token(hf)
    by_prompt = defaultdict(list)
    for r in todo:
        by_prompt[r["prompt"]].append(r)
    with open(out_path, "a") as fo:
        for prompt, group in by_prompt.items():
            pids = build_ids(hf, prompt, date)
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
                fo.write(json.dumps({"key": row_key(r), "endpoint": r["endpoint"], "id": r["id"], "n_tokens": len(rids), "lp": token_logps(logits, rids)}) + "\n")
    del model
    mx.clear_cache()
    print(f"{scheme}: done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--ids", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--schemes", default="plain," + ",".join(SCHEMES))
    args = ap.parse_args()
    from score import load_rows
    rows = load_rows(args.files.split(","), args.endpoints, [])
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    for scheme in args.schemes.split(","):
        run_pass(args.model, scheme, rows, args.date, f"lr_{args.label}_{scheme}.jsonl")


if __name__ == "__main__":
    main()
