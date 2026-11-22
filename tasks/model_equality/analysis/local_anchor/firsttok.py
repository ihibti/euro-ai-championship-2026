import argparse
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from gen import build_ids, load_prompts


def topk_after_prefix(model, hf_tok, ids, prefix_ids, k):
    cache = make_prompt_cache(model)
    seq = ids + prefix_ids
    logits = model(mx.array(seq)[None], cache=cache)[0, -1, :].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    order = mx.argsort(-lp)[:k].tolist()
    vals = lp[mx.array(order)].tolist()
    return [(hf_tok.decode([t]), round(float(mx.exp(mx.array(v)).item()), 4)) for t, v in zip(order, vals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--probes", default="e2_leak_quote,e1_leak_repeat")
    ap.add_argument("--prefixes", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="firsttok.jsonl")
    args = ap.parse_args()
    model, tok = load(args.model)
    hf_tok = tok._tokenizer
    prompts = load_prompts()
    prefixes = [""] + [p for p in args.prefixes.split("|") if p]
    with open(args.out, "a") as f:
        for pid in args.probes.split(","):
            ids = build_ids(hf_tok, prompts[pid], args.date)
            for pre in prefixes:
                pre_ids = hf_tok.encode(pre, add_special_tokens=False) if pre else []
                top = topk_after_prefix(model, hf_tok, ids, pre_ids, args.k)
                rec = {"label": args.label, "id": pid, "prefix": pre, "top": top}
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
