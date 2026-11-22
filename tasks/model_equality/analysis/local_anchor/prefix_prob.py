import argparse
import json
import math
import sys

import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from decoder import decoder_stats

SETTINGS = [(0.7, 1.0), (1.0, 1.0), (1.0, 0.95)]


def path_probs(model, prompt_ids, prefix_ids):
    seq = prompt_ids + prefix_ids
    logits = model(mx.array(seq[:-1])[None])[0, len(prompt_ids) - 1:, :]
    st = decoder_stats(logits, prefix_ids, temps=[0.7, 1.0], top_ps=[1.0, 0.95], top_ks=[0])
    out = {}
    for tau, tp in SETTINGS:
        s, nv = st[f"{tau}|{tp}|0"]
        out[f"T{tau}/p{tp}"] = 0.0 if nv else math.exp(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--probe", default="txt_story")
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--out", default="prefix_probs.jsonl")
    args = ap.parse_args()
    from mlx_lm import load
    from gen import build_ids, load_prompts, load_prompts_from_answers
    model, tok = load(args.model)
    hf = tok._tokenizer
    prompts = load_prompts()
    prompts.update(load_prompts_from_answers())
    pids = build_ids(hf, prompts[args.probe], args.date)
    logits = model(mx.array(pids)[None])[0, -1, :].astype(mx.float32)
    p07 = mx.softmax(logits / 0.7)
    top = mx.argsort(-p07)[:8].tolist()
    print(args.label, "first tokens at T0.7:", [(hf.decode([t]), round(p07[t].item(), 4)) for t in top], flush=True)
    with open(args.out, "a") as f:
        for pre in args.prefixes.split("|"):
            ids = list(hf.encode(pre, add_special_tokens=False))
            pr = path_probs(model, pids, ids)
            rec = {"label": args.label, "probe": args.probe, "prefix": pre, "n_tokens": len(ids), "p": pr}
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
