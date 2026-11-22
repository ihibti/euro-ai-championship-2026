import argparse
import csv
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROBE_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/probes"
PROBE_FILES = ["leak.csv", "d1.csv", "bat.csv", "idn.csv", "recon.csv"]


def load_prompts():
    prompts = {}
    for name in PROBE_FILES:
        path = os.path.join(PROBE_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                prompts[row["id"]] = row["prompt"]
    return prompts


ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"


def load_prompts_from_answers(files=("round1.jsonl", "night-d.jsonl", "night-e.jsonl")):
    from collections import Counter, defaultdict
    seen = defaultdict(Counter)
    for name in files:
        path = os.path.join(ANS_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["endpoint"].startswith("ep-"):
                    seen[r["id"]][r["prompt"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in seen.items()}


def existing_count(out_path, endpoint, pid):
    if not os.path.exists(out_path):
        return 0
    n = 0
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("endpoint") == endpoint and r.get("id") == pid:
                n += 1
    return n


def sample_step(logits, seen, temp, top_p, top_k, rep_pen):
    logits = logits.astype(mx.float32)
    if rep_pen != 1.0:
        pen = mx.where(logits > 0, logits / rep_pen, logits * rep_pen)
        logits = mx.where(seen, pen, logits)
    if temp <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temp
    if top_k <= 0 and top_p >= 1.0:
        return mx.random.categorical(logits, axis=-1)
    if 0 < top_k < logits.shape[-1]:
        idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[:, :top_k]
        vals = mx.take_along_axis(logits, idx, axis=-1)
        local = mx.argsort(-vals, axis=-1)
        order = mx.take_along_axis(idx, local, axis=-1)
        sorted_logits = mx.take_along_axis(vals, local, axis=-1)
    else:
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
    pos = mx.arange(sorted_logits.shape[-1])[None, :]
    if 0 < top_p < 1.0:
        probs = mx.softmax(sorted_logits, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        drop = (cum - probs) >= top_p
        drop = mx.logical_and(drop, pos > 0)
        sorted_logits = mx.where(drop, -mx.inf, sorted_logits)
    choice = mx.random.categorical(sorted_logits, axis=-1)
    return mx.take_along_axis(order, choice[:, None], axis=-1)[:, 0]


def build_ids(hf_tok, prompt, date_string):
    kwargs = {}
    if date_string:
        kwargs["date_string"] = date_string
    out = hf_tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, **kwargs
    )
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def generate_batch(model, ids, batch, eos, temp, top_p, top_k, rep_pen, max_tokens, vocab):
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    for c in cache:
        c.keys = mx.repeat(c.keys, batch, axis=0)
        c.values = mx.repeat(c.values, batch, axis=0)
    logits = mx.repeat(logits, batch, axis=0)
    seen = mx.zeros((batch, vocab), dtype=mx.bool_)
    prompt_idx = mx.array(sorted(set(ids)))
    seen[:, prompt_idx] = True
    outs = [[] for _ in range(batch)]
    done = [False] * batch
    rows = mx.arange(batch)
    for step in range(max_tokens):
        tok = sample_step(logits, seen, temp, top_p, top_k, rep_pen)
        mx.eval(tok)
        tl = tok.tolist()
        for b, t in enumerate(tl):
            if done[b]:
                continue
            if t in eos:
                done[b] = True
            else:
                outs[b].append(t)
        if all(done):
            break
        seen[rows, tok] = True
        logits = model(tok[:, None], cache=cache)[:, -1, :]
    finish = ["stop" if d else "length" for d in done]
    return outs, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--genconf", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--rep_pen", type=float, default=None)
    ap.add_argument("--date", default="")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="answers.jsonl")
    ap.add_argument("--meta", default="meta.jsonl")
    args = ap.parse_args()

    gc = json.load(open(args.genconf))
    temp = args.temp if args.temp is not None else gc.get("temperature", 1.0)
    top_p = args.top_p if args.top_p is not None else gc.get("top_p", 1.0)
    top_k = args.top_k if args.top_k is not None else gc.get("top_k", 0)
    rep_pen = args.rep_pen if args.rep_pen is not None else gc.get("repetition_penalty", 1.0)
    eos = gc.get("eos_token_id", [])
    eos = set(eos if isinstance(eos, list) else [eos])

    mx.random.seed(args.seed)
    t0 = time.time()
    model, tok = load(args.model)
    hf_tok = tok._tokenizer
    if hf_tok.eos_token_id is not None:
        eos.add(hf_tok.eos_token_id)
    vocab = model.args.vocab_size
    print(f"loaded {args.model} in {time.time()-t0:.1f}s temp={temp} top_p={top_p} top_k={top_k} rp={rep_pen} eos={sorted(eos)}", flush=True)

    prompts = load_prompts()
    prompts.update(load_prompts_from_answers())
    for pid in args.probes.split(","):
        prompt = prompts[pid]
        ids = build_ids(hf_tok, prompt, args.date)
        start = existing_count(args.out, args.label, pid)
        remaining = args.n - start
        rep = start
        while remaining > 0:
            b = min(args.batch, remaining)
            t1 = time.time()
            outs, finish = generate_batch(model, ids, b, eos, temp, top_p, top_k, rep_pen, args.max_tokens, vocab)
            ntok = sum(len(o) for o in outs)
            with open(args.out, "a") as f, open(args.meta, "a") as fm:
                for o, fin in zip(outs, finish):
                    text = hf_tok.decode(o, skip_special_tokens=True)
                    f.write(json.dumps({"endpoint": args.label, "id": pid, "repeat": rep, "prompt": prompt, "response": text}) + "\n")
                    fm.write(json.dumps({"endpoint": args.label, "id": pid, "repeat": rep, "n_tokens": len(o), "finish": fin, "temp": temp, "top_p": top_p, "top_k": top_k, "rep_pen": rep_pen, "model": args.model}) + "\n")
                    rep += 1
            remaining -= b
            print(f"{args.label} {pid} +{b} ({rep}/{args.n}) {ntok} tok {time.time()-t1:.1f}s peak={mx.get_peak_memory()/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
