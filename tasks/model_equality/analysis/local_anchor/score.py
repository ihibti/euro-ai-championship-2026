import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")

ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
CAP = 256
CAP_MARGIN = 2


def seen_mask(prompt_ids, resp_ids, vocab):
    m = np.zeros((len(resp_ids), vocab), dtype=np.bool_)
    if prompt_ids:
        m[:, np.unique(np.array(prompt_ids))] = True
    first = {}
    for i, t in enumerate(resp_ids):
        if t not in first:
            first[t] = i
    for t, i in first.items():
        m[i + 1:, t] = True
    return m


def penalize(logits, seen, rep_pen):
    if rep_pen == 1.0:
        return logits
    pen = mx.where(logits > 0, logits / rep_pen, logits * rep_pen)
    return mx.where(seen, pen, logits)


def mass_above(logits, target_logit, temp):
    z = logits / temp
    p = mx.softmax(z, axis=-1)
    above = logits > target_logit[:, None]
    return mx.sum(mx.where(above, p, 0.0), axis=-1), p


def token_stats(logits, targets, seen, rep_pen, top_k=20):
    logits = logits.astype(mx.float32)
    t = mx.array(targets)
    tl = mx.take_along_axis(logits, t[:, None], axis=-1)[:, 0]
    lse = mx.logsumexp(logits, axis=-1)
    lp = tl - lse
    logp_all = logits - lse[:, None]
    ent = -mx.sum(mx.exp(logp_all) * logp_all, axis=-1)
    rank = mx.sum(logits > tl[:, None], axis=-1) + 1
    mx.eval(lp, ent, rank)
    ma07, _ = mass_above(logits, tl, 0.7)
    mx.eval(ma07)
    ma10, _ = mass_above(logits, tl, 1.0)
    mx.eval(ma10)
    pl = penalize(logits, mx.array(seen), rep_pen)
    ptl = mx.take_along_axis(pl, t[:, None], axis=-1)[:, 0]
    rank_pen = mx.sum(pl > ptl[:, None], axis=-1) + 1
    mx.eval(pl, ptl, rank_pen)
    out = {
        "lp": lp, "ent": ent, "rank": rank, "ma07": ma07, "ma10": ma10, "rank_pen": rank_pen,
    }
    for temp, key in ((0.7, "k07"), (1.0, "k10")):
        ma, p = mass_above(pl, ptl, temp)
        topk_mass = mx.sum(mx.topk(p, top_k, axis=-1), axis=-1)
        out["ma_pen_" + key] = ma
        out["ma_pen_k20_" + key] = mx.where(rank_pen <= top_k, ma / topk_mass, 1.0)
        mx.eval(out["ma_pen_" + key], out["ma_pen_k20_" + key])
    mx.eval(out)
    return {k: np.array(v) for k, v in out.items()}


TAUS = [round(0.4 + 0.05 * i, 3) for i in range(23)]


def tau_sums(z, targets, taus=TAUS):
    z = z.astype(mx.float32)
    zt = mx.take_along_axis(z, mx.array(targets)[:, None], axis=-1)[:, 0]
    lps, hs = [], []
    for tau in taus:
        s = z / tau
        lse = mx.logsumexp(s, axis=-1)
        p = mx.exp(s - lse[:, None])
        h = lse - mx.sum(p * s, axis=-1)
        lps.append(mx.sum(zt / tau - lse))
        hs.append(mx.sum(h))
        mx.eval(lps[-1], hs[-1])
    lps = mx.stack(lps)
    hs = mx.stack(hs)
    mx.eval(lps, hs)
    return [round(float(x), 4) for x in np.array(lps)], [round(float(x), 4) for x in np.array(hs)]


def summarize_answer(ts):
    n = len(ts["lp"])
    if n == 0:
        return {"n_tokens": 0}
    excess = -ts["lp"] - ts["ent"]
    return {
        "n_tokens": n,
        "mean_lp": float(np.mean(ts["lp"])),
        "mean_excess": float(np.mean(excess)),
        "sum_excess": float(np.sum(excess)),
        "mean_ent": float(np.mean(ts["ent"])),
        "share_rank_gt20": float(np.mean(ts["rank"] > 20)),
        "share_out_p08_t07": float(np.mean(ts["ma07"] >= 0.8)),
        "share_out_p08_t10": float(np.mean(ts["ma10"] >= 0.8)),
        "worst_rank": int(np.max(ts["rank"])),
        "max_ma07": float(np.max(ts["ma07"])),
        "max_ma10": float(np.max(ts["ma10"])),
        "share_rankpen_gt20": float(np.mean(ts["rank_pen"] > 20)),
        "worst_rank_pen": int(np.max(ts["rank_pen"])),
        "share_out_gc_t07": float(np.mean(ts["ma_pen_k20_k07"] >= 0.8)),
        "share_out_gc_t10": float(np.mean(ts["ma_pen_k20_k10"] >= 0.8)),
        "n_out_gc_t07": int(np.sum(ts["ma_pen_k20_k07"] >= 0.8)),
        "n_out_gc_t10": int(np.sum(ts["ma_pen_k20_k10"] >= 0.8)),
        "max_ma_pen_k20_t07": float(np.max(ts["ma_pen_k20_k07"])),
        "max_ma_pen_k20_t10": float(np.max(ts["ma_pen_k20_k10"])),
    }


def response_ids(hf_tok, response, eot_id):
    ids = list(hf_tok.encode(response, add_special_tokens=False))
    capped = len(ids) >= CAP - CAP_MARGIN
    if not capped:
        ids = ids + [eot_id]
    return ids, capped


def eot_token(hf_tok):
    for name in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        tid = hf_tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != hf_tok.unk_token_id and tid is not None:
            return tid
    return hf_tok.eos_token_id


def load_rows(files, pattern, local_files):
    rx = re.compile(pattern)
    rows = []
    for f in files:
        path = os.path.join(ANS_DIR, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if rx.search(r["endpoint"]):
                    r["_src"] = f
                    rows.append(r)
    for f in local_files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if rx.search(r["endpoint"]):
                        r["_src"] = os.path.basename(f)
                        rows.append(r)
    return rows


def row_key(r):
    return f'{r["_src"]}|{r["endpoint"]}|{r["id"]}|{r["repeat"]}'


def clone_cache(cache):
    from mlx_lm.models.cache import KVCache
    out = []
    for c in cache:
        n = KVCache()
        n.keys = c.keys[..., : c.offset, :]
        n.values = c.values[..., : c.offset, :]
        n.offset = c.offset
        out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--local", default="")
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--rep_pen", type=float, default=1.0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--tau_pen", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from gen import build_ids

    out_path = args.out or f"scores_{args.label}.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    rows = load_rows([f for f in args.files.split(",") if f], args.endpoints,
                     [f for f in args.local.split(",") if f])
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    rows = [r for r in rows if row_key(r) not in done]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"{len(rows)} rows to score, {len(done)} already done", flush=True)
    if not rows:
        return
    t0 = time.time()
    model, tok = load(args.model)
    hf_tok = tok._tokenizer
    eot = eot_token(hf_tok)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    n_done = 0
    with open(out_path, "a") as fo:
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
                seen = seen_mask(pids, rids, logits.shape[-1])
                ts = token_stats(logits, rids, seen, args.rep_pen)
                s = summarize_answer(ts)
                s["tau_lp_raw"], s["tau_h_raw"] = tau_sums(logits, rids)
                if args.tau_pen and args.rep_pen != 1.0:
                    s["tau_lp_pen"], s["tau_h_pen"] = tau_sums(penalize(logits.astype(mx.float32), mx.array(seen), args.rep_pen), rids)
                rec = {"key": row_key(r), "model": args.label, "endpoint": r["endpoint"], "id": r["id"],
                       "repeat": r["repeat"], "src": r["_src"], "capped": capped}
                rec.update(s)
                fo.write(json.dumps(rec) + "\n")
                n_done += 1
                if n_done % 50 == 0:
                    fo.flush()
                    print(f"{n_done}/{len(rows)} {time.time()-t0:.0f}s peak={mx.get_peak_memory()/1e9:.2f}GB", flush=True)
    print(f"done {n_done} in {time.time()-t0:.0f}s peak={mx.get_peak_memory()/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
