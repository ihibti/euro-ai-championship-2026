import argparse
import json
import math
import os
import re
from collections import defaultdict

import torch
from transformers import AutoTokenizer

ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
CAP = 256


class GreenLists:
    def __init__(self, vocab_size, gamma, hash_key):
        self.vocab_size = vocab_size
        self.size = int(vocab_size * gamma)
        self.gamma = gamma
        self.hash_key = hash_key
        self.rng = torch.Generator(device="cpu")
        self.cache = {}

    def green(self, prev):
        g = self.cache.get(prev)
        if g is None:
            self.rng.manual_seed((self.hash_key * prev) % (2 ** 64 - 1))
            perm = torch.randperm(self.vocab_size, generator=self.rng)
            g = set(perm[: self.size].tolist())
            self.cache[prev] = g
        return g


def z_score(green, total, gamma):
    if total == 0:
        return 0.0
    return (green - gamma * total) / math.sqrt(total * gamma * (1 - gamma))


def count_answer(gl, prev_last, resp_ids, unique_bigrams):
    prevs = [prev_last] + resp_ids[:-1]
    seen = set()
    g = t = 0
    for p, x in zip(prevs, resp_ids):
        if unique_bigrams:
            if (p, x) in seen:
                continue
            seen.add((p, x))
        t += 1
        g += x in gl.green(p)
    return g, t


def build_prompt_ids(tok, prompt, date):
    kw = {"date_string": date} if date else {}
    out = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, **kw)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def load_rows(files, pattern):
    rx = re.compile(pattern)
    rows = []
    for f in files:
        path = os.path.join(ANS_DIR, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if rx.search(r["endpoint"]):
                        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--files", default="round1.jsonl,night-d.jsonl,night-e.jsonl")
    ap.add_argument("--vocab_sizes", required=True)
    ap.add_argument("--gammas", default="0.25,0.5")
    ap.add_argument("--hash_key", type=int, default=15485863)
    ap.add_argument("--date", default="")
    ap.add_argument("--exclude_ids", default="d1_repeat150")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_rows(args.files.split(","), args.endpoints)
    drop = set(args.exclude_ids.split(",")) if args.exclude_ids else set()
    rows = [r for r in rows if r["id"] not in drop]
    prompt_cache = {}
    prepared = []
    for r in rows:
        if r["prompt"] not in prompt_cache:
            prompt_cache[r["prompt"]] = build_prompt_ids(tok, r["prompt"], args.date)
        resp = list(tok.encode(r["response"], add_special_tokens=False))
        if resp:
            prepared.append((r["endpoint"], prompt_cache[r["prompt"]][-1], resp))
    print(f"{len(prepared)} answers; tokenizer len {len(tok)}", flush=True)
    gammas = [float(g) for g in args.gammas.split(",")]
    eps = sorted({ep for ep, _, _ in prepared})
    occ = defaultdict(list)
    for ep, last, resp in prepared:
        prevs = [last] + resp[:-1]
        seen = set()
        for p, x in zip(prevs, resp):
            first = (p, x) not in seen
            seen.add((p, x))
            occ[p].append((ep, x, first))
    for V in [int(v) for v in args.vocab_sizes.split(",")]:
        counts = {(ep, g, u): [0, 0] for ep in eps for g in gammas for u in (False, True)}
        rng = torch.Generator(device="cpu")
        for p, lst in occ.items():
            rng.manual_seed((args.hash_key * p) % (2 ** 64 - 1))
            perm = torch.randperm(V, generator=rng)
            inv = torch.empty(V, dtype=torch.long)
            inv[perm] = torch.arange(V)
            toks = torch.tensor([min(x, V - 1) for _, x, _ in lst])
            pos = inv[toks].tolist()
            for (ep, x, first), q in zip(lst, pos):
                for g in gammas:
                    green = x < V and q < int(V * g)
                    c = counts[(ep, g, False)]
                    c[0] += green
                    c[1] += 1
                    if first:
                        c = counts[(ep, g, True)]
                        c[0] += green
                        c[1] += 1
        for g in gammas:
            for u in (False, True):
                cells = []
                for ep in eps:
                    gr, t = counts[(ep, g, u)]
                    cells.append(f"{ep}:{gr/t:.3f}(z{z_score(gr, t, g):+.1f})")
                print(f"V={V} gamma={g} unique_bigrams={u} | " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
