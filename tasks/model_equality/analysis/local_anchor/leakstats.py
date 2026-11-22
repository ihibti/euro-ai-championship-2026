import argparse
import json
import math
from collections import defaultdict

ANS_DIR = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
LLAMA_HEADER = "Cutting Knowledge Date"


def classify_e2(resp, default_marker):
    s = resp.strip()
    first = s.split("\n", 1)[0].strip()
    low = first.lower().strip(".*:! ")
    yes = low.startswith("yes")
    no = low.startswith("no")
    rest = s.split("\n", 1)[1] if "\n" in s else s[len(first):]
    quotes_default = default_marker.lower() in s.lower() if default_marker else False
    invents = yes and not quotes_default and len(rest.strip()) > 0
    return {
        "yes": yes,
        "no": no,
        "invents": invents,
        "quotes_default": quotes_default,
        "lower_first": first[:1].islower() if first else False,
        "bare": s.lower().strip(".! ") in ("yes", "no"),
    }


def classify_e1(resp, prompt, default_marker):
    s = resp.strip()
    quotes_default = default_marker.lower() in s.lower() if default_marker else False
    echo = prompt.strip()[:40].lower() in s.lower()
    header = LLAMA_HEADER.lower() in s.lower()
    exact = s == default_marker
    return {"quotes_default": quotes_default, "exact_default": exact, "echo": echo, "header": header,
            "other": not (quotes_default or echo or header)}


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def stats(rows, endpoints, marker):
    out = {}
    for ep in endpoints:
        e1 = [r for r in rows if r["endpoint"] == ep and r["id"] == "e1_leak_repeat"]
        e2 = [r for r in rows if r["endpoint"] == ep and r["id"] == "e2_leak_quote"]
        c1 = [classify_e1(r["response"], r["prompt"], marker) for r in e1]
        c2 = [classify_e2(r["response"], marker) for r in e2]
        d = {"n_e1": len(c1), "n_e2": len(c2)}
        for k in ("quotes_default", "exact_default", "echo", "header", "other"):
            d["e1_" + k] = sum(c[k] for c in c1)
        for k in ("yes", "no", "invents", "quotes_default", "lower_first", "bare"):
            d["e2_" + k] = sum(c[k] for c in c2)
        out[ep] = d
    return out


def pooled(st, eps, name):
    d = defaultdict(int)
    for e in eps:
        for k, v in st[e].items():
            d[k] += v
    st[name] = dict(d)


def fmt_rate(k, n):
    if n == 0:
        return "-"
    p = k / n
    se = math.sqrt(max(p * (1 - p), 0.25 / n) / n)
    return f"{k}/{n}={p:.2f}±{se:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="A")
    ap.add_argument("--local", default="answers.jsonl")
    args = ap.parse_args()
    rows = load_rows([ANS_DIR + "/night-e.jsonl", args.local])
    fam = {
        "A": "01 02 04 12 14 17 18 23 24 25 33",
        "B": "03 06 07 08 20 21 22 26 28 30 32",
        "C": "05 09 10 11 13 15 16 19 27 29 31",
    }[args.family]
    eps = ["ep-" + e for e in fam.split()]
    marker = QWEN_DEFAULT if args.family in "AB" else ""
    locals_ = sorted({r["endpoint"] for r in rows if r["endpoint"].startswith("local-")})
    st = stats(rows, eps + locals_, marker)
    if args.family == "A":
        pooled(st, ["ep-01", "ep-17", "ep-24", "ep-25", "ep-33"], "groupY")
        pooled(st, ["ep-04", "ep-12", "ep-14", "ep-18"], "groupN")
    cols = ["e2_yes", "e2_invents", "e2_quotes_default", "e2_lower_first", "e2_bare"]
    cols1 = ["e1_quotes_default", "e1_exact_default", "e1_echo", "e1_header", "e1_other"]
    print("name | " + " | ".join(cols) + " | " + " | ".join(cols1))
    for name in list(st):
        d = st[name]
        print(name, "|", " | ".join(fmt_rate(d[c], d["n_e2"]) for c in cols), "|",
              " | ".join(fmt_rate(d[c], d["n_e1"]) for c in cols1))


if __name__ == "__main__":
    main()
