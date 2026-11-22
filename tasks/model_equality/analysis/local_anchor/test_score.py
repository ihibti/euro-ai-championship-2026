import math
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from score import seen_mask, token_stats, summarize_answer, tau_sums


def test_seen_mask():
    m = seen_mask([1, 2], [3, 1, 3, 4], 6)
    assert m[0].tolist() == [False, True, True, False, False, False]
    assert m[1].tolist() == [False, True, True, True, False, False]
    assert m[3][3] and not m[3][4]


def test_token_stats_toy():
    p = np.array([[0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]])
    logits = mx.array(np.log(p).astype(np.float32))
    seen = np.zeros((2, 4), dtype=np.bool_)
    ts = token_stats(logits, [2, 0], seen, 1.0, top_k=2)
    assert abs(ts["lp"][0] - math.log(0.15)) < 1e-5
    assert ts["rank"][0] == 3 and ts["rank"][1] == 1
    assert abs(ts["ma10"][0] - 0.8) < 1e-5
    assert abs(ts["ent"][1] - math.log(4)) < 1e-5
    assert ts["ma_pen_k20_k07"][0] == 1.0
    s = summarize_answer(ts)
    assert s["n_tokens"] == 2 and s["worst_rank"] == 3
    assert abs(s["mean_excess"] - ((-math.log(0.15) - ts["ent"][0]) + 0.0) / 2) < 1e-4


def test_rep_penalty_changes_rank():
    logits = mx.array([[2.0, 1.9, 0.0]])
    seen = np.array([[True, False, False]])
    ts = token_stats(logits, [1], seen, 1.2, top_k=2)
    assert ts["rank"][0] == 2 and ts["rank_pen"][0] == 1


def test_tau_sums_recover_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 30)).astype(np.float32) * 2.0
    true_tau = 0.7
    p = np.exp(z / true_tau)
    p /= p.sum(axis=1, keepdims=True)
    x = [int(rng.choice(30, p=row)) for row in p]
    taus = [0.5, 0.6, 0.7, 0.8, 0.9]
    lp, h = tau_sums(mx.array(z), x, taus)
    best = taus[int(np.argmax(lp))]
    assert best == 0.7, (best, lp)
    resid = -(lp[2] + h[2]) / len(x)
    assert abs(resid) < 0.05, resid
    resid_wrong = -(lp[4] + h[4]) / len(x)
    assert resid_wrong < resid


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
