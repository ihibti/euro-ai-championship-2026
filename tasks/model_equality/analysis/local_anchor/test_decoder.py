import math
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from decoder import decoder_stats


def brute(z, x, tau, tp, k):
    order = np.argsort(-z)
    keep = list(order[:k]) if k else list(order)
    lg = z[keep] / tau
    p = np.exp(lg - lg.max())
    p /= p.sum()
    cum = 0.0
    kept = []
    for i, pi in zip(keep, p):
        if tp < 1.0 and cum >= tp:
            break
        kept.append(i)
        cum += pi
    if x not in kept:
        return None
    lk = z[kept] / tau
    m = lk.max()
    return z[x] / tau - (m + math.log(np.exp(lk - m).sum()))


def test_against_brute_force():
    rng = np.random.default_rng(0)
    T, V = 40, 300
    z = (rng.normal(size=(T, V)) * 3).astype(np.float32)
    x = [int(rng.integers(0, 30)) for _ in range(T)]
    st = decoder_stats(mx.array(z), x, temps=[0.7, 1.0], top_ps=[1.0, 0.95, 0.8], top_ks=[0, 20], kmax=256)
    for tau in (0.7, 1.0):
        for tp in (1.0, 0.95, 0.8):
            for k in (0, 20):
                s = 0.0
                nv = 0
                for t in range(T):
                    v = brute(z[t], x[t], tau, tp, k)
                    if v is None:
                        nv += 1
                    else:
                        s += v
                got_s, got_nv = st[f"{tau}|{tp}|{k}"]
                assert got_nv == nv, (tau, tp, k, got_nv, nv)
                assert abs(got_s - s) < 1e-2 * max(1, abs(s)), (tau, tp, k, got_s, s)


if __name__ == "__main__":
    test_against_brute_force()
    print("ok test_against_brute_force")
