import sys

import mlx.core as mx

sys.path.insert(0, "SCRATCHPAD/local_anchor")
from gen import sample_step


def freq(logits, seen, temp, top_p, top_k, rep, n=4000):
    b = mx.repeat(logits[None], n, axis=0)
    s = mx.repeat(seen[None], n, axis=0)
    t = sample_step(b, s, temp, top_p, top_k, rep).tolist()
    v = logits.shape[-1]
    return [t.count(i) / n for i in range(v)]


def test_top_k_restricts_support():
    mx.random.seed(0)
    lg = mx.array([3.0, 2.9, 2.8, 0.0, -1.0, 2.7])
    f = freq(lg, mx.zeros((6,), dtype=mx.bool_), 1.0, 1.0, 2, 1.0)
    assert f[2] == 0 and f[3] == 0 and f[4] == 0 and f[5] == 0
    assert abs(f[0] - 0.525) < 0.04


def test_top_p_keeps_smallest_nucleus():
    mx.random.seed(1)
    p = mx.array([0.5, 0.3, 0.15, 0.05])
    f = freq(mx.log(p), mx.zeros((4,), dtype=mx.bool_), 1.0, 0.79, 0, 1.0)
    assert f[2] == 0 and f[3] == 0
    assert abs(f[0] - 0.625) < 0.04
    f = freq(mx.log(p), mx.zeros((4,), dtype=mx.bool_), 1.0, 0.81, 0, 1.0)
    assert f[2] > 0 and f[3] == 0


def test_temperature_before_top_p():
    mx.random.seed(2)
    p = mx.array([0.5, 0.3, 0.15, 0.05])
    f = freq(mx.log(p), mx.zeros((4,), dtype=mx.bool_), 0.5, 0.79, 0, 1.0)
    assert f[2] == 0 and f[3] == 0
    q0 = 0.25 / (0.25 + 0.09)
    assert abs(f[0] - q0) < 0.04


def test_repetition_penalty_sign():
    mx.random.seed(3)
    lg = mx.array([1.0, 1.0, -1.0, -1.0])
    seen = mx.array([True, False, True, False])
    f = freq(lg, seen, 1.0, 1.0, 0, 2.0)
    assert f[0] < f[1]
    assert f[2] < f[3]


def expected(lg, temp, top_p, top_k):
    import math
    v = [x / temp for x in lg]
    order = sorted(range(len(v)), key=lambda i: -v[i])
    if top_k > 0:
        order = order[:top_k]
    m = max(v[i] for i in order)
    w = [math.exp(v[i] - m) for i in order]
    z = sum(w)
    keep, cum = [], 0.0
    for i, wi in zip(order, w):
        if cum >= top_p:
            break
        keep.append(i)
        cum += wi / z
    zk = sum(math.exp(v[i] - m) for i in keep)
    out = [0.0] * len(v)
    for i in keep:
        out[i] = math.exp(v[i] - m) / zk
    return out


def test_partition_path_matches_analytic():
    mx.random.seed(4)
    lg = mx.random.normal((50,)) * 2
    seen = mx.zeros((50,), dtype=mx.bool_)
    for top_p in (0.8, 0.99):
        got = freq(lg, seen, 0.7, top_p, 20, 1.0, n=20000)
        exp = expected(lg.tolist(), 0.7, top_p, 20)
        assert sum(abs(x - y) for x, y in zip(got, exp)) < 0.05, (top_p, got, exp)
        got = freq(lg, seen, 0.7, top_p, 0, 1.0, n=20000)
        exp = expected(lg.tolist(), 0.7, top_p, 0)
        assert sum(abs(x - y) for x, y in zip(got, exp)) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
