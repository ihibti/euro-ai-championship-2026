import sys

sys.path.insert(0, "SCRATCHPAD/local_anchor")
import mlx.core as mx
from stop import stop_probs, answer_stop_stats, stop_probs_multi, answer_stop_stats_viol


def test_stop_probs_and_stats():
    z = mx.log(mx.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
    h = stop_probs(z, [2], 1.0)
    assert abs(h[0] - 0.25) < 1e-5 and abs(h[1] - 0.8) < 1e-5
    s, v = answer_stop_stats(h, True)
    assert abs(s - (1 - 0.25 - 0.8)) < 1e-5
    assert abs(v - (0.25 * 0.75 + 0.8 * 0.2)) < 1e-5
    s, v = answer_stop_stats(h, False)
    assert abs(s + 1.05) < 1e-5


def brute_h(row, eos_ids, tau, tp):
    import math
    w = [math.exp(x / tau) for x in row]
    tot = sum(w)
    p = [x / tot for x in w]
    order = sorted(range(len(p)), key=lambda i: -p[i])
    kept, cum = [], 0.0
    for i in order:
        if cum < tp:
            kept.append(i)
        cum += p[i]
    km = sum(p[i] for i in kept)
    return sum(p[e] / km for e in eos_ids if e in kept)


def test_stop_probs_multi_matches_brute_force():
    import random
    random.seed(0)
    rows = [[random.gauss(0, 2) for _ in range(40)] for _ in range(6)]
    rows[0][3] = 9.0
    rows[1][3] = -9.0
    z = mx.array(rows)
    eos = [3, 7]
    settings = [(0.7, 1.0), (1.0, 0.95), (1.05, 0.5), (0.85, 0.95)]
    hm = stop_probs_multi(z, eos, settings, ksort=40)
    for tau, tp in settings:
        for i, row in enumerate(rows):
            b = brute_h(row, eos, tau, tp)
            got = hm[f"{tau}|{tp}"][i]
            assert abs(got - b) < 1e-4, (tau, tp, i, got, b)


def test_violation_when_stop_outside_nucleus():
    s, v, nv = answer_stop_stats_viol([0.1, 0.0], True)
    assert nv == 1 and abs(s + 0.1) < 1e-9
    s, v, nv = answer_stop_stats_viol([0.1, 0.4], True)
    assert nv == 0 and abs(s - 0.5) < 1e-9


if __name__ == "__main__":
    test_stop_probs_and_stats()
    test_stop_probs_multi_matches_brute_force()
    test_violation_when_stop_outside_nucleus()
    print("ok stop tests")
