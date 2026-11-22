import sys

sys.path.insert(0, "SCRATCHPAD/local_anchor")
import mlx.core as mx
import mlx.nn as nn
from fp8lr import fp8_roundtrip, tensor_scale, channel_scale, quantize_layers, FP8_MAX, act_roundtrip, ActQuant, wrap_activations, ACT_MODE
from lrsum import lr_table_rows, in_half


def test_roundtrip_keeps_grid_values_and_maps_amax():
    s = 0.01
    grid = mx.array([0.0, 1.0, 1.125, -2.5, 448.0, -448.0, 2 ** -9]) * s
    back = fp8_roundtrip(grid.astype(mx.float32), mx.array(s))
    assert mx.allclose(back, grid, rtol=1e-6, atol=0).item()


def test_roundtrip_relative_error_bound():
    w = mx.random.normal((64, 64), key=mx.random.key(1)).astype(mx.float32)
    s = tensor_scale([w])
    q = fp8_roundtrip(w, s)
    big = mx.abs(w) > s * 2 ** -6
    rel = mx.where(big, mx.abs(q - w) / mx.abs(w), 0.0)
    assert mx.max(rel).item() <= 2 ** -4 + 1e-6
    assert abs(mx.max(mx.abs(q)).item() - mx.max(mx.abs(w)).item()) < 1e-6


def test_merged_scale_uses_group_max():
    a = mx.ones((2, 2)) * 1.0
    b = mx.ones((2, 2)) * 4.48
    assert abs(tensor_scale([a, b]).item() - 4.48 / FP8_MAX) < 1e-9
    c = mx.array([[1.0, -2.0], [0.5, 0.25]])
    assert mx.allclose(channel_scale(c), mx.array([[2.0], [0.5]]) / FP8_MAX).item()


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = nn.Linear(4, 4), nn.Linear(4, 2), nn.Linear(4, 2), nn.Linear(4, 4)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 8), nn.Linear(4, 8), nn.Linear(8, 4)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = _Attn(), _Mlp()


def test_quantize_layers_merged_shares_qkv_scale():
    mx.random.seed(3)
    layer = _Layer()
    layer.self_attn.k_proj.weight = layer.self_attn.k_proj.weight * 100
    q_before = layer.self_attn.q_proj.weight
    s = tensor_scale([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight])
    quantize_layers([layer], "tensor_merged")
    expect = fp8_roundtrip(q_before, s)
    assert mx.array_equal(layer.self_attn.q_proj.weight, expect).item()
    b_before = layer.self_attn.q_proj.bias
    assert b_before is not None


def test_lr_table_sign():
    plain = {"k1": {"endpoint": "e", "id": "p", "lp": {"0.7": [-1.0, -2.0]}}}
    fp8 = {"k1": {"endpoint": "e", "id": "p", "lp": {"0.7": [-0.5, -2.5]}}}
    rows = lr_table_rows(plain, fp8, "0.7")
    assert rows["e"]["n_tok"] == 2 and abs(rows["e"]["lr"] - 0.0) < 1e-9
    fp8["k1"]["lp"]["0.7"] = [-0.5, -1.5]
    rows = lr_table_rows(plain, fp8, "0.7")
    assert abs(rows["e"]["lr"] - 1.0) < 1e-9


def test_in_half():
    assert in_half("round1.jsonl|ref-01|p|4", "even") and not in_half("round1.jsonl|ref-01|p|4", "odd") and in_half("a|b|c|7", "odd") and in_half("a|b|c|7", "all")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_act_roundtrip_token_mode_scales_each_position_alone():
    x = mx.array([[[448.0, 1.0625, 0.0], [4.48, 0.010625, 0.0]]])
    tok = act_roundtrip(x, "token")
    ten = act_roundtrip(x, "tensor")
    assert abs(tok[0, 0, 1].item() - 1.0) < 1e-6
    assert abs(tok[0, 1, 1].item() - 0.01) < 1e-7
    assert abs(ten[0, 1, 1].item() - 0.01171875) < 1e-7 or abs(ten[0, 1, 1].item() - 0.009765625) < 1e-7
    assert abs(tok[0, 0, 0].item() - 448.0) < 1e-4 and abs(tok[0, 1, 0].item() - 4.48) < 1e-6


def test_wrap_activations_quantizes_the_input_of_every_linear():
    layer = _Layer()
    wrap_activations([layer])
    assert isinstance(layer.self_attn.q_proj, ActQuant) and isinstance(layer.mlp.down_proj, ActQuant)
    x = mx.array([[[448.0, 1.0625, 3.0, -7.0]]])
    ACT_MODE[0] = "token"
    got = layer.self_attn.q_proj(x)
    want = layer.self_attn.q_proj.inner(mx.array([[[448.0, 1.0, 3.0, -7.0]]]))
    assert mx.allclose(got, want, rtol=1e-6, atol=1e-6).item()
