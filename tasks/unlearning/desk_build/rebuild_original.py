import sys

import torch

t4_path, t3_path, out_path = sys.argv[1:4]
t4 = torch.load(t4_path, map_location="cpu", weights_only=True)
t3 = torch.load(t3_path, map_location="cpu", weights_only=True)
assert list(t4) == list(t3)
head = {"head.weight", "head.bias"}
same = all(torch.equal(t4[key], t3[key]) for key in t4 if key not in head)
original = {key: value.clone() for key, value in t4.items()}
worst = 0.0
for key in head:
    original[key] = t4[key] * 4.0
    assert torch.equal(original[key] / 4.0, t4[key])
    scale = original[key].abs().clamp_min(1e-12)
    worst = max(worst, ((t3[key] * 3.0 - original[key]).abs() / scale).max().item())
print(f"tensors: {len(original)}; body identical in both files: {same}; head from t3 against head from t4, largest relative gap: {worst:.2e}")
assert same and worst < 1e-6
torch.save(original, out_path)
