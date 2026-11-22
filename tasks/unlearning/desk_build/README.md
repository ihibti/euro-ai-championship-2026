# How the desk built `k11_b_soup3_x2.pt` (public 0.2858)

- `soup.py`: average of state dicts in float64, refusing mismatched keys, shapes or dtypes and differing integer tensors; `test_soup.py` holds its four tests.
- `splice_head_latest.py`: the unlearning agent's last-layer scaling, as used with `--scale 0.5`.
- `rebuild_original.py`: rebuilds the original checkpoint from the organizers' files for comparison.

The file is the average of `k8_e3t2_rm1_lr1e4`, `k11_b_s1` and `k11_b_s2` with the last layer times 2; sha256 `53dfdc9d0067c1c727cce13a659a76399bca6d856dde269fae29b488da4aaf9e`.
