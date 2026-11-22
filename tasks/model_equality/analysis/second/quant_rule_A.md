# Family A: other quantization schemes, rule registered before any reading

Written 2026-11-22 08:31 Paris, before any number below was computed for family A. Only the `tensor_merged` fp8 reading for A was known (all nine endpoints negative, `local_anchor/README.md`).

## Why

Family A has five plain-like endpoints (01, 12, 17, 18, 25) and no fp8 copy under the scheme that found B's and C's. Their fitted temperatures are all 0.69 to 0.73 and their calibration at 0.70 is within 1.2 standard errors of zero (`plainness_A.md`), so the existing channels do not tell them apart. If A's quantized copy uses another scheme, the likelihood-ratio test of the fp8 rule finds it once the scheme is simulated.

## Test

Same statistic as `local_anchor/fp8lr.py`: for each answer, the sum over its tokens of log q(token) minus log p(token) at temperature 0.70, where p is Qwen2.5-3B-Instruct in bf16 and q the same weights after a simulated quantization; z is the endpoint sum over its answer-level standard error. Run on this laptop (MLX), on the 64 answers per endpoint of A's nine non-prompted endpoints (01, 04, 12, 14, 17, 18, 24, 25, 33).

Schemes: fp8 `tensor_sep` and `channel` (the desk's code), `nf4` (bitsandbytes 4-bit, blocks of 64, no double quantization), `int8row` (8-bit, one scale per output row), `int8g128`, `int4g128a` (4-bit asymmetric, groups of 128), `int4g128s` (4-bit symmetric, groups of 128).

Validation first: `plain` and `tensor_merged` rerun here must reproduce the desk's A table (z within about 0.5).

## Rule

1. A scheme reads an endpoint as its copy when that endpoint is at z of +3 or more, and the three same-model controls (04 watermark, 14 top-p, 33 temperature 1.0) are all below +1 under that scheme.
2. If the copy is 12 or 18 (the closest pair to the labelled copies), the other member of the pair is A's plain model and joins the same-model group {02, 04, 14, 23, 33}. This is the fp8 rule's logic (B: fp8 22 inside the pair {22, 32}, 32 joined, public 0.85 to 1.0).
3. If the copy is 01, 17 or 25, or no endpoint qualifies, or two schemes name two different endpoints of the pair, nothing changes.
4. No threshold, scheme or endpoint is chosen after reading. Every scheme's table is published, including those that read nothing.
