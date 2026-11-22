# P3 confirmation on night_d, second method

Devin, 2026-11-22 01:52 Paris. Fresh answers: `answers/night-d.jsonl` (`67a9498`, 01:38), 10 per
reference endpoint and probe. Both features were committed before it: the frozen one at
`897fcbb`, `capped` at `f7840f1` (01:33).

## Frozen rule on the fresh `d1_repeat150` answers: not confirmed

| feature | p | rates | base | called LoRA | MCC |
|---|---|---|---|---|---|
| long (frozen) | 0.002 | 0.00 / 0.48 | 01 02 04 06 07 09 | 03, 05, 08 | 0 |
| capped | 0.002 | 0.05 / 0.63 | 01 02 04 07 09 | 03, 05, 06, 08 | 0 |

The mixture finds two rates, but the classes are not LoRA against the rest. Three disguised
copies of S never run long (ref-03, ref-05, ref-08: 0 of 10), while the plain-looking ref-01 runs
long 4 of 10 and the fp8 copy 4 to 5 of 10. The larger class is then the high-rate one, so the
rule calls the quiet disguises LoRA. The stop rate depends on the disguise as well as on the
weights.

Counts of long runs (of 10), night-b then night-d: S ref-01 3, 4; ref-03 0, 0; ref-05 3, 0;
ref-06 2, 2; ref-08 0, 0. fp8 ref-02 2, 5. LoRA ref-04 8, 8; ref-07 7, 5; ref-09 6, 5.

## With 20 answers per endpoint (night-b and night-d pooled)

| feature | p | rates | base | called LoRA | MCC |
|---|---|---|---|---|---|
| long | 0.002 | 0.09 / 0.53 | 01 02 04 07 09 | 03, 05, 06, 08 | 0 |
| capped | 0.002 | 0.18 / 0.80 | 01 02 03 05 06 08 | 04, 07, 09 | 0.7338 |

Under `capped`, S and fp8 run long at 0.00 to 0.35 and the LoRA copies at 0.60 to 0.90.

Draws of n of the 20 answers per endpoint (40 draws, 199 bootstrap draws each,
`p3_subsample.py`):

| feature | n | mean MCC | share of draws at MCC 0 |
|---|---|---|---|
| capped | 10 | 0.566 | 0.12 |
| capped | 15 | 0.647 | 0.03 |
| capped | 20 | 0.734 | 0.00 |
| long | 10 | 0.198 | 0.72 |
| long | 15 | 0.269 | 0.65 |
| long | 20 | 0.000 | 1.00 |

## The pairwise statistic on every night_d probe (`noise_night-d.md`)

Median z and share of pairs at p < 0.05:

| pair type | d1_repeat150 | d2_story100 | d3_openlist | d4_copycont | all four |
|---|---|---|---|---|---|
| same model, disguised (10) | 0.64 (0.10) | -0.57 (0.10) | 1.10 (0.30) | -0.45 (0.30) | 0.14 (0.20) |
| fp8 against S (5) | 0.44 (0.00) | 0.90 (0.20) | 0.15 (0.20) | -0.60 (0.00) | -0.71 (0.40) |
| LoRA against S (15) | 1.86 (0.47) | 1.44 (0.47) | 1.07 (0.40) | -0.48 (0.20) | 7.43 (0.73) |
| LoRA against LoRA (3) | -1.27 (0.00) | 8.91 (0.67) | 1.29 (0.33) | -0.50 (0.33) | 9.00 (0.67) |

- `d1_repeat150` is weaker than on night-b: 7 of 15 LoRA pairs at p < 0.05, against 12 of 15.
- `d2_story100` is as strong on LoRA and the only probe that tells LoRA strengths apart (2 of 3
  pairs, median z 8.91); 1 of 10 disguised pairs.
- `d3_openlist` and `d4_copycont` separate disguised copies (3 of 10 each): not for the
  same-model decision.
- fp8 stays invisible.

Grouping with the frozen rule (`group2.py`, all four probes): MCC 0.3964 (fp8 joins S, ref-03
splits off, ref-04 joins ref-09); keeping 7 of the 10 answers: mean 0.251 over 20 draws.
Exploratory, chosen after seeing the table above: `d1_repeat150` and `d2_story100` only give MCC
0.5241 (fp8 and ref-09 join S), mean 0.468 over 20 draws.

## Reading for the scored spend

- My rule needs about 20 stop answers per endpoint, read with `capped`. At 10 it fails on the
  fresh answers.
- `d2_story100` is worth buying next to `d1`. `d3` and `d4` are not.
- The risk on the scored families is the one shown here: a disguise that changes the stop rate.
  My rule only guards against it through "the larger class is the base".

## Commands

```
cd tasks/model_equality/analysis/second
uv run python p3rule.py --answers ../../answers/night-d.jsonl --prefix ref- --probe d1_repeat150 [--feature capped]
uv run python pool.py --into p3_repeat --out /tmp/p3_pooled_refs.jsonl \
    ../../answers/night-b.jsonl:p3_repeat ../../answers/night-d.jsonl:d1_repeat150
uv run python p3rule.py --answers /tmp/p3_pooled_refs.jsonl --prefix ref- [--feature capped]
uv run python p3_subsample.py --answers /tmp/p3_pooled_refs.jsonl --sizes 10 15 20 --draws 40
uv run python noise.py --probes night --answers ../../answers/night-d.jsonl
uv run python group2.py --probes night --prefix ref- --answers ../../answers/night-d.jsonl \
    --out grouping_refs_night-d.csv --subsample 20 --subsample-size 7
```
