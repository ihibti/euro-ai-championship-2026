# Simulator: share of p < 0.05, one 30-item binary battery

120 replicates per cell; 99 bootstrap draws; permutations exact. Base item logits are drawn from N(0, sd^2) anew in each replicate; 5% of items are unparsed. A same-model row should read about 0.05 or less. Columns: base sd and answers per endpoint. Standard error about 0.015 at 0.05 with 200 replicates.

| scenario | test | sd 1.5, n=4 | sd 1.5, n=8 | sd 3.0, n=4 | sd 3.0, n=8 |
|---|---|---|---|---|---|
| same model: identical | affine/bootstrap | 0.067 | 0.067 | 0.125 | 0.100 |
| same model: identical | affine/permutation | 0.025 | 0.058 | 0.042 | 0.058 |
| same model: identical | delta/permutation | 0.008 | 0.042 | 0.033 | 0.042 |
| same model: temperature x1.6 | affine/bootstrap | 0.142 | 0.017 | 0.175 | 0.083 |
| same model: temperature x1.6 | affine/permutation | 0.050 | 0.042 | 0.083 | 0.133 |
| same model: temperature x1.6 | delta/permutation | 0.033 | 0.067 | 0.025 | 0.117 |
| same model: uniform shift 0.7 | affine/bootstrap | 0.100 | 0.067 | 0.092 | 0.125 |
| same model: uniform shift 0.7 | affine/permutation | 0.017 | 0.067 | 0.033 | 0.067 |
| same model: uniform shift 0.7 | delta/permutation | 0.058 | 0.292 | 0.058 | 0.217 |
| different: item shifts sd 0.3 | affine/bootstrap | 0.075 | 0.092 | 0.142 | 0.108 |
| different: item shifts sd 0.3 | affine/permutation | 0.025 | 0.092 | 0.067 | 0.050 |
| different: item shifts sd 0.3 | delta/permutation | 0.042 | 0.100 | 0.058 | 0.058 |
| different: item shifts sd 0.7 | affine/bootstrap | 0.183 | 0.292 | 0.192 | 0.258 |
| different: item shifts sd 0.7 | affine/permutation | 0.075 | 0.258 | 0.100 | 0.192 |
| different: item shifts sd 0.7 | delta/permutation | 0.067 | 0.292 | 0.033 | 0.233 |
| different: item shifts sd 1.5 | affine/bootstrap | 0.442 | 0.867 | 0.525 | 0.817 |
| different: item shifts sd 1.5 | affine/permutation | 0.217 | 0.758 | 0.258 | 0.675 |
| different: item shifts sd 1.5 | delta/permutation | 0.317 | 0.900 | 0.275 | 0.783 |
| different: item shifts sd 3 | affine/bootstrap | 0.833 | 1.000 | 0.892 | 1.000 |
| different: item shifts sd 3 | affine/permutation | 0.483 | 0.975 | 0.658 | 0.992 |
| different: item shifts sd 3 | delta/permutation | 0.908 | 1.000 | 0.842 | 1.000 |
