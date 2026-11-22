# Noise floor on the reference endpoints, `round1.jsonl`

## Statistic: night idn_identity (idn_identity)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.00 | 0.00 | 0.00 |
| 1 same model, disguised | 10 | 0.00 | 0.00 | 0.00 |
| 2 fp8 copy against S | 5 | 0.00 | 0.00 | 0.00 |
| 3 LoRA copy against S | 15 | 1.15 | 0.00 | 0.00 |
| 4 LoRA against LoRA | 3 | 2.09 | 0.00 | 0.71 |
| 5 LoRA against fp8 | 3 | 1.15 | 0.00 | 0.00 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.87 | 1.0000 | -0.18 |
| ref-01 | ref-03 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-04 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-01 | ref-05 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-01 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-01 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.20 | 0.6571 | -0.18 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.00 | 1.0000 | 0.00 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 1.15 | 0.4286 | 0.00 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.00 | 1.0000 | 0.00 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.15 | 0.4286 | 0.00 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 1.15 | 0.4286 | 0.71 |
| ref-02 | ref-08 | 2 fp8 copy against S | -0.87 | 1.0000 | -0.18 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.83 | 1.0000 | -0.35 |
| ref-03 | ref-04 | 3 LoRA copy against S | -0.41 | 1.0000 | -0.35 |
| ref-03 | ref-05 | 1 same model, disguised | -0.87 | 1.0000 | -0.18 |
| ref-03 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-03 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-03 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-03 | ref-09 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-06 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.00 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.71 |
| ref-04 | ref-08 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.00 |
| ref-05 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-05 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-05 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.71 |
| ref-06 | ref-08 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-06 | ref-09 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.00 |
| ref-07 | ref-08 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.71 |
| ref-08 | ref-09 | 3 LoRA copy against S | -0.41 | 1.0000 | -0.35 |

## Statistic: night, all probes (idn_identity)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.00 | 0.00 | 0.00 |
| 1 same model, disguised | 10 | 0.00 | 0.00 | 0.00 |
| 2 fp8 copy against S | 5 | 0.00 | 0.00 | 0.00 |
| 3 LoRA copy against S | 15 | 1.15 | 0.00 | 0.00 |
| 4 LoRA against LoRA | 3 | 2.09 | 0.00 | 0.71 |
| 5 LoRA against fp8 | 3 | 1.15 | 0.00 | 0.00 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.87 | 1.0000 | -0.18 |
| ref-01 | ref-03 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-04 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-01 | ref-05 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-01 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-01 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.20 | 0.6571 | -0.18 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.00 | 1.0000 | 0.00 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 1.15 | 0.4286 | 0.00 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.00 | 1.0000 | 0.00 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.15 | 0.4286 | 0.00 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 1.15 | 0.4286 | 0.71 |
| ref-02 | ref-08 | 2 fp8 copy against S | -0.87 | 1.0000 | -0.18 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.83 | 1.0000 | -0.35 |
| ref-03 | ref-04 | 3 LoRA copy against S | -0.41 | 1.0000 | -0.35 |
| ref-03 | ref-05 | 1 same model, disguised | -0.87 | 1.0000 | -0.18 |
| ref-03 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-03 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-03 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-03 | ref-09 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-06 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.00 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.71 |
| ref-04 | ref-08 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.00 |
| ref-05 | ref-06 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-05 | ref-07 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-05 | ref-08 | 1 same model, disguised | 0.00 | 1.0000 | 0.00 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.00 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.71 |
| ref-06 | ref-08 | 1 same model, disguised | 1.15 | 0.4286 | 0.00 |
| ref-06 | ref-09 | 3 LoRA copy against S | 2.09 | 0.1714 | 0.00 |
| ref-07 | ref-08 | 3 LoRA copy against S | 1.15 | 0.4286 | 0.71 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 2.09 | 0.1714 | 0.71 |
| ref-08 | ref-09 | 3 LoRA copy against S | -0.41 | 1.0000 | -0.35 |
