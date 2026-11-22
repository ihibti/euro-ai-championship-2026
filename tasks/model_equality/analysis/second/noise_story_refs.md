# Noise floor on the reference endpoints, `story_refs.jsonl`

## Statistic: night txt_story (txt_story)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.54 | 0.04 | 0.54 |
| 1 same model, disguised | 10 | -0.81 | 0.10 | 0.42 |
| 2 fp8 copy against S | 5 | -1.12 | 0.00 | 0.10 |
| 3 LoRA copy against S | 15 | -0.64 | 0.00 | 0.20 |
| 4 LoRA against LoRA | 3 | -0.32 | 0.00 | 0.24 |
| 5 LoRA against fp8 | 3 | -0.20 | 0.00 | -0.23 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 1.32 | 0.1410 | -0.95 |
| ref-01 | ref-03 | 1 same model, disguised | -1.40 | 0.8690 | -0.82 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.01 | 0.5620 | 0.80 |
| ref-01 | ref-05 | 1 same model, disguised | 1.10 | 0.2920 | -0.23 |
| ref-01 | ref-06 | 1 same model, disguised | -1.17 | 0.9265 | 0.07 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.06 | 0.5370 | -0.43 |
| ref-01 | ref-08 | 1 same model, disguised | -1.38 | 0.8900 | 0.45 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.07 | 0.3320 | 0.00 |
| ref-02 | ref-03 | 2 fp8 copy against S | -1.27 | 0.8455 | 0.10 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -0.43 | 0.6590 | 1.01 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.97 | 0.3035 | -0.35 |
| ref-02 | ref-06 | 2 fp8 copy against S | -1.12 | 0.8730 | 0.27 |
| ref-02 | ref-07 | 5 LoRA against fp8 | -0.20 | 0.5455 | -1.20 |
| ref-02 | ref-08 | 2 fp8 copy against S | -1.28 | 0.8805 | 1.95 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 1.21 | 0.0930 | -0.23 |
| ref-03 | ref-04 | 3 LoRA copy against S | -1.94 | 0.9755 | 1.88 |
| ref-03 | ref-05 | 1 same model, disguised | -0.96 | 0.7890 | -0.53 |
| ref-03 | ref-06 | 1 same model, disguised | 2.80 | 0.0130 | 0.94 |
| ref-03 | ref-07 | 3 LoRA copy against S | -1.77 | 0.9550 | -1.29 |
| ref-03 | ref-08 | 1 same model, disguised | -0.78 | 0.7070 | 2.49 |
| ref-03 | ref-09 | 3 LoRA copy against S | -0.91 | 0.7695 | -0.06 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.29 | 0.5310 | 0.69 |
| ref-04 | ref-06 | 3 LoRA copy against S | -1.78 | 0.9250 | 1.68 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.13 | 0.2435 | -0.26 |
| ref-04 | ref-08 | 3 LoRA copy against S | -0.64 | 0.7020 | 0.32 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -0.55 | 0.6720 | 0.95 |
| ref-05 | ref-06 | 1 same model, disguised | -0.65 | 0.5665 | 0.39 |
| ref-05 | ref-07 | 3 LoRA copy against S | -0.25 | 0.4465 | -0.11 |
| ref-05 | ref-08 | 1 same model, disguised | -0.83 | 0.7490 | 0.80 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.20 | 0.1745 | 0.20 |
| ref-06 | ref-07 | 3 LoRA copy against S | -1.61 | 0.9225 | -1.05 |
| ref-06 | ref-08 | 1 same model, disguised | -0.76 | 0.7705 | 1.11 |
| ref-06 | ref-09 | 3 LoRA copy against S | -0.54 | 0.6305 | -0.31 |
| ref-07 | ref-08 | 3 LoRA copy against S | -1.38 | 0.9470 | 0.44 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.32 | 0.4960 | 0.24 |
| ref-08 | ref-09 | 3 LoRA copy against S | -1.48 | 0.9305 | 0.30 |

## Statistic: night, all probes (txt_story)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.54 | 0.04 | 0.54 |
| 1 same model, disguised | 10 | -0.79 | 0.10 | 0.42 |
| 2 fp8 copy against S | 5 | -1.16 | 0.00 | 0.10 |
| 3 LoRA copy against S | 15 | -0.60 | 0.00 | 0.20 |
| 4 LoRA against LoRA | 3 | -0.31 | 0.00 | 0.24 |
| 5 LoRA against fp8 | 3 | -0.19 | 0.00 | -0.23 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 1.25 | 0.1555 | -0.95 |
| ref-01 | ref-03 | 1 same model, disguised | -1.43 | 0.8745 | -0.82 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.02 | 0.5660 | 0.80 |
| ref-01 | ref-05 | 1 same model, disguised | 1.14 | 0.2825 | -0.23 |
| ref-01 | ref-06 | 1 same model, disguised | -1.19 | 0.9270 | 0.07 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.04 | 0.5245 | -0.43 |
| ref-01 | ref-08 | 1 same model, disguised | -1.39 | 0.8940 | 0.45 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.02 | 0.3510 | 0.00 |
| ref-02 | ref-03 | 2 fp8 copy against S | -1.23 | 0.8340 | 0.10 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -0.39 | 0.6465 | 1.01 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.94 | 0.3290 | -0.35 |
| ref-02 | ref-06 | 2 fp8 copy against S | -1.16 | 0.8870 | 0.27 |
| ref-02 | ref-07 | 5 LoRA against fp8 | -0.19 | 0.5470 | -1.20 |
| ref-02 | ref-08 | 2 fp8 copy against S | -1.18 | 0.8555 | 1.95 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 1.25 | 0.0865 | -0.23 |
| ref-03 | ref-04 | 3 LoRA copy against S | -1.89 | 0.9710 | 1.88 |
| ref-03 | ref-05 | 1 same model, disguised | -1.02 | 0.8040 | -0.53 |
| ref-03 | ref-06 | 1 same model, disguised | 2.72 | 0.0170 | 0.94 |
| ref-03 | ref-07 | 3 LoRA copy against S | -1.73 | 0.9460 | -1.29 |
| ref-03 | ref-08 | 1 same model, disguised | -0.74 | 0.6935 | 2.49 |
| ref-03 | ref-09 | 3 LoRA copy against S | -0.85 | 0.7615 | -0.06 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.33 | 0.5370 | 0.69 |
| ref-04 | ref-06 | 3 LoRA copy against S | -1.80 | 0.9280 | 1.68 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.12 | 0.2405 | -0.26 |
| ref-04 | ref-08 | 3 LoRA copy against S | -0.60 | 0.6770 | 0.32 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -0.53 | 0.6658 | 0.95 |
| ref-05 | ref-06 | 1 same model, disguised | -0.66 | 0.5730 | 0.39 |
| ref-05 | ref-07 | 3 LoRA copy against S | -0.22 | 0.4355 | -0.11 |
| ref-05 | ref-08 | 1 same model, disguised | -0.83 | 0.7515 | 0.80 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.23 | 0.1685 | 0.20 |
| ref-06 | ref-07 | 3 LoRA copy against S | -1.69 | 0.9335 | -1.05 |
| ref-06 | ref-08 | 1 same model, disguised | -0.75 | 0.7700 | 1.11 |
| ref-06 | ref-09 | 3 LoRA copy against S | -0.55 | 0.6245 | -0.31 |
| ref-07 | ref-08 | 3 LoRA copy against S | -1.36 | 0.9480 | 0.44 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.31 | 0.4870 | 0.24 |
| ref-08 | ref-09 | 3 LoRA copy against S | -1.46 | 0.9285 | 0.30 |
