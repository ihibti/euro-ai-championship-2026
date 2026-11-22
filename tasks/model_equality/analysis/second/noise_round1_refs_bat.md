# Noise floor on the reference endpoints, `round1_refs_bat.jsonl`

## Statistic: night bat_choice (bat_choice)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.00 | 0.00 | 0.00 |
| 1 same model, disguised | 10 | 0.69 | 0.10 | 0.49 |
| 2 fp8 copy against S | 5 | 0.32 | 0.20 | 0.43 |
| 3 LoRA copy against S | 15 | 0.48 | 0.13 | 0.35 |
| 4 LoRA against LoRA | 3 | 0.36 | 0.00 | 0.35 |
| 5 LoRA against fp8 | 3 | -0.48 | 0.00 | 0.24 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 1.69 | 0.0857 | 0.44 |
| ref-01 | ref-03 | 1 same model, disguised | -0.02 | 0.4857 | -0.02 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.10 | 0.4857 | 0.66 |
| ref-01 | ref-05 | 1 same model, disguised | 0.42 | 0.1429 | 0.18 |
| ref-01 | ref-06 | 1 same model, disguised | 1.01 | 0.1143 | 1.06 |
| ref-01 | ref-07 | 3 LoRA copy against S | 2.32 | 0.0286 | 0.28 |
| ref-01 | ref-08 | 1 same model, disguised | 0.95 | 0.2286 | 0.83 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.35 | 0.7429 | -0.53 |
| ref-02 | ref-03 | 2 fp8 copy against S | -0.42 | 0.6857 | 0.43 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 1.64 | 0.0571 | 0.66 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.33 | 0.9143 | 0.04 |
| ref-02 | ref-06 | 2 fp8 copy against S | 2.12 | 0.0286 | 1.41 |
| ref-02 | ref-07 | 5 LoRA against fp8 | -1.04 | 0.8571 | -0.88 |
| ref-02 | ref-08 | 2 fp8 copy against S | 0.32 | 0.3714 | 0.37 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.48 | 0.7143 | 0.24 |
| ref-03 | ref-04 | 3 LoRA copy against S | -0.69 | 0.7714 | 0.31 |
| ref-03 | ref-05 | 1 same model, disguised | -1.11 | 0.9143 | -0.31 |
| ref-03 | ref-06 | 1 same model, disguised | 1.16 | 0.0857 | 1.41 |
| ref-03 | ref-07 | 3 LoRA copy against S | -1.20 | 0.9143 | 0.11 |
| ref-03 | ref-08 | 1 same model, disguised | -0.40 | 0.6571 | 0.27 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.09 | 0.3714 | -0.60 |
| ref-04 | ref-05 | 3 LoRA copy against S | 0.48 | 0.3143 | 0.62 |
| ref-04 | ref-06 | 3 LoRA copy against S | 1.07 | 0.1714 | 1.41 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 0.75 | 0.2286 | 0.79 |
| ref-04 | ref-08 | 3 LoRA copy against S | 1.02 | 0.1714 | 1.35 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.36 | 0.4286 | 0.35 |
| ref-05 | ref-06 | 1 same model, disguised | 1.83 | 0.1143 | 0.71 |
| ref-05 | ref-07 | 3 LoRA copy against S | -1.15 | 0.9143 | -0.11 |
| ref-05 | ref-08 | 1 same model, disguised | -0.76 | 0.8286 | 0.15 |
| ref-05 | ref-09 | 3 LoRA copy against S | -0.63 | 0.6286 | 0.18 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.40 | 0.0286 | 1.41 |
| ref-06 | ref-08 | 1 same model, disguised | 2.71 | 0.0286 | 1.41 |
| ref-06 | ref-09 | 3 LoRA copy against S | 1.12 | 0.1143 | 0.71 |
| ref-07 | ref-08 | 3 LoRA copy against S | 0.69 | 0.2571 | 0.35 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.17 | 0.6000 | -0.18 |
| ref-08 | ref-09 | 3 LoRA copy against S | 1.25 | 0.1143 | 0.48 |

## Statistic: night bat_trivia (bat_trivia)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.71 | 0.00 | -0.71 |
| 1 same model, disguised | 10 | 0.22 | 0.00 | 0.27 |
| 2 fp8 copy against S | 5 | 1.00 | 0.40 | -0.27 |
| 3 LoRA copy against S | 15 | 0.48 | 0.07 | 0.29 |
| 4 LoRA against LoRA | 3 | 0.61 | 0.00 | -0.35 |
| 5 LoRA against fp8 | 3 | 1.35 | 0.00 | 0.18 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 0.03 | 0.4286 | -0.88 |
| ref-01 | ref-03 | 1 same model, disguised | -0.68 | 0.6286 | -0.88 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.08 | 0.3714 | -1.24 |
| ref-01 | ref-05 | 1 same model, disguised | -0.52 | 0.6857 | 0.18 |
| ref-01 | ref-06 | 1 same model, disguised | 0.62 | 0.3714 | -0.97 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.17 | 0.5143 | -0.11 |
| ref-01 | ref-08 | 1 same model, disguised | 0.09 | 0.4857 | 0.44 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.37 | 0.0857 | 0.35 |
| ref-02 | ref-03 | 2 fp8 copy against S | 1.00 | 0.2286 | 0.00 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 2.06 | 0.1143 | 0.18 |
| ref-02 | ref-05 | 2 fp8 copy against S | -0.17 | 0.4571 | -0.27 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.82 | 0.0286 | -0.42 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 0.93 | 0.1714 | -0.18 |
| ref-02 | ref-08 | 2 fp8 copy against S | 3.43 | 0.0286 | 1.30 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 1.35 | 0.2571 | 0.53 |
| ref-03 | ref-04 | 3 LoRA copy against S | -0.46 | 0.7714 | -0.53 |
| ref-03 | ref-05 | 1 same model, disguised | -0.15 | 0.5143 | 1.01 |
| ref-03 | ref-06 | 1 same model, disguised | 0.41 | 0.3429 | -0.84 |
| ref-03 | ref-07 | 3 LoRA copy against S | -0.72 | 0.7143 | -0.18 |
| ref-03 | ref-08 | 1 same model, disguised | 1.14 | 0.2000 | 0.35 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.86 | 0.4857 | 0.88 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.40 | 0.1714 | 1.17 |
| ref-04 | ref-06 | 3 LoRA copy against S | 0.27 | 0.4000 | -0.22 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 0.61 | 0.2571 | -0.35 |
| ref-04 | ref-08 | 3 LoRA copy against S | 2.19 | 0.0286 | 0.86 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.77 | 0.2286 | 0.53 |
| ref-05 | ref-06 | 1 same model, disguised | -0.04 | 0.6000 | 0.00 |
| ref-05 | ref-07 | 3 LoRA copy against S | 0.28 | 0.4286 | 0.29 |
| ref-05 | ref-08 | 1 same model, disguised | 0.36 | 0.3714 | 1.19 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.84 | 0.1143 | 1.41 |
| ref-06 | ref-07 | 3 LoRA copy against S | 0.48 | 0.2286 | -0.84 |
| ref-06 | ref-08 | 1 same model, disguised | 0.86 | 0.1714 | 0.55 |
| ref-06 | ref-09 | 3 LoRA copy against S | -0.44 | 0.8571 | 0.18 |
| ref-07 | ref-08 | 3 LoRA copy against S | 0.71 | 0.1714 | 0.90 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.34 | 0.3429 | -0.53 |
| ref-08 | ref-09 | 3 LoRA copy against S | 2.27 | 0.0857 | 1.39 |

## Statistic: night, all probes (bat_choice, bat_trivia)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.16 | 0.00 | 0.16 |
| 1 same model, disguised | 10 | 0.90 | 0.10 | 0.66 |
| 2 fp8 copy against S | 5 | 1.72 | 0.40 | 0.43 |
| 3 LoRA copy against S | 15 | 1.21 | 0.20 | 0.58 |
| 4 LoRA against LoRA | 3 | 1.13 | 0.00 | 0.44 |
| 5 LoRA against fp8 | 3 | 0.87 | 0.33 | 0.77 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 1.72 | 0.1429 | -0.44 |
| ref-01 | ref-03 | 1 same model, disguised | -0.70 | 0.6539 | -0.90 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.02 | 0.4678 | -0.58 |
| ref-01 | ref-05 | 1 same model, disguised | -0.10 | 0.4743 | 0.35 |
| ref-01 | ref-06 | 1 same model, disguised | 1.63 | 0.1257 | 0.09 |
| ref-01 | ref-07 | 3 LoRA copy against S | 2.15 | 0.0800 | 0.17 |
| ref-01 | ref-08 | 1 same model, disguised | 1.04 | 0.2302 | 1.27 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.02 | 0.2596 | -0.18 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.57 | 0.3306 | 0.43 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 3.70 | 0.0139 | 0.84 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.50 | 0.8767 | -0.23 |
| ref-02 | ref-06 | 2 fp8 copy against S | 3.93 | 0.0008 | 0.99 |
| ref-02 | ref-07 | 5 LoRA against fp8 | -0.11 | 0.5339 | -1.06 |
| ref-02 | ref-08 | 2 fp8 copy against S | 3.75 | 0.0171 | 1.67 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 0.87 | 0.2816 | 0.77 |
| ref-03 | ref-04 | 3 LoRA copy against S | -1.15 | 0.8008 | -0.22 |
| ref-03 | ref-05 | 1 same model, disguised | -1.26 | 0.8229 | 0.71 |
| ref-03 | ref-06 | 1 same model, disguised | 1.58 | 0.1380 | 0.58 |
| ref-03 | ref-07 | 3 LoRA copy against S | -1.92 | 0.9322 | -0.07 |
| ref-03 | ref-08 | 1 same model, disguised | 0.75 | 0.2939 | 0.62 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.95 | 0.2278 | 0.29 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.88 | 0.1061 | 1.79 |
| ref-04 | ref-06 | 3 LoRA copy against S | 1.34 | 0.1682 | 1.19 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.35 | 0.1690 | 0.44 |
| ref-04 | ref-08 | 3 LoRA copy against S | 3.21 | 0.0171 | 2.21 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 1.13 | 0.2057 | 0.88 |
| ref-05 | ref-06 | 1 same model, disguised | 1.79 | 0.1184 | 0.71 |
| ref-05 | ref-07 | 3 LoRA copy against S | -0.86 | 0.7200 | 0.18 |
| ref-05 | ref-08 | 1 same model, disguised | -0.40 | 0.5788 | 1.34 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.21 | 0.2016 | 1.59 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.88 | 0.0416 | 0.58 |
| ref-06 | ref-08 | 1 same model, disguised | 3.57 | 0.0131 | 1.96 |
| ref-06 | ref-09 | 3 LoRA copy against S | 0.68 | 0.2514 | 0.88 |
| ref-07 | ref-08 | 3 LoRA copy against S | 1.41 | 0.1494 | 1.25 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.17 | 0.4041 | -0.71 |
| ref-08 | ref-09 | 3 LoRA copy against S | 3.52 | 0.0147 | 1.87 |
