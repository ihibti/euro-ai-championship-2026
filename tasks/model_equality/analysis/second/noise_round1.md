# Noise floor on the reference endpoints, `round1.jsonl`

## Statistic: batteries (bat_choice, bat_trivia)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.35 | 0.00 | 0.35 |
| 1 same model, disguised | 10 | 1.05 | 0.10 | 0.61 |
| 2 fp8 copy against S | 5 | 3.10 | 0.60 | 0.90 |
| 3 LoRA copy against S | 15 | 1.73 | 0.20 | 0.53 |
| 4 LoRA against LoRA | 3 | 1.85 | 0.00 | 0.54 |
| 5 LoRA against fp8 | 3 | 2.22 | 0.33 | 0.63 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 3.47 | 0.0180 | 0.42 |
| ref-01 | ref-03 | 1 same model, disguised | -1.57 | 0.8555 | -0.73 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.87 | 0.7200 | -0.42 |
| ref-01 | ref-05 | 1 same model, disguised | -0.42 | 0.5396 | -0.33 |
| ref-01 | ref-06 | 1 same model, disguised | 1.05 | 0.2098 | 0.77 |
| ref-01 | ref-07 | 3 LoRA copy against S | 2.12 | 0.0808 | 0.43 |
| ref-01 | ref-08 | 1 same model, disguised | 1.81 | 0.1053 | 1.09 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.15 | 0.4824 | -0.16 |
| ref-02 | ref-03 | 2 fp8 copy against S | 1.87 | 0.1078 | 1.30 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 3.85 | 0.0098 | 1.24 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.65 | 0.3257 | -0.11 |
| ref-02 | ref-06 | 2 fp8 copy against S | 3.10 | 0.0253 | 0.90 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 2.10 | 0.0653 | 0.48 |
| ref-02 | ref-08 | 2 fp8 copy against S | 3.89 | 0.0065 | 1.54 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 2.22 | 0.0669 | 0.63 |
| ref-03 | ref-04 | 3 LoRA copy against S | -1.39 | 0.8473 | 0.53 |
| ref-03 | ref-05 | 1 same model, disguised | -0.08 | 0.4751 | -0.18 |
| ref-03 | ref-06 | 1 same model, disguised | 1.45 | 0.1502 | 1.24 |
| ref-03 | ref-07 | 3 LoRA copy against S | -0.68 | 0.6678 | 0.38 |
| ref-03 | ref-08 | 1 same model, disguised | 0.14 | 0.4204 | 0.45 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.50 | 0.3069 | 0.38 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.96 | 0.1012 | 1.16 |
| ref-04 | ref-06 | 3 LoRA copy against S | 1.73 | 0.1192 | 0.89 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.85 | 0.1029 | 0.47 |
| ref-04 | ref-08 | 3 LoRA copy against S | 3.08 | 0.0286 | 1.41 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 1.51 | 0.1380 | 0.54 |
| ref-05 | ref-06 | 1 same model, disguised | 1.05 | 0.2327 | 0.84 |
| ref-05 | ref-07 | 3 LoRA copy against S | 1.94 | 0.0914 | 0.04 |
| ref-05 | ref-08 | 1 same model, disguised | 1.45 | 0.1461 | 0.21 |
| ref-05 | ref-09 | 3 LoRA copy against S | 0.48 | 0.3257 | 0.34 |
| ref-06 | ref-07 | 3 LoRA copy against S | 4.86 | 0.0033 | 1.90 |
| ref-06 | ref-08 | 1 same model, disguised | 2.60 | 0.0490 | 1.00 |
| ref-06 | ref-09 | 3 LoRA copy against S | 0.45 | 0.3535 | 1.18 |
| ref-07 | ref-08 | 3 LoRA copy against S | 3.02 | 0.0155 | 1.44 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 2.17 | 0.0873 | 0.59 |
| ref-08 | ref-09 | 3 LoRA copy against S | 2.10 | 0.0873 | 0.70 |

## Statistic: sequences (seq_alphabet, seq_count30, seq_days, seq_months, seq_words40)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.56 | 0.04 | -0.56 |
| 1 same model, disguised | 10 | 2.41 | 0.20 | 0.75 |
| 2 fp8 copy against S | 5 | 0.13 | 0.00 | -0.40 |
| 3 LoRA copy against S | 15 | 1.87 | 0.13 | 1.23 |
| 4 LoRA against LoRA | 3 | 0.24 | 0.00 | 1.61 |
| 5 LoRA against fp8 | 3 | -0.90 | 0.00 | 0.14 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 0.13 | 0.4435 | -0.61 |
| ref-01 | ref-03 | 1 same model, disguised | 0.54 | 0.3041 | 0.38 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.85 | 0.3500 | 0.71 |
| ref-01 | ref-05 | 1 same model, disguised | 2.95 | 0.0829 | -0.13 |
| ref-01 | ref-06 | 1 same model, disguised | 2.30 | 0.1279 | -0.41 |
| ref-01 | ref-07 | 3 LoRA copy against S | 3.25 | 0.0848 | 1.82 |
| ref-01 | ref-08 | 1 same model, disguised | 0.37 | 0.3933 | 0.80 |
| ref-01 | ref-09 | 3 LoRA copy against S | -1.50 | 0.7352 | -1.06 |
| ref-02 | ref-03 | 2 fp8 copy against S | 3.00 | 0.0730 | 1.53 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -1.65 | 0.8041 | 0.14 |
| ref-02 | ref-05 | 2 fp8 copy against S | -2.35 | 0.9011 | -0.40 |
| ref-02 | ref-06 | 2 fp8 copy against S | -0.76 | 0.6898 | -1.25 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 1.35 | 0.2218 | 1.08 |
| ref-02 | ref-08 | 2 fp8 copy against S | 1.74 | 0.2255 | 1.25 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.90 | 0.6162 | -0.16 |
| ref-03 | ref-04 | 3 LoRA copy against S | 1.87 | 0.1230 | 1.64 |
| ref-03 | ref-05 | 1 same model, disguised | 2.51 | 0.0792 | 2.28 |
| ref-03 | ref-06 | 1 same model, disguised | 4.60 | 0.0189 | 2.65 |
| ref-03 | ref-07 | 3 LoRA copy against S | 5.02 | 0.0255 | 2.99 |
| ref-03 | ref-08 | 1 same model, disguised | 3.86 | 0.0506 | 3.16 |
| ref-03 | ref-09 | 3 LoRA copy against S | 2.54 | 0.0957 | 1.23 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.93 | 0.7648 | 0.10 |
| ref-04 | ref-06 | 3 LoRA copy against S | 0.31 | 0.3429 | 0.19 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 2.67 | 0.0696 | 2.21 |
| ref-04 | ref-08 | 3 LoRA copy against S | 4.03 | 0.0427 | 2.30 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -0.31 | 0.5290 | 0.45 |
| ref-05 | ref-06 | 1 same model, disguised | -0.46 | 0.5894 | 0.01 |
| ref-05 | ref-07 | 3 LoRA copy against S | 2.39 | 0.1289 | 1.79 |
| ref-05 | ref-08 | 1 same model, disguised | 0.90 | 0.3333 | 0.70 |
| ref-05 | ref-09 | 3 LoRA copy against S | -0.44 | 0.5690 | 0.14 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.07 | 0.1285 | 1.59 |
| ref-06 | ref-08 | 1 same model, disguised | 5.30 | 0.0101 | 1.98 |
| ref-06 | ref-09 | 3 LoRA copy against S | 0.70 | 0.3923 | -0.30 |
| ref-07 | ref-08 | 3 LoRA copy against S | 2.08 | 0.1774 | 2.30 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.24 | 0.4143 | 1.61 |
| ref-08 | ref-09 | 3 LoRA copy against S | 0.78 | 0.3429 | 0.85 |

## Statistic: all (bat_choice, bat_trivia, seq_alphabet, seq_count30, seq_days, seq_months, seq_words40)

Answers per endpoint (minimum over these prompts): ref-01 4, ref-02 4, ref-03 4, ref-04 4, ref-05 4, ref-06 4, ref-07 4, ref-08 4, ref-09 4.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.03 | 0.04 | -0.03 |
| 1 same model, disguised | 10 | 2.48 | 0.20 | 1.40 |
| 2 fp8 copy against S | 5 | 3.61 | 0.40 | -0.19 |
| 3 LoRA copy against S | 15 | 2.87 | 0.33 | 1.61 |
| 4 LoRA against LoRA | 3 | 2.42 | 0.33 | 2.20 |
| 5 LoRA against fp8 | 3 | 2.19 | 0.00 | 1.38 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 3.61 | 0.0832 | -0.19 |
| ref-01 | ref-03 | 1 same model, disguised | -1.03 | 0.6499 | -0.35 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.02 | 0.4727 | 0.29 |
| ref-01 | ref-05 | 1 same model, disguised | 2.52 | 0.1500 | -0.46 |
| ref-01 | ref-06 | 1 same model, disguised | 3.35 | 0.0930 | 0.36 |
| ref-01 | ref-07 | 3 LoRA copy against S | 5.37 | 0.0276 | 2.25 |
| ref-01 | ref-08 | 1 same model, disguised | 2.18 | 0.1960 | 1.89 |
| ref-01 | ref-09 | 3 LoRA copy against S | -1.65 | 0.7178 | -1.22 |
| ref-02 | ref-03 | 2 fp8 copy against S | 4.87 | 0.0273 | 2.83 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 2.19 | 0.1444 | 1.38 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.70 | 0.7483 | -0.51 |
| ref-02 | ref-06 | 2 fp8 copy against S | 2.34 | 0.1289 | -0.35 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 3.44 | 0.0685 | 1.55 |
| ref-02 | ref-08 | 2 fp8 copy against S | 5.63 | 0.0197 | 2.79 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 1.32 | 0.2890 | 0.47 |
| ref-03 | ref-04 | 3 LoRA copy against S | 0.49 | 0.3795 | 2.17 |
| ref-03 | ref-05 | 1 same model, disguised | 2.44 | 0.1460 | 2.11 |
| ref-03 | ref-06 | 1 same model, disguised | 6.05 | 0.0076 | 3.89 |
| ref-03 | ref-07 | 3 LoRA copy against S | 4.35 | 0.0600 | 3.37 |
| ref-03 | ref-08 | 1 same model, disguised | 4.00 | 0.0748 | 3.61 |
| ref-03 | ref-09 | 3 LoRA copy against S | 3.04 | 0.1129 | 1.61 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.03 | 0.3229 | 1.26 |
| ref-04 | ref-06 | 3 LoRA copy against S | 2.03 | 0.1265 | 1.09 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 4.51 | 0.0254 | 2.68 |
| ref-04 | ref-08 | 3 LoRA copy against S | 7.11 | 0.0065 | 3.71 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 1.19 | 0.2903 | 0.98 |
| ref-05 | ref-06 | 1 same model, disguised | 0.60 | 0.3871 | 0.86 |
| ref-05 | ref-07 | 3 LoRA copy against S | 4.33 | 0.0445 | 1.83 |
| ref-05 | ref-08 | 1 same model, disguised | 2.35 | 0.1862 | 0.92 |
| ref-05 | ref-09 | 3 LoRA copy against S | 0.03 | 0.4778 | 0.48 |
| ref-06 | ref-07 | 3 LoRA copy against S | 6.93 | 0.0034 | 3.49 |
| ref-06 | ref-08 | 1 same model, disguised | 7.90 | 0.0024 | 2.98 |
| ref-06 | ref-09 | 3 LoRA copy against S | 1.15 | 0.3170 | 0.88 |
| ref-07 | ref-08 | 3 LoRA copy against S | 5.09 | 0.0384 | 3.74 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 2.42 | 0.1695 | 2.20 |
| ref-08 | ref-09 | 3 LoRA copy against S | 2.87 | 0.1379 | 1.55 |
