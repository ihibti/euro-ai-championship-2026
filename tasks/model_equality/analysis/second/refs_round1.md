# Reference grouping, round 1 at 40 answers per endpoint (4 per prompt)

All six combinations tried, in the order run. Subsampling keeps 3 of the 4 answers per prompt and endpoint, 30 draws.

| prompts | linkage | groups | MCC | subsample mean | subsample min / median / max |
|---|---|---|---|---|---|
| batteries | average | ref-01 ref-03 ref-04 ref-05 ref-06 ref-08 ref-09; ref-02 ref-07 | 0.4947 | 0.072 | 0.000 / 0.000 / 0.331 |
| batteries | complete | ref-01 ref-02 ref-03 ref-04 ref-05 ref-06 ref-07 ref-08 ref-09 | 0.0000 | 0.000 | 0.000 / 0.000 / 0.000 |
| all | average | ref-01 ref-02 ref-03 ref-04 ref-05 ref-06 ref-09; ref-07; ref-08 | 0.0210 | 0.074 | 0.000 / 0.000 / 0.524 |
| all | complete | ref-01 ref-07 ref-08 ref-09; ref-02 ref-05 ref-06; ref-03 ref-04 | 0.0000 | 0.022 | 0.000 / 0.000 / 0.331 |
| sequences | average | ref-01 ref-02 ref-03 ref-04 ref-05 ref-06 ref-07 ref-08 ref-09 | 0.0000 | 0.012 | 0.000 / 0.000 / 0.331 |
| sequences | complete | ref-01 ref-02 ref-03 ref-04 ref-05 ref-06 ref-07 ref-08 ref-09 | 0.0000 | 0.000 | 0.000 / 0.000 / 0.000 |
