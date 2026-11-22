# Plain against fp8 on the repetition prompt, 40 answers each (second method)

Answers: `night-b.jsonl` p3_repeat at 30 repeats (`af2f973`) and `night-d.jsonl` d1_repeat150 at 10, pooled with `pool.py` (same prompt).

| reference | answers | capped | long run (>= 100) | median run length |
|---|---|---|---|---|
| ref-01 (plain) | 40 | 14 | 14 | 10 |
| ref-02 (fp8) | 40 | 9 | 11 | 1 |

- capped, Fisher exact p = 0.32; run length, Mann-Whitney p = 0.14;
- mode statistic (`clique.py`, alpha 0.01): the pair is not rejected.

At 40 answers none of the second method's features separates fp8 from plain on this prompt.

## Trivia battery and identity at 14 answers each (`6260e78`)

Mode statistic, plain (ref-01) against fp8 (ref-02):

| probe | standardised z | p |
|---|---|---|
| bat_trivia | 2.13 | 0.074 |
| idn_identity | 1.37 | 0.078 |

The two probes together are not rejected either (`clique.py`, alpha 0.01). Mean trivia correctness is 0.338 for plain and 0.360 for fp8. Both probes lean the same way, but at 14 answers neither separates fp8.

## Stop probes at 30 answers each, repetition prompt at 60 (`d67accb`)

| probe | answers each | result |
|---|---|---|
| repetition (p3 30 + d1 30, pooled) | 60 | capped 21 vs 17, Fisher p 0.56; long 21 vs 18; median run 10 vs 1, Mann-Whitney p 0.041; mode statistic p 0.70 |
| d2_story100 | 30 | mode statistic z 0.54, p 0.38 |
| d3_openlist | 30 | z -1.38, p 0.89 |
| d4_copycont | 30 | z 0.17, p 0.50 |

Only the run-length distribution differs, at p 0.041, which is one test among six and does not survive a correction. fp8 stays invisible to the second method.
