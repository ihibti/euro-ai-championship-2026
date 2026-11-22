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
