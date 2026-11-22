# closeness_draws.py: how many candidates sit with the labelled copies (largest-gap cut, 20 draws keeping 8 answers)

## References (truth: plain 01, fp8 02, LoRA 0.3 09, 0.7 04, 1.5 07)
```
all answers: count 3 (ref-01 ref-02 ref-09); scores ref-01 -0.11, ref-02 +0.04, ref-09 +0.72, ref-04 +1.65, ref-07 +2.31
count over draws: 1: 1, 3: 16, 4: 3
share of draws before the cut: ref-01 1.00, ref-02 0.95, ref-09 0.95, ref-04 0.15, ref-07 0.00
```

## Scored families
```
=== family A
probes bat_choice, bat_trivia, idn_identity, txt_story, e1_leak_repeat, e2_leak_quote; 20 draws keeping 8 answers
all answers: count 4 (ep-12 ep-18 ep-24 ep-01); scores ep-12 +1.33, ep-18 +1.49, ep-24 +1.52, ep-01 +1.58, ep-25 +2.32, ep-17 +2.48
count over draws: 3: 2, 4: 16, 5: 2
share of draws before the cut: ep-12 1.00, ep-18 1.00, ep-01 0.95, ep-24 0.95, ep-17 0.05, ep-25 0.05
=== family B
probes bat_choice, bat_trivia, idn_identity, txt_story, d1_repeat150; 20 draws keeping 8 answers
all answers: count 2 (ep-22 ep-32); scores ep-22 +0.53, ep-32 +0.59, ep-26 +1.33, ep-03 +1.55, ep-21 +1.71, ep-28 +2.05
count over draws: 1: 2, 2: 7, 3: 8, 4: 2, 5: 1
share of draws before the cut: ep-22 0.95, ep-32 0.85, ep-26 0.65, ep-03 0.15, ep-21 0.05, ep-28 0.00
=== family C
probes bat_choice, bat_trivia, idn_identity, txt_story; 20 draws keeping 8 answers
all answers: count 2 (ep-19 ep-13); scores ep-19 -0.12, ep-13 +0.01, ep-15 +0.22, ep-09 +0.35, ep-11 +0.45, ep-05 +0.60
count over draws: 1: 3, 2: 3, 3: 3, 4: 4, 5: 7
share of draws before the cut: ep-13 0.90, ep-19 0.85, ep-09 0.70, ep-15 0.60, ep-11 0.30, ep-05 0.10
```
