# Full-distribution battery test (round-1 MMD statistic) on plain against fp8, and on the scored tied pairs

## References: ref-01 (plain) against ref-02 (fp8), round 1 at 14 repeats (`e433767`)

- batteries (bat_choice + bat_trivia), 14 answers: z 4.63, p 0.0055 (bat_choice 2.35, bat_trivia 2.29); sequences: z -0.07, p 0.47; mode statistic on the batteries: not rejected.
- at 10 of the 14 answers (20 draws): median z 3.32 (0.95 to 5.73), p < 0.05 in 60% of draws, p < 0.10 in 75%.

## Scored candidates, 10 battery answers each (group2.py --probes round1 --prompts batteries)

### Family A
```
ep-12 ep-24 z 4.090 p 0.01330
ep-01 ep-24 z 3.957 p 0.01645
ep-14 ep-24 z 2.118 p 0.08180
ep-01 ep-18 z 1.946 p 0.08815
ep-01 ep-33 z 1.891 p 0.10750
ep-01 ep-12 z 1.634 p 0.12635
ep-12 ep-18 z 1.101 p 0.18750
ep-01 ep-14 z 0.946 p 0.22525
ep-18 ep-24 z 0.595 p 0.30450
ep-24 ep-33 z 0.195 p 0.39620
ep-14 ep-18 z 0.085 p 0.43010
ep-12 ep-33 z -0.048 p 0.45625
ep-18 ep-33 z -0.644 p 0.65930
ep-14 ep-33 z -1.349 p 0.84350
ep-12 ep-14 z -2.279 p 0.97005
```

### Family B
```
ep-22 ep-30 z 3.271 p 0.02390
ep-07 ep-22 z 1.653 p 0.12490
ep-22 ep-26 z 1.458 p 0.14875
ep-07 ep-26 z 1.311 p 0.17400
ep-07 ep-32 z 1.289 p 0.18245
ep-26 ep-30 z 0.648 p 0.27795
ep-22 ep-32 z 0.607 p 0.31450
ep-30 ep-32 z 0.375 p 0.35765
ep-26 ep-32 z -0.284 p 0.55740
ep-07 ep-30 z -1.592 p 0.87360
```

### Family C
```
ep-09 ep-31 z 4.387 p 0.00455
ep-19 ep-31 z 3.967 p 0.00885
ep-15 ep-31 z 3.159 p 0.02585
ep-09 ep-15 z 2.999 p 0.02655
ep-13 ep-31 z 2.835 p 0.03490
ep-15 ep-19 z 2.684 p 0.04105
ep-15 ep-27 z 2.627 p 0.04260
ep-27 ep-31 z 2.085 p 0.08360
ep-13 ep-15 z 0.893 p 0.24300
ep-09 ep-27 z 0.876 p 0.25435
ep-13 ep-27 z 0.720 p 0.27615
ep-19 ep-27 z -0.468 p 0.60385
ep-09 ep-19 z -0.813 p 0.69995
ep-13 ep-19 z -1.067 p 0.76310
ep-09 ep-13 z -1.346 p 0.82670
```
