# Battery rows at 10 per scored endpoint (round1.jsonl at 69f3fb3), second method

## clique.py, bat_choice + bat_trivia, alpha 0.01 (frozen at ac91a9f)
```
probes bat_choice, bat_trivia; alpha 0.01; left out (too few answers): none
## family A: 11 endpoints, largest group 6, 4 such group(s), 19 of 55 pairs rejected
  group: ep-01 ep-12 ep-14 ep-17 ep-18 ep-33 | outside: ep-02 ep-04 ep-23 ep-24 ep-25
  group: ep-01 ep-12 ep-14 ep-18 ep-25 ep-33 | outside: ep-02 ep-04 ep-17 ep-23 ep-24
  group: ep-12 ep-14 ep-17 ep-18 ep-24 ep-33 | outside: ep-01 ep-02 ep-04 ep-23 ep-25
  group: ep-12 ep-14 ep-18 ep-24 ep-25 ep-33 | outside: ep-01 ep-02 ep-04 ep-17 ep-23
  rejected ep-02 ep-17: p 0.0000, z 10.55
  rejected ep-02 ep-25: p 0.0000, z 12.29
  rejected ep-17 ep-23: p 0.0000, z 10.18
  rejected ep-23 ep-24: p 0.0000, z 9.68
  rejected ep-02 ep-04: p 0.0001, z 12.62
  rejected ep-18 ep-23: p 0.0001, z 9.86
  rejected ep-01 ep-23: p 0.0001, z 9.28
  rejected ep-23 ep-25: p 0.0001, z 6.87
  rejected ep-02 ep-24: p 0.0002, z 10.04
  rejected ep-04 ep-23: p 0.0003, z 8.84
  rejected ep-04 ep-17: p 0.0005, z 6.09
  rejected ep-04 ep-25: p 0.0005, z 8.50
  rejected ep-01 ep-02: p 0.0006, z 8.66
  rejected ep-04 ep-12: p 0.0006, z 5.68
  rejected ep-17 ep-25: p 0.0010, z 5.07
  rejected ep-02 ep-18: p 0.0019, z 6.59
  rejected ep-01 ep-24: p 0.0024, z 4.66
  rejected ep-01 ep-04: p 0.0039, z 5.02
  rejected ep-14 ep-23: p 0.0055, z 4.99
## family B: 11 endpoints, largest group 9, 1 such group(s), 11 of 55 pairs rejected
  group: ep-03 ep-07 ep-08 ep-20 ep-21 ep-22 ep-26 ep-30 ep-32 | outside: ep-06 ep-28
  rejected ep-06 ep-21: p 0.0005, z 6.93
  rejected ep-06 ep-26: p 0.0006, z 7.45
  rejected ep-06 ep-28: p 0.0007, z 7.72
  rejected ep-03 ep-06: p 0.0008, z 8.31
  rejected ep-06 ep-20: p 0.0009, z 8.18
  rejected ep-06 ep-22: p 0.0010, z 8.44
  rejected ep-26 ep-28: p 0.0044, z 4.00
  rejected ep-28 ep-32: p 0.0061, z 3.89
  rejected ep-20 ep-28: p 0.0063, z 3.88
  rejected ep-08 ep-28: p 0.0078, z 3.98
  rejected ep-06 ep-32: p 0.0091, z 4.42
## family C: 11 endpoints, largest group 8, 2 such group(s), 6 of 55 pairs rejected
  group: ep-09 ep-11 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 | outside: ep-05 ep-10 ep-31
  group: ep-09 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 ep-31 | outside: ep-05 ep-10 ep-11
  rejected ep-09 ep-10: p 0.0007, z 5.45
  rejected ep-10 ep-16: p 0.0022, z 4.81
  rejected ep-10 ep-29: p 0.0037, z 4.31
  rejected ep-05 ep-15: p 0.0043, z 4.41
  rejected ep-05 ep-29: p 0.0060, z 4.10
  rejected ep-11 ep-31: p 0.0073, z 4.07
```

## rank.py (frozen at e3771c1), prompted endpoints excluded

## family A: features used bat_choice, bat_trivia, e1_leak_repeat, e2_leak_quote, idn_identity
| rank | endpoint | combined | bat_choice | bat_trivia | e1_leak_repeat | e2_leak_quote | idn_identity |
|---|---|---|---|---|---|---|---|
| 1 | ep-17 | 0.60 | 2.57 | -0.48 | 1.31 | 1.44 | 0.03 |
| 2 | ep-04 | 0.51 | 3.90 | -0.82 | 6.96 | 0.57 | -1.05 |
| 3 | ep-25 | 0.37 | 3.61 | -0.27 | 1.04 | 0.18 | -0.09 |
| 4 | ep-12 | 0.27 | 1.00 | 0.22 | 1.28 | 0.28 | 0.02 |
| 5 | ep-01 | 0.03 | 2.21 | -0.20 | 0.69 | -0.12 | -0.00 |
| 6 | ep-24 | -0.22 | 1.80 | -0.23 | 1.09 | 0.55 | -1.16 |
| 7 | ep-33 | -0.45 | 0.71 | -0.59 | 1.26 | 0.51 | -0.71 |
| 8 | ep-14 | -0.54 | 1.03 | -0.82 | 0.68 | -0.53 | 0.44 |
| 9 | ep-18 | -0.57 | 1.56 | -0.90 | 1.31 | -0.07 | -0.43 |

## family B: features used bat_choice, bat_trivia, idn_identity, stop
| rank | endpoint | combined | bat_choice | bat_trivia | idn_identity | stop |
|---|---|---|---|---|---|---|
| 1 | ep-28 | 0.87 | 0.95 | 2.39 | 0.73 | 0.90 |
| 2 | ep-06 | 0.69 | -0.88 | 6.95 | 2.05 | 0.50 |
| 3 | ep-26 | 0.26 | 0.97 | 0.82 | 0.20 | 0.70 |
| 4 | ep-22 | 0.02 | -0.20 | 1.32 | -0.18 | 0.95 |
| 5 | ep-21 | -0.14 | 0.01 | 1.20 | -0.47 | 0.85 |
| 6 | ep-32 | -0.27 | 0.30 | 0.58 | 0.06 | 0.55 |
| 7 | ep-03 | -0.32 | -0.70 | 1.70 | -0.02 | 0.75 |
| 8 | ep-07 | -0.38 | 0.28 | -0.31 | -0.42 | 0.70 |
| 9 | ep-30 | -0.73 | -0.33 | 1.38 | 0.24 | 0.25 |

## family C: features used bat_choice, bat_trivia, idn_identity
| rank | endpoint | combined | bat_choice | bat_trivia | idn_identity |
|---|---|---|---|---|---|
| 1 | ep-10 | 0.90 | 1.47 | 0.76 | -0.12 |
| 2 | ep-15 | 0.22 | 0.44 | 0.50 | -0.14 |
| 3 | ep-31 | -0.01 | 0.21 | 2.05 | -1.25 |
| 4 | ep-13 | -0.08 | -0.73 | 1.28 | -0.11 |
| 5 | ep-19 | -0.10 | -0.23 | 0.57 | -0.10 |
| 6 | ep-05 | -0.10 | -0.26 | 2.28 | -1.15 |
| 7 | ep-11 | -0.14 | -0.54 | 1.36 | -0.42 |
| 8 | ep-27 | -0.29 | 0.28 | -0.53 | -0.14 |
| 9 | ep-09 | -0.40 | -0.10 | 1.50 | -1.24 |

