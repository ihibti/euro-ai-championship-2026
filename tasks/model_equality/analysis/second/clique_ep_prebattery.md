# clique.py on the scored endpoints before the battery rows (frozen at ac91a9f)

## night-d: d1_repeat150 (min 3 answers)
```
probes d1_repeat150; alpha 0.01; left out (too few answers): ep-01 ep-02 ep-04 ep-05 ep-09 ep-10 ep-11 ep-12 ep-13 ep-14 ep-15 ep-16 ep-17 ep-18 ep-19 ep-23 ep-24 ep-25 ep-27 ep-29 ep-31 ep-33
## family B: 11 endpoints, largest group 7, 1 such group(s), 19 of 55 pairs rejected
  group: ep-03 ep-06 ep-07 ep-21 ep-26 ep-28 ep-32 | outside: ep-08 ep-20 ep-22 ep-30
  rejected ep-03 ep-08: p 0.0005, z 5.63
  rejected ep-03 ep-20: p 0.0005, z 10.00
  rejected ep-03 ep-22: p 0.0005, z 6.56
  rejected ep-03 ep-30: p 0.0005, z 5.88
  rejected ep-08 ep-21: p 0.0005, z 5.80
  rejected ep-20 ep-21: p 0.0005, z 10.20
  rejected ep-20 ep-28: p 0.0005, z 13.31
  rejected ep-21 ep-22: p 0.0005, z 6.65
  rejected ep-21 ep-30: p 0.0005, z 6.33
  rejected ep-22 ep-28: p 0.0005, z 4.10
  rejected ep-28 ep-30: p 0.0005, z 4.88
  rejected ep-20 ep-26: p 0.0010, z 4.57
  rejected ep-06 ep-08: p 0.0020, z 4.48
  rejected ep-08 ep-28: p 0.0025, z 3.86
  rejected ep-08 ep-26: p 0.0035, z 4.31
  rejected ep-26 ep-30: p 0.0045, z 4.10
  rejected ep-30 ep-32: p 0.0060, z 3.39
  rejected ep-06 ep-30: p 0.0070, z 3.98
  rejected ep-06 ep-20: p 0.0075, z 2.49
```

## round1: idn_identity (min 3 answers)
```
probes idn_identity; alpha 0.01; left out (too few answers): none
## family A: 11 endpoints, largest group 9, 1 such group(s), 18 of 55 pairs rejected
  group: ep-01 ep-04 ep-12 ep-14 ep-17 ep-18 ep-24 ep-25 ep-33 | outside: ep-02 ep-23
  rejected ep-01 ep-02: p 0.0005, z 9.17
  rejected ep-01 ep-23: p 0.0005, z 11.21
  rejected ep-02 ep-04: p 0.0005, z 10.37
  rejected ep-02 ep-12: p 0.0005, z 9.40
  rejected ep-02 ep-17: p 0.0005, z 9.23
  rejected ep-02 ep-18: p 0.0005, z 8.60
  rejected ep-02 ep-24: p 0.0005, z 8.18
  rejected ep-02 ep-25: p 0.0005, z 9.68
  rejected ep-02 ep-33: p 0.0005, z 10.54
  rejected ep-04 ep-23: p 0.0005, z 12.09
  rejected ep-12 ep-23: p 0.0005, z 11.78
  rejected ep-14 ep-23: p 0.0005, z 10.92
  rejected ep-17 ep-23: p 0.0005, z 11.46
  rejected ep-18 ep-23: p 0.0005, z 10.44
  rejected ep-23 ep-24: p 0.0005, z 9.45
  rejected ep-23 ep-25: p 0.0005, z 11.61
  rejected ep-23 ep-33: p 0.0005, z 12.58
  rejected ep-02 ep-14: p 0.0010, z 10.00
## family B: 11 endpoints, largest group 8, 1 such group(s), 18 of 55 pairs rejected
  group: ep-03 ep-07 ep-21 ep-22 ep-26 ep-28 ep-30 ep-32 | outside: ep-06 ep-08 ep-20
  rejected ep-03 ep-20: p 0.0005, z 10.67
  rejected ep-07 ep-20: p 0.0005, z 7.53
  rejected ep-20 ep-21: p 0.0005, z 10.09
  rejected ep-20 ep-22: p 0.0005, z 7.43
  rejected ep-20 ep-26: p 0.0005, z 8.26
  rejected ep-20 ep-28: p 0.0005, z 8.92
  rejected ep-20 ep-30: p 0.0005, z 8.23
  rejected ep-20 ep-32: p 0.0005, z 8.27
  rejected ep-06 ep-20: p 0.0010, z 8.97
  rejected ep-08 ep-21: p 0.0010, z 2.37
  rejected ep-06 ep-21: p 0.0015, z 3.37
  rejected ep-06 ep-28: p 0.0030, z 3.99
  rejected ep-08 ep-28: p 0.0030, z 3.89
  rejected ep-06 ep-08: p 0.0035, z 4.24
  rejected ep-03 ep-08: p 0.0040, z 2.86
  rejected ep-08 ep-22: p 0.0040, z 3.84
  rejected ep-08 ep-20: p 0.0060, z 3.00
  rejected ep-07 ep-08: p 0.0075, z 3.45
## family C: 11 endpoints, largest group 11, 1 such group(s), 0 of 55 pairs rejected
  group: ep-05 ep-09 ep-10 ep-11 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 ep-31 | outside: -
```

## night-e: e1_leak_repeat e2_leak_quote (min 10 answers)
```
probes e1_leak_repeat, e2_leak_quote; alpha 0.01; left out (too few answers): ep-03 ep-05 ep-06 ep-07 ep-08 ep-09 ep-10 ep-11 ep-13 ep-15 ep-16 ep-19 ep-20 ep-21 ep-22 ep-26 ep-27 ep-28 ep-29 ep-30 ep-31 ep-32
## family A: 11 endpoints, largest group 7, 2 such group(s), 16 of 55 pairs rejected
  group: ep-01 ep-12 ep-14 ep-18 ep-24 ep-25 ep-33 | outside: ep-02 ep-04 ep-17 ep-23
  group: ep-01 ep-12 ep-17 ep-18 ep-24 ep-25 ep-33 | outside: ep-02 ep-04 ep-14 ep-23
  rejected ep-02 ep-17: p 0.0002, z 8.74
  rejected ep-23 ep-24: p 0.0003, z 6.44
  rejected ep-23 ep-25: p 0.0003, z 6.57
  rejected ep-04 ep-18: p 0.0006, z 7.79
  rejected ep-04 ep-24: p 0.0007, z 11.72
  rejected ep-04 ep-25: p 0.0008, z 10.87
  rejected ep-12 ep-23: p 0.0009, z 5.40
  rejected ep-04 ep-33: p 0.0009, z 7.75
  rejected ep-04 ep-17: p 0.0011, z 7.87
  rejected ep-17 ep-23: p 0.0018, z 7.80
  rejected ep-23 ep-33: p 0.0021, z 5.48
  rejected ep-14 ep-17: p 0.0031, z 6.31
  rejected ep-02 ep-33: p 0.0032, z 4.26
  rejected ep-18 ep-23: p 0.0047, z 4.28
  rejected ep-04 ep-12: p 0.0050, z 6.74
  rejected ep-02 ep-25: p 0.0073, z 3.57
```

