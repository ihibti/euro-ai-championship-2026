# Story continuation prefixes (txt_story, 10 answers), clique.py at alpha 0.01

References (truth S 01 03 05 06-watermark 08, fp8 02, LoRA 09 04 07): nothing rejected, see noise_story_refs.md; the view carries no reference validation.

```
probes txt_story; alpha 0.01; left out (too few answers): none
## family A: 11 endpoints, largest group 11, 1 such group(s), 0 of 55 pairs rejected
  group: ep-01 ep-02 ep-04 ep-12 ep-14 ep-17 ep-18 ep-23 ep-24 ep-25 ep-33 | outside: -
## family B: 11 endpoints, largest group 11, 1 such group(s), 0 of 55 pairs rejected
  group: ep-03 ep-06 ep-07 ep-08 ep-20 ep-21 ep-22 ep-26 ep-28 ep-30 ep-32 | outside: -
## family C: 11 endpoints, largest group 9, 1 such group(s), 9 of 55 pairs rejected
  group: ep-05 ep-09 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 ep-31 | outside: ep-10 ep-11
  rejected ep-11 ep-19: p 0.0005, z 5.85
  rejected ep-10 ep-11: p 0.0010, z 4.43
  rejected ep-10 ep-15: p 0.0010, z 7.09
  rejected ep-11 ep-15: p 0.0010, z 2.50
  rejected ep-09 ep-10: p 0.0015, z 5.87
  rejected ep-10 ep-13: p 0.0015, z 6.87
  rejected ep-10 ep-19: p 0.0015, z 6.56
  rejected ep-11 ep-13: p 0.0030, z 3.28
  rejected ep-09 ep-11: p 0.0095, z 2.82
```

Family C, the three commonest 6-token openings per endpoint:

- ep-05: "The old lighthouse on the northern" 3; "The storm rolled in like a" 2; "The storm raged on , its" 1
- ep-09: "The old lighthouse on the northern" 7; "The darkness was palpable as the" 1; "As the dark clouds rolled in" 1
- ep-10: "As the dark sky twisted and" 7; "The old lighthouse on the northern" 2; "As the dark storm clouds gathered" 1
- ep-11: "The darkness seemed to swallow the" 1; "The darkness was palpable as the" 1; "The winds howled and the rain" 1
- ep-13: "The old lighthouse on the northern" 7; "The old lighthouse stood sentinel on" 1; "The storm brewed on the horizon" 1
- ep-15: "The old lighthouse on the northern" 8; "The wind howled through the empty" 1; "The storm had been brewing for" 1
- ep-16: "The old lighthouse on the northern" 3; "The storm brewed on the horizon" 1; ". . . the lighthouse '" 1
- ep-19: "The old lighthouse on the northern" 7; "As the storm clouds gathered on" 1; "The last rays of sunlight faded" 1
- ep-27: "The old lighthouse on the northern" 2; "The wind howled through the desolate" 2; "The darkness was absolute , a" 1
- ep-29: "The old lighthouse on the northern" 4; "The storm rolled in like a" 2; "The storm that rolled in on" 1
- ep-31: "The old lighthouse on the northern" 4; "It was as if the storm" 1; "As the dark storm clouds gathered" 1
