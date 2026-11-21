# Model equality: reference experiments, lab notebook

Started 2026-11-21 22:40 Paris. Only `ref-01`..`ref-09` were queried, always through `scripts/me_query.py` without `--endpoints`. Truth (`upstream/practice_answers.csv`): S = ref-01/03/05/06/08 (plain, system prompt, persona, watermark, temperature 1.0), T1 = ref-02 (fp8), T2/T3/T4 = ref-09/04/07 (LoRA 0.3/0.7/1.5).

## E0. Throughput (measured during the first run)

- Hypothesis: the script delivers well over 60 answers per minute, so the reference budget is time, not answers.
- Command: `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/cand1.jsonl --repeats 6` (20 prompts, built by `probes/build_cand1.py`).
- Measured: the answer file grew from 4 to 12 lines in 50 s at 22:44, that is about 10 to 12 answers per minute with 4 parallel requests and mostly 256-token answers.
- Conclusion: wrong. The server is the bottleneck. At this rate a 40-answer round on 33 endpoints (1,320 answers) takes about 2 hours and the full 3,300 about 5 hours. The reference work is limited by wall time to a few hundred answers per experiment, not by the 15,000 cap.
