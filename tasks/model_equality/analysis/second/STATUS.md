# Devin: status and next steps (read this first after a restart or a context compaction)

Updated 2026-11-22 01:12 Paris. Report to the desk: description of draft PR #27 (keep it current with
`gh pr view 27 --json body --jq .body > /tmp/devin-pr.md`, edit, `gh pr edit 27 --body-file /tmp/devin-pr.md`).

## Who and where

- Role: Devin, second independent model-equality method. Brief: `docs_and_plans/work/doing/devin-model-equality-second-method.md`.
- Worktree `~/worktrees/euro-ai-championship-2026/devin-model-equality-second`, branch `devin/model-equality-second`, code in `tasks/model_equality/analysis/second/`. Never touch the main checkout, `kkrgp/*`, `/Volumes/*`.
- Channels: desk comments on PR #27 (answer there, prefix `Devin:`), status and roll call #19, research #28, shared Mac #24.
- Never: query endpoints, spend, submit, JURECA, GPU, edit `answers/`, push outside `devin/*`, read `analysis/group.py` or the main line's plurality/recon/stop code. No AI attribution in commits, PRs or comments. `make lint && make test` before each commit, push after every result.
- Stop rules: no scored answers at all by 03:00 -> finish reference work and stop; 07:00 -> stop new work, bring #27 up to date.

## Loop

Every ~10 minutes: `git fetch origin -q`; if `origin/ihibti/model-equality-refs` moved, `git merge --no-edit origin/ihibti/model-equality-refs`; look for new answers files or new `ep-*` rows; read new `Desk:` comments on #27, #19, #28; act; commit; push; update #27.

## Open asks from the desk (00:27 and 00:46 comments on #27)

1. **Map of the 33 endpoints by about 02:00**, once the reconnaissance rows (`ep-*`, 5 round-1 prompts x 2) are in `answers/round1.jsonl` on the main line's branch:
   `cd tasks/model_equality/analysis/second; uv run python recon_map.py --prefix ep- --answers ../../answers/round1.jsonl --out recon_ep`
   then read `recon_ep.md`, write the families with evidence, visible disguises and look-alikes, commit, post on #27. Do not read the main line's map.
2. **P3 confirmation on the references** (`night_d`, id `d1_repeat150`), with the rule frozen at `897fcbb`:
   `uv run python p3rule.py --answers ../../answers/<night-d file>.jsonl --prefix ref- --probe d1_repeat150`
3. **night_c on the references** (`c1_equiv`, `c2_edge`, `c3_next`): `noise.py --probes night --answers ...` and `group2.py --probes night --prefix ref- ...`.
4. **Scored P3**: `p3rule.py --prefix ep- --probe <id> --families recon_ep.csv --answers ...`, then compare with the main line's grouping (`compare.py`).

## Results so far (all pushed)

- Round 1 refs (40 answers): batteries + average linkage MCC 0.4947, fragile (subsample mean 0.072); sequences measure disguises. `refs_round1.md`, `noise_round1.md`.
- Simulators: `simulate.md` (bootstrap affine test miscalibrated, permutation calibrated), `simulate_night.md` (mode statistic calibrated and temperature-robust).
- night-a refs: MCC 0; p1 separates the watermark copy ref-06; fp8 invisible. Desk: P1 and P2 get no scored answers.
- night-b refs: p3_repeat separates LoRA from all disguises; grouping MCC 0.6202; frozen P3 rule MCC 0.7338 (only fp8 pairs wrong).
- Sample size for P3: `p3_power.md` (10 answers: power 0.945 at plain 0.1 / LoRA 0.6; 15 needed at plain 0.2).
- D1 research delivered on #28 (`research_night.md`).
