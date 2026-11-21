# After the talk: first hour

## Needs a human (account / credential actions)
- [ ] Confirm JuDoor MFA is enabled (SSH requires TOTP).
- [ ] Generate `~/.ssh/id_ed25519_jsc` with a passphrase; upload in JuDoor with `from="<ranges>"` for the venue network (+ home/university if needed). Wait ≤15 min.
- [ ] Fill `<JUDOOR_USERNAME>` and append `jsc/ssh_config.template` to `~/.ssh/config`; `ssh jureca` (passphrase + TOTP).

## Confirm with organizers
- [ ] Task, data, metric, submission format/endpoint, rules, deadlines, judging → fill README "Challenge — TBD".
- [ ] Reservation name; leaderboard reachability from compute nodes; personal data in dataset?

## On the cluster
- [ ] `jutil user projects` → budget; `jutil env activate -p training2666`; `mkdir -p $PROJECT/$USER`.
- [ ] rsync repo; Option A `uv sync --extra dev --extra torch` on the login node.
- [ ] Render sbatch, `mkdir -p logs`, submit the smoke job; confirm 4 GPUs + `cuda_available: true` in the log.

## If watermarking is confirmed (talk signal, 2026-11-21)
- [ ] Answer the watermark questions in NOTES.md "Signals from the opening talk" (role, media, white/black-box, metric).
- [ ] Re-read `papers/briefs/2512.16874_pixel-seal.md` §6–§8 and `guides/baseline-options.md` §C.
- [ ] On a login node: `uv sync --extra dev --extra vision`; try a Git install of videoseal; fetch any allowed checkpoint into `$SCRATCH`.

## Then
- [ ] Choose project; write `docs_and_plans/work/doing/<project>.md` plan (task, baseline ladder from `guides/baseline-options.md`, eval protocol, compute budget).
- [ ] Trivial baseline in the exact submission format first.
