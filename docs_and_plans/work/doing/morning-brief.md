# Morning brief, 22 November

Draft kept current through the night by the desk. Final version by 08:30. The event ends at 12:00; last submissions by 11:00.

## Decisions waiting for you

1. **Delete two unused Python environments on scratch?** `model_equality/.venv` (6,703 files) and `Hackathon_Watermark_Localisation_Task/.venv` (26,655 files). An organizer asked every team at about 01:35 to delete files: scratch ran out of file slots. They are not this session's to delete without your word. The unlearning environment stays; jobs use it.
2. **Ask an organizer which submission counts** for the final ranking on the hidden part: the last one or the best public one. Until we know, the last submission per task must be the one we believe in.
3. **Ask an organizer** whether `sbatch --chdir=<team folder>` from a login node that does not show scratch counts as "from inside the team folder". Tonight only one login node of four showed scratch after JSC's reboots.

## Scores

| Task | Public score | Submitted | Board's best at 22:33 |
|---|---|---|---|
| Watermark | 0.7444 | c6, 23:36 | 0.75 |
| Model equality | 0.1 | family grouping, 01:30 | 1.00 |
| Unlearning | 0.066 | `k7_e3t2_rm.pt`, 01:38 | 0.40 |

Public board at 23:45: Begg 20th of 32 with 0.33; the 6th had 0.73. Not re-read since.

## What was spent

- **Submissions under the night authorization** (ends 09:00): model equality 2 of 5, unlearning 5 of 12, watermark 1 of 1.
- **Scored answers:** 10 per endpoint by 01:08 (reconnaissance), 18 after the identity run at about 02:01. Night cap 80, organizers' cap 100.
- **GPU:** jobs 15903485 (ended 00:28 at its limit), 15904049 (cancelled by the desk after 10 minutes, the batch was replaced), 15904082 (26 minutes), 15904237 (started 01:26). Never more than one job running or pending; team check empty before each `sbatch`.

## Unlearning submissions

| Time | File | Public |
|---|---|---|
| 23:37 | `x3_ext_m80.pt` | 0.0063 |
| 00:07 | `x4_e20.pt` | 0.0336 |
| 00:38 | `scale_t4.pt` | 0.0057 |
| 01:08 | `dial15.pt` | 0.0 |
| 01:38 | `k7_e3t2_rm.pt` | 0.066 |

## Model equality

- **Families: settled.** Three families of 11, found identically by the main line, by Devin and by the desk, and right on the public pairs (the 0.1 is the whole family share).
- **Inside the families (90% of the score): open.** To be filled: the two groupings, the pairs where they disagree, the command for the remaining answers.

## Incidents you should know

- **00:22, a test sent one request to the organizers' server** with a test key (HTTP 401). The script tests acted on the night authorization. Fixed in PR #31 (merged 00:37) with two regression tests. The stray record was never committed.
- **00:41, the JURECA link dropped.** JSC rebooted login nodes one by one during the night, and a rebooted node comes back without `/p/scratch`. You reopened the link at 00:54.
- **Changes on your laptop made by the desk:**
  - `~/.ssh/config`: reuse-only aliases for the other login nodes added; the previous file is `~/.ssh/config.pre-backup-aliases-20261122`.
  - `~/.config/euroai/jureca_relog.sh` and `ssh_askpass.sh`: one guarded reconnect attempt, with a log in `~/.config/euroai/jureca_relog.log`. Used once, as a test, on `jureca01`.
  - The key passphrase is in `~/.config/euroai/.env`, as you put it. Remove it after the event.
- **01:30, a model equality submission scored less than the desk expected.** The desk had the metric wrong in its head: the model part is computed inside families only, where "one model per family" scores zero. One of five night submissions went to a result worth 0.1, plus the confirmation of the families.
- **Scratch file quota.** Free slots fell from 11,800 to under 1,000 between 00:58 and 01:25. The desk created each job's output files in advance so that a full quota could not break a running job.

## Before 11:00

To be filled: the last submission per task, with the exact commands.

## Open, not urgent

- A fine-grained GitHub token for the second laptop, limited to this repository, instead of full access to your account.
- An old desk session may still be open in Orca (PID 74888 at 23:30). It confirmed it runs nothing.
- The judges' defense pack: `docs_and_plans/guides/judges-defense.md`.
