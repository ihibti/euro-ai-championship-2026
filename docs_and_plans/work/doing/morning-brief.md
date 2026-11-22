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
| Unlearning | 0.2837 | `k8_e3t2_rm1_lr1e4_x2.pt`, 02:38 | 0.40 |

Public board at 23:45: Begg 20th of 32 with 0.33; the 6th had 0.73. Not re-read since.

## What was spent

- **Submissions under the night authorization** (ends 09:00): model equality 2 of 5, unlearning 9 of 12, watermark 1 of 1 (at 04:04).
- **Scored answers per endpoint** (night cap 80, organizers' cap 100): 64 in family A, 66 in B, 48 in C once the story prompt finishes at about 04:13. Every prompt set ran on the nine references first.
- **GPU:** jobs 15903485 (ended 00:28 at its limit), 15904049 (cancelled by the desk after 10 minutes, the batch was replaced), 15904082 (26 minutes), 15904237 (33 minutes, ended 01:59), 15904499 (25 minutes, ended 02:48), 15904691 (31 minutes, ended 03:42), 15904844 (submitted 03:47, limit 45 minutes, pending). Never more than one job running or pending; team check empty before each `sbatch`.

## Unlearning submissions

| Time | File | Public |
|---|---|---|
| 23:37 | `x3_ext_m80.pt` | 0.0063 |
| 00:07 | `x4_e20.pt` | 0.0336 |
| 00:38 | `scale_t4.pt` | 0.0057 |
| 01:08 | `dial15.pt` | 0.0 |
| 01:38 | `k7_e3t2_rm.pt` | 0.066 |
| 02:08 | `k7_e3t2_rm_x2.pt` (the same checkpoint, last layer times 2) | 0.2133 |
| 02:38 | `k8_e3t2_rm1_lr1e4_x2.pt` (the same recipe at a higher learning rate, last layer times 2) | 0.2837 |
| 03:08 | `k8_e3t2_rm1_lr1e4_x2p5.pt` (the 0.2837 checkpoint, last layer times 2.5) | 0.2222 |
| 03:38 | `k9_e3t1_lr1e4_x1p25.pt` (a gentler recipe, last layer times 1.25) | 0.256 |

## Model equality

- **Families: settled.** Three families of 11, found identically by the main line, by Devin and by the desk, and right on the public pairs (the 0.1 is the whole family share).
- **Prompted copies found (02:25).** Each family has one system-prompt copy (ep-02, ep-08, ep-29) and one persona copy (ep-23, ep-20, ep-16), shown by what they quote when asked for their instructions. They are the same model as their family's plain copy. A submission that split them, prepared at 02:05, was held back and would have been wrong.
- **Inside the families (90% of the score): the method that works (03:00 to 04:00).** The laptop runs each family's public base model (A: Qwen2.5-3B-Instruct, B: Qwen2.5-1.5B-Instruct, C: a Llama 3.x, being fitted) and scores every recorded answer token by token. That gives, per endpoint, the sampling temperature it fits and whether the fit is the same on every prompt. Validated on the nine references, where the truth is published:
  - temperature copy: fits 1.0 on every prompt;
  - watermark copy: fits the plain temperature, but unevenly, in both directions;
  - strong adapters (LoRA 0.7, 1.5): shifted and uneven, one direction;
  - fp8 and LoRA 0.3: indistinguishable from plain. This is the limit of every method we have.
- **Labels at 04:00.**

  | Family | Same model as plain (positively identified) | Adapter candidate | Look like the base model (plain, fp8, weak adapters hide here) |
  |---|---|---|---|
  | A | ep-02, ep-23 (prompted), ep-33 (temperature 1.0), ep-14 (truncated sampling, top-p 0.95), ep-04 (watermark) | ep-24 | ep-01, 12, 17, 18, 25 |
  | B | ep-08, ep-20 (prompted), ep-30 (temperature 1.0), ep-07 (truncated sampling); watermark: ep-22 by likelihood, ep-06 by the batteries, open | ep-28 | ep-03, 21, 26, 32, and ep-06 or ep-22 |
  | C | ep-29, ep-16 (prompted); the rest pending | | |

- **Submission rule.** Per family, the positively identified copies form one model and every other endpoint is its own model. It never claims two endpoints are the same without evidence. Checked by Devin's code: it scores 0.77 on the model part if exactly one more endpoint is the plain copy, 0.60 or 0.46 if one or two further undetectable same-model copies hide among the rest, 0.51 if one of our labels is wrong.
- **Where the two analysts disagree.** ep-12 as A's plain copy: Devin 0.6, the main line 0.3 to 0.4; not added. ep-06 against ep-22 as B's watermark copy.
- **Devin's second method** (battery modes, cliques) and the main line's battery rules were frozen before each read. A split of family A proposed by the desk at 02:25 failed its pre-registered test at 02:58 and was never submitted.

## Incidents you should know

- **00:22, a test sent one request to the organizers' server** with a test key (HTTP 401). The script tests acted on the night authorization. Fixed in PR #31 (merged 00:37) with two regression tests. The stray record was never committed.
- **00:41, the JURECA link dropped.** JSC rebooted login nodes one by one during the night, and a rebooted node comes back without `/p/scratch`. You reopened the link at 00:54.
- **Changes on your laptop made by the desk:**
  - `~/.ssh/config`: reuse-only aliases for the other login nodes added; the previous file is `~/.ssh/config.pre-backup-aliases-20261122`.
  - `~/.config/euroai/jureca_relog.sh` and `ssh_askpass.sh`: one guarded reconnect attempt, with a log in `~/.config/euroai/jureca_relog.log`. Used once, as a test, on `jureca01`.
  - The key passphrase is in `~/.config/euroai/.env`, as you put it. Remove it after the event.
- **01:30, a model equality submission scored less than the desk expected.** The desk had the metric wrong in its head: the model part is computed inside families only, where "one model per family" scores zero. One of five night submissions went to a result worth 0.1, plus the confirmation of the families.
- **Disk on your main laptop.** The local base models were downloaded from Hugging Face into `~/.cache/huggingface` (free disk fell from 87 GB to 27 GB at the lowest; 59 GB free at 04:00 after the ruled-out models were deleted). The models kept are listed at the end of the night for you to delete.
- **Scratch file quota.** Free slots fell from 11,800 to under 1,000 between 00:58 and 01:25. The desk created each job's output files in advance so that a full quota could not break a running job.

## Before 11:00

To be filled: the last submission per task, with the exact commands.

## Open, not urgent

- A fine-grained GitHub token for the second laptop, limited to this repository, instead of full access to your account.
- An old desk session may still be open in Orca (PID 74888 at 23:30). It confirmed it runs nothing.
- The judges' defense pack: `docs_and_plans/guides/judges-defense.md`.
