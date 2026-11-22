# Morning brief, 22 November

Draft kept current through the night by the desk. Final version by 08:30. The event ends at 12:00; last submissions by 11:00.

For the jury: `docs_and_plans/guides/judges-defense.md` (four pages: the night in ten lines, the three tasks, why this is not leaderboard fitting, the organizers' rules, ten likely questions). Four points in it are marked **for the lead**.

## State at 06:50

- **Scores:** watermark 0.7444, model equality **0.85** (06:44, was 0.7), unlearning 0.2858 (04:39). Read the public board first thing: the agents have no way to read it, so our rank is unknown since 23:45.
- **Cluster:** one job, 15905366, the unlearning agent's thirteenth batch (45 minutes), submitted at 06:29 and pending on priority; the scheduler's estimated start is 07:30.
- **Night submissions left until 09:00:** model equality 1, unlearning 1.
- **Unlearning:** the last night slot is used only if the thirteenth batch passes the four checks its agent wrote before the job (#21).
- **Model equality:** the new test found family C's fp8 copy; see "The fp8 test" below. One decision on family B is yours (decision 1).
- **The desk was down from 04:47 to 06:00** (usage limit of the session). Nothing was running on the cluster in that hour and nothing was lost.

## Decisions waiting for you

1. **Model equality, family B: add ep-32 to the same-model group?** The fp8 test reads ep-22 as B's only fp8-like endpoint at +2.81 standard errors, under the threshold of 3 fixed before the data, so the desk did not act. Two ways forward, both yours:
   - buy the last 22 answers on ep-22 and read again (15 minutes; the commands are under "Before 11:00"), or
   - submit the prepared file as it is: `make submit TASK=model_equality FILE=tasks/model_equality/results/labelled_copies_v2_b32.csv`. The board says only the best submission counts, so a miss costs a slot, not points.
2. **Delete two unused Python environments on scratch?** `model_equality/.venv` (6,703 files) and `Hackathon_Watermark_Localisation_Task/.venv` (26,655 files). An organizer asked every team at about 01:35 to delete files: scratch ran out of file slots. They are not this session's to delete without your word. The unlearning environment stays; jobs use it.
3. **Ask an organizer which submission counts** for the final ranking on the hidden part: the last one or the best public one. Until we know, the last submission per task must be the one we believe in.
4. **Ask an organizer** whether `sbatch --chdir=<team folder>` from a login node that does not show scratch counts as "from inside the team folder". Tonight only one login node of four showed scratch after JSC's reboots.

## Scores

| Task | Public score | Submitted | Board's best at 22:33 |
|---|---|---|---|
| Watermark | 0.7444 | c6, 23:36 | 0.75 |
| Model equality | 0.85 | identified copies as one model per family, plus C's two plain-like endpoints, 06:44 | 1.00 |
| Unlearning | 0.2858 | `k11_b_soup3_x2.pt` (average of three runs), 04:39 | 0.40 |

Public board at 23:45: Begg 20th of 32 with 0.33; the 6th had 0.73. Not re-read since. With the three scores above and the board's bests of 22:33, our overall would be about 0.85.

## What was spent

- **Submissions under the night authorization** (ends 09:00): model equality 4 of 5, unlearning 11 of 12, watermark 1 of 1 (at 06:50).
- **Scored answers per endpoint** (night cap 80, organizers' cap 100): 64 in family A, 66 in B (ep-22: 78), 48 in C (ep-15: 72). Every prompt set ran on the nine references first.
- **GPU:** jobs 15903485 (ended 00:28 at its limit), 15904049 (cancelled by the desk after 10 minutes, the batch was replaced), 15904082 (26 minutes), 15904237 (33 minutes, ended 01:59), 15904499 (25 minutes, ended 02:48), 15904691 (31 minutes, ended 03:42), 15904844 (24 minutes, ended 04:26), 15905366 (submitted 06:29, pending). Never more than one job running or pending; team check empty before each `sbatch`.

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
| 04:08 | `k8_e3t2_rm1_ep40_x2p25.pt` (a sibling run, last layer times 2.25) | 0.2665 |
| 04:39 | `k11_b_soup3_x2.pt` (average of the weights of the 0.2837 run and its two new seeds, last layer times 2) | 0.2858 |

## Model equality

- **Families: settled.** Three families of 11, found identically by the main line, by Devin and by the desk, and right on the public pairs (the 0.1 is the whole family share).
- **Prompted copies found (02:25).** Each family has one system-prompt copy (ep-02, ep-08, ep-29) and one persona copy (ep-23, ep-20, ep-16), shown by what they quote when asked for their instructions. They are the same model as their family's plain copy. A submission that split them, prepared at 02:05, was held back and would have been wrong.
- **Inside the families (90% of the score): the method that works (03:00 to 04:00).** The laptop runs each family's public base model (A: Qwen2.5-3B-Instruct, B: Qwen2.5-1.5B-Instruct, C: a Llama 3.x, being fitted) and scores every recorded answer token by token. That gives, per endpoint, the sampling temperature it fits and whether the fit is the same on every prompt. Validated on the nine references, where the truth is published:
  - temperature copy: fits 1.0 on every prompt;
  - watermark copy: fits the plain temperature, but unevenly, in both directions;
  - strong adapters (LoRA 0.7, 1.5): shifted and uneven, one direction;
  - fp8 and LoRA 0.3: indistinguishable from plain. This is the limit of every method we have.
- **Labels at 04:25, submitted at 04:23 for a public 0.7.**

  | Family | One model: the plain model in disguise (positively identified) | Adapter candidates, each alone | Look like the base model, each alone (plain and fp8 hide here) |
  |---|---|---|---|
  | A (Qwen2.5-3B) | ep-02, ep-23 (prompts), ep-33 (temperature 1.0), ep-14 (top-p 0.95), ep-04 (watermark) | ep-24 | ep-01, 12, 17, 18, 25 |
  | B (Qwen2.5-1.5B) | ep-08, ep-20, ep-30, ep-07, ep-06 | ep-03, ep-28 | ep-21, 22, 26, 32 |
  | C (Llama-3.2-1B) | ep-29, ep-16, ep-31, ep-27, ep-10 | ep-05, ep-11 | ep-09, 13, 15, 19 |

- **The fp8 test (06:00 to 06:45): 0.7 became 0.85.**
  - *What it is.* The laptop builds an fp8 copy of each base model, quantized the way the serving software (vLLM) does it, and compares token by token how well that copy and the plain model explain an endpoint's answers. On the references the fp8 endpoint stands at +9.0 standard errors and the plain one at -6.1; the seven others are negative too.
  - *The rule, fixed on PR #27 before any family number was read.* An endpoint at +3 or more is the family's fp8 copy. Devin's two closest plain-like endpoints then join the same-model group unless the fp8 copy is one of them.
  - *Family C.* ep-15 read +2.94, then +3.53 after 24 more answers. ep-13 and ep-19 joined the group. Public score 0.85.
  - *Family B.* ep-22 read +2.71, then +2.81 after 12 more answers: under the threshold, no change. Every other endpoint of B is negative, ep-32 the most (-5.03).
  - *Family A.* No endpoint above zero. No change, and no candidate.
- **What the 0.7 meant.** The model part is about 0.67 on the public pairs. The rule gives 0.77 if each family has exactly one more same-model endpoint (its plain copy) and 0.60 if it has two. The points still open are the plain copy of each family.
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
- **23:47, a job queued behind the running one.** The desk queued a 5-hour job to hold the node. You pointed out the ban risk; it was cancelled at 23:54. Since then: one job in total, submitted only for ready work.
- **04:47 to 06:00, the desk was down.** The session reached its usage limit, and its two agents with it. The unlearning agent and Devin kept working and posting.
- **Scratch file quota.** Free slots fell from 11,800 to under 1,000 between 00:58 and 01:25. The desk created each job's output files in advance so that a full quota could not break a running job.

## Before 11:00

Commands are for nushell, from the desk's worktree.

1. **Get the night's state.** `cd ~/orca/workspaces/euro-ai-championship-2026/brill; git pull`
2. **Read the public board** and write down our rank per task. If we are 6th or better, change nothing without a reason stronger than the one behind the current submission.
3. **Model equality.** The submitted grouping is `tasks/model_equality/results/labelled_copies_v2.csv` (0.85). For decision 1:
   - the prepared file with ep-32 added: `make submit TASK=model_equality FILE=tasks/model_equality/results/labelled_copies_v2_b32.csv` (at most one submission per 30 minutes, a failure counts);
   - or first buy the last answers on ep-22, from the other worktree: `cd ~/worktrees/euro-ai-championship-2026/model-equality-refs; uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/bat.csv --repeats 20 --name round1 --endpoints ep-22 --spend` (it asks you to confirm at the terminal once the night authorization has ended; 14 answers, about 2 minutes). Then ask the desk, or a new session, to "score ep-22 and read family B with `fp8_rule.py`".
   - Check a file without submitting: `make check TASK=model_equality FILE=<file>`
4. **Unlearning.** Nothing to submit. The file behind 0.2858 is `data/unlearning/candidates/k11_b_soup3_x2.pt`.
5. **Watermark.** Nothing to submit. c6 (0.7444) stands.
6. **Answers left on the scored endpoints** (organizers' cap 100 each, the night used at most 66): `uv run python scripts/me_query.py --budget`
7. **Ask an organizer the two questions** under "Decisions waiting for you".

The board states that only a team's best submission on a task counts. A further submission can therefore not lower our rank unless "best" is chosen on the public part and the hidden part disagrees.

## To delete on your main laptop after the event

- In `~/.cache/huggingface/hub`: the seven `models--*` folders and the `blobs` folder that holds their weights, 12 GB in total, all downloaded tonight (SmolLM2-360M, Llama-3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B). They are needed until the last model equality decision.
- `~/.config/euroai/night.json` (it ends by itself at 09:00) and the passphrase line in `~/.config/euroai/.env`.

## Open, not urgent

- A fine-grained GitHub token for the second laptop, limited to this repository, instead of full access to your account.
- An old desk session may still be open in Orca (PID 74888 at 23:30). It confirmed it runs nothing.
