# Morning brief, 22 November

Written by the desk through the night. Final at 07:20; anything later is added under "State". The event ends at 12:00; last submissions by 11:00.


The two other agents' final documents:
- Unlearning: the method write-up and the morning plan in `night.md`, on `kkrgp/unlearning-first-moves`, final at `538147e` (07:38); draft PR #32. Its closing note is on #12 (07:25): nothing to submit after 09:00.
- Devin (model equality, second method): the description of draft PR #27 is the final report; the fp8 rule and its readings are in the comments there.

## On arrival at the venue (written 08:25)

You said at 08:15: we are 8th, we need 6th, push something new. Model equality is at 1.0 and watermark is near the best, so unlearning is the only task with room. Both other agents were called at 08:19 (#12 and PR #27) and owe a ranked list on #12 by 09:15.

0. **A GPU job is running for this: 15905724**, submitted at 08:30, the unlearning agent's fourteenth batch (four runs at a lower target temperature, about 25 minutes, ends about 08:56). Its agent gives it 1 chance in 3 of a line more than 2% above 0.2858, and says nothing else on the laptop is likely to beat 0.2858. If the laptop was offline when the job ended, the results are still on the cluster: the desk pulls them when you wake it.
1. **Wake the desk:** type anything in its session once the laptop is online. It reopens the cluster link, pulls the results, reads #12 and tells you what to submit.
2. **If the desk does not answer, submit these yourself, in this order** (one unlearning submission per 30 minutes; the last was at 08:21; each asks you to type a confirmation). The order is the unlearning agent's ranked list on #12 (08:32); both files are built and checked:
   - First: `cd ~/orca/workspaces/euro-ai-championship-2026/brill; make submit TASK=unlearning FILE=data/unlearning/candidates/m14_t1_blend3_x1p5.pt`
     The agent's prediction: 0.29 to 0.31, about 2 chances in 5 of a gain the board registers.
   - Second, 30 minutes later, unless the first scored under 0.25: `make submit TASK=unlearning FILE=data/unlearning/candidates/m14_t15_blend_x1p74.pt` (about 1 in 4).
   - Third: the best line of the GPU batch, which the agent names on #12 once the reports are pulled. That needs the desk.
   - Spare, the desk's own line, which the agent rates lower: `data/unlearning/candidates/k9_e3t2_lr1e4_ep40_x2.pt`.
3. **After each submission:** `git add tasks/unlearning/submissions; git commit -m "Unlearning submission record"; git push`
4. **Tell the desk three numbers from the board:** the best score per task, and the overall scores of the 6th and 7th teams. They say how much unlearning must gain.
5. **Ask an organizer** whether the best public submission or the last one counts. The server's reply reads "Your previous best on this task still leads", which suggests the best. If it is the last, finish with `data/unlearning/candidates/k11_b_soup3_x2.pt` (0.2858) unless something scored higher.

Honest outlook: this axis looks worth a few percent on unlearning (towards 0.30), not a jump to the board's best.

## State at 08:10

- **Scores:** watermark 0.7444, model equality **1.0** (08:07; 0.85 at 06:44, 0.7 before), unlearning 0.2858 (04:39). Read the public board first thing: the agents have no way to read it, so our rank is unknown since 23:45.
- **Cluster:** no job running or pending. The last one, 15905366 (the unlearning agent's thirteenth batch), ran from 06:48 to 07:12 without a failed run.
- **Night submissions left until 09:00:** model equality 0, unlearning 1 (stays unused).
- **Unlearning:** the night ends at 0.2858. The last batch (a target mixed from three cross-fits) failed two of the four checks its agent had fixed before the job, so the agent left the last slot empty (#12, 07:15). No GPU request is open.
- **Model equality:** a new test found family C's fp8 copy, which gave the 0.85, and pointed to family B's, which gave the 1.0 on your go; see "Model equality" below. Nothing more to submit (decision 1).
- **The desk was down a second time, between 07:39 and 08:04** (the session exited while idle, and the four connections to JURECA dropped in the same window; cause unknown, the laptop did not reboot). Nothing was running or owed. The connections are still down and are not needed: no job is running or planned.
- **The desk was down from 04:47 to 06:00** (usage limit of the session). Nothing was running on the cluster in that hour and nothing was lost.

## Decisions waiting for you

1. **Model equality: done, submit nothing more.** You said go at 08:06; the file with ep-32 added to family B's group scored **1.0** on the public part at 08:07.
   - *Do not submit the second prepared file* (`labelled_copies_v2_b26_b32.csv`) or any other grouping. At 1.0 every public pair is right. A further addition can no longer be confirmed by the board: a drop would refute it, and no change would prove nothing, because the score also stays at 1.0 when none of the endpoint's pairs is in the public part.
   - *What the 1.0 says about the hidden part.* A same-model endpoint that we still keep alone would put 5 or 6 wrong pairs in its family. If the public part is about 30% of the pairs (the team's working figure), the chance that none of them is public is about 1 in 6. Rough reading: about 1 chance in 5 that family A still hides one such endpoint, about 1 in 10 for ep-26 in B. The hidden score should be close to the public one, not guaranteed equal.
   - *For the record:* the fifth submission was a single endpoint named by a test that stopped just under its own threshold, submitted on your decision. The methods summary says so (section 3).
2. **Delete two unused Python environments on scratch?** `model_equality/.venv` (6,703 files) and `Hackathon_Watermark_Localisation_Task/.venv` (26,655 files). An organizer asked every team at about 01:35 to delete files: scratch ran out of file slots. They are not this session's to delete without your word. The unlearning environment stays; jobs use it.
3. **Which submission counts** for the final ranking: the last one or the best public one? The server now answers each submission with "the final ranking uses the held out subset". On all three tasks our last submission is also our best, so the question matters only if someone submits again. Worth asking an organizer before any further submission.
4. **Ask an organizer** whether `sbatch --chdir=<team folder>` from a login node that does not show scratch counts as "from inside the team folder". Tonight only one login node of four showed scratch after JSC's reboots.

## Scores

| Task | Public score | Submitted | Board's best at 22:33 |
|---|---|---|---|
| Watermark | 0.7444 | c6, 23:36 | 0.75 |
| Model equality | 1.0 | identified copies as one model per family, plus C's two plain-like endpoints (06:44, 0.85) and B's ep-32 (08:07) | 1.00 |
| Unlearning | 0.2858 | `k11_b_soup3_x2.pt` (average of three runs), 04:39 | 0.40 |

Public board at 23:45: Begg 20th of 32 with 0.33; the 6th had 0.73. Not re-read since. With the three scores above and the board's bests of 22:33, our overall would be about 0.90.

## What was spent

- **Submissions under the night authorization** (ends 09:00): model equality 5 of 5, unlearning 12 of 12, watermark 1 of 1 (at 08:22).
- **Scored answers per endpoint** (night cap 80, organizers' cap 100): 64 in family A, 66 in B (ep-22: 78), 48 in C (ep-15: 72). Every prompt set ran on the nine references first.
- **GPU:** jobs 15903485 (ended 00:28 at its limit), 15904049 (cancelled by the desk after 10 minutes, the batch was replaced), 15904082 (26 minutes), 15904237 (33 minutes, ended 01:59), 15904499 (25 minutes, ended 02:48), 15904691 (31 minutes, ended 03:42), 15904844 (24 minutes, ended 04:26), 15905366 (24 minutes, ended 07:12). Never more than one job running or pending; team check empty before each `sbatch`.

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

After the 0.2858, two more batches were prepared and nothing more was submitted:
- *Twelfth* (more seeds of the same recipe): withdrawn by the agent before it ran, because the seeds are the same model where the score is measured.
- *Thirteenth* (job 15905366, 06:48 to 07:12, a target mixed from three independent cross-fits): the three cross-fits agree on 95% of the images, so the mix adds little, and the result sits between the 0.2133's level and the 0.2837's. Checks 1 and 3 of four failed. A larger last-layer factor would move it closer, but that is a dial chosen after the result, so it was not tried.

## Model equality

- **Families: settled.** Three families of 11, found identically by the main line, by Devin and by the desk, and right on the public pairs (0.1 of the score).
- **Inside the families (0.9 of the score), how each copy of the plain model was found.** Every method ran first on the nine free references, where the truth is published.
  - *Prompted copies* (02:25): asked for their instructions, one copy per family quotes a fixed system prompt and one gives a persona.
  - *Temperature, top-p and watermark copies* (03:00 to 04:20): the laptop runs each family's public base model (A: Qwen2.5-3B-Instruct, B: Qwen2.5-1.5B-Instruct, C: Llama-3.2-1B-Instruct) and scores every answer token by token. That gives the sampling temperature each endpoint fits, the tokens a truncated sampler could not have produced, and the watermark's uneven fit.
  - *fp8 copy* (06:00 to 06:45): the laptop builds an fp8 copy of each base model, quantized the way the serving software (vLLM) does it, and compares how well that copy and the plain model explain an endpoint's answers. On the references the fp8 endpoint stands at +9.0 standard errors and the eight others between -2.6 and -6.7.
- **The fp8 rule, fixed on PR #27 before any family number was read.** An endpoint at +3 or more is the family's fp8 copy. Devin's two closest plain-like endpoints then join the same-model group, unless the fp8 copy is one of them.
  - *Family C.* ep-15 read +2.94, then +3.53 after 24 more answers. ep-13 and ep-19 joined the group: public 0.7 became 0.85.
  - *Family B.* ep-22 read +2.71, then +2.81 after 12 more answers: under the threshold, so the desk made no change. Every other endpoint of B is negative, except the top-p copy ep-07 at +0.78. At 08:07, on the lead's go, ep-32 (the other member of Devin's closest pair) joined the group: public 0.85 became 1.0.
  - *Family A.* No endpoint above zero. No change, and no candidate.
- **Submitted at 08:07, public 1.0:** `tasks/model_equality/results/labelled_copies_v2_b32.csv` (sha256 `c0d8a1a4…`). Before it, at 06:44, public 0.85: `labelled_copies_v2.csv`.

  | Family | One model: the plain model and its disguises | fp8 copy, alone | Adapter marks, each alone | Plain-like, each alone |
  |---|---|---|---|---|
  | A (Qwen2.5-3B) | ep-02, 23 (prompts), 33 (temperature 1.0), 14 (top-p 0.95), 04 (watermark) | not found | ep-24 | ep-01, 12, 17, 18, 25 |
  | B (Qwen2.5-1.5B) | ep-08, 20, 30, 07, 06, and since 08:07 ep-32 | ep-22 | ep-03, 21, 28 | ep-26 |
  | C (Llama-3.2-1B) | ep-29, 16, 31, 27, 10, and since 06:44 ep-13, 19 | ep-15 | ep-05, 11 | ep-09 |

- **What may still be missing.** Nothing on the public pairs. On the hidden pairs, the open points are whether family A holds an unmarked copy of the plain model among its plain-like endpoints, and ep-26 in B (see decision 1 for the odds).
- **Submission rule the desk kept.** A grouping is submitted only when a method fixed on the references names it. Four night submissions followed it: 0.0 (floor), 0.1 (families), 0.7, 0.85. The fifth (1.0) added one endpoint that the fp8 test pointed to below its threshold; the desk had declined it, and the lead decided it at 08:06.
- **Devin's second method** (battery modes, closeness, a full-distribution test) reached the same 15 labels independently and supplied the closest pairs the fp8 rule uses. Its own prediction for C's fp8 copy weakened on the second read; he said so before the desk's number came in.

## Incidents you should know

- **00:22, a test sent one request to the organizers' server** with a test key (HTTP 401). The script tests acted on the night authorization. Fixed in PR #31 (merged 00:37) with two regression tests. The stray record was never committed.
- **00:41, the JURECA link dropped.** JSC rebooted login nodes one by one during the night, and a rebooted node comes back without `/p/scratch`. You reopened the link at 00:54.
- **Changes on your laptop made by the desk:**
  - `~/.ssh/config`: reuse-only aliases for the other login nodes added; the previous file is `~/.ssh/config.pre-backup-aliases-20261122`.
  - `~/.config/euroai/jureca_relog.sh` and `ssh_askpass.sh`: one guarded reconnect attempt, with a log in `~/.config/euroai/jureca_relog.log`. Used once, as a test, on `jureca01`.
  - The key passphrase is in `~/.config/euroai/.env`, as you put it. Remove it after the event.
- **01:30, a model equality submission scored less than the desk expected.** The desk had the metric wrong in its head: the model part is computed inside families only, where "one model per family" scores zero. One of five night submissions went to a result worth 0.1, plus the confirmation of the families.
- **Disk on your main laptop.** The local base models were downloaded from Hugging Face into `~/.cache/huggingface` (free disk fell from 87 GB to 27 GB at the lowest; 63 GB free at 07:00 after the ruled-out models were deleted). What to delete is listed near the end of this brief.
- **23:47, a job queued behind the running one.** The desk queued a 5-hour job to hold the node. You pointed out the ban risk; it was cancelled at 23:54. Since then: one job in total, submitted only for ready work.
- **04:47 to 06:00, the desk was down.** The session reached its usage limit, and its two agents with it. The unlearning agent and Devin kept working and posting.
- **Scratch file quota.** Free slots fell from 11,800 to under 1,000 between 00:58 and 01:25. The desk created each job's output files in advance so that a full quota could not break a running job.

## Before 11:00

Commands are for nushell, from the desk's worktree.

1. **Get the night's state.** `cd ~/orca/workspaces/euro-ai-championship-2026/brill; git pull`
2. **Read the public board** and write down our rank per task.
3. **Model equality.** Nothing to submit. The submitted grouping is `tasks/model_equality/results/labelled_copies_v2_b32.csv` (public 1.0). Do not submit `labelled_copies_v2_b26_b32.csv`.
4. **Unlearning.** Nothing to submit. The file behind 0.2858 is `data/unlearning/candidates/k11_b_soup3_x2.pt`.
5. **Watermark.** Nothing to submit. c6 (0.7444) stands.
6. **Answers left on the scored endpoints** (organizers' cap 100 each; the night used 78 on ep-22, 72 on ep-15, at most 66 elsewhere): `uv run python scripts/me_query.py --budget`
7. **Ask an organizer the two questions** under "Decisions waiting for you".

The board states that only a team's best submission on a task counts. A further submission can therefore not lower our rank unless "best" is chosen on the public part and the hidden part disagrees.

## To delete on your main laptop after the event

- In `~/.cache/huggingface/hub`: the seven `models--*` folders and the `blobs` folder that holds their weights, 12 GB in total, all downloaded tonight (SmolLM2-360M, Llama-3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B). They are needed until the last model equality decision.
- `~/.config/euroai/night.json` (it ends by itself at 09:00) and the passphrase line in `~/.config/euroai/.env`.

## Open, not urgent

- A fine-grained GitHub token for the second laptop, limited to this repository, instead of full access to your account.
- An old desk session may still be open in Orca (PID 74888 at 23:30). It confirmed it runs nothing.
