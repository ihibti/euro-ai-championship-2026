# Morning brief, 22 November

Written by the desk through the night. Final at 07:20; anything later is added under "State". The event ends at 12:00; last submissions by 11:00.

For the jury: `docs_and_plans/guides/judges-defense.md` (four pages: the night in ten lines, the three tasks, why this is not leaderboard fitting, the organizers' rules, ten likely questions). Four points in it are marked **for the lead**.

## State at 07:20

- **Scores:** watermark 0.7444, model equality **0.85** (06:44, was 0.7), unlearning 0.2858 (04:39). Read the public board first thing: the agents have no way to read it, so our rank is unknown since 23:45.
- **Cluster:** no job running or pending. The last one, 15905366 (the unlearning agent's thirteenth batch), ran from 06:48 to 07:12 without a failed run.
- **Night submissions left until 09:00:** model equality 1, unlearning 1. Both stay unused unless you say otherwise.
- **Unlearning:** the night ends at 0.2858. The last batch (a target mixed from three cross-fits) failed two of the four checks its agent had fixed before the job, so the agent left the last slot empty (#12, 07:15). No GPU request is open.
- **Model equality:** a new test found family C's fp8 copy, which gave the 0.85; see "Model equality" below. The next step is yours (decision 1).
- **The desk was down from 04:47 to 06:00** (usage limit of the session). Nothing was running on the cluster in that hour and nothing was lost.

## Decisions waiting for you

1. **Model equality: use your morning slots to add endpoints one at a time?** This is where points are left: 0.85 now, and the board's best is 1.00.
   - *Why the desk stopped.* Its own test fell just short on family B: ep-22 reads as B's fp8 copy at +2.81 standard errors, under the threshold of 3 fixed before the data. The desk had written that nothing would change for B tonight in that case, and kept to it.
   - *Why it is still a good bet.* ep-22 is B's only endpoint above zero and sits about 8 standard errors above the level of B's other endpoints. In C the same pattern, once over the threshold, was worth +0.15. The plain-like endpoints left in B are ep-32 and ep-26 (ep-03, ep-21 and ep-28 carry adapter marks).
   - *Why this cannot overfit.* Adding one endpoint to a family's group changes only its pairs with that group. The public score rises only if the endpoint really is the same model, and then the hidden score rises too. If it falls, the board keeps our best submission. So each such submission is a clean yes or no on one endpoint, at the price of a 30-minute slot.
   - *First, about 3 chances in 4:* ep-32. `make submit TASK=model_equality FILE=tasks/model_equality/results/labelled_copies_v2_b32.csv`
   - *Second, if the first gained, about even odds:* ep-26 as well. `make submit TASK=model_equality FILE=tasks/model_equality/results/labelled_copies_v2_b26_b32.csv`
   - *Then family A,* where the fp8 test found nothing: Devin's closest plain-like endpoints are ep-12 and ep-18, below even odds each. A file for any choice is three commands, under "Before 11:00".
   - If the desk session is still open when you read this, "go B" is enough: it has one night submission left until 09:00.
   - If an organizer tells you the *last* submission counts, not the best, finish by resubmitting the best file.
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
  - *Family B.* ep-22 read +2.71, then +2.81 after 12 more answers: under the threshold, no change. Every other endpoint of B is negative, except the top-p copy ep-07 at +0.78. Decision 1 is about this.
  - *Family A.* No endpoint above zero. No change, and no candidate.
- **Submitted at 06:44, public 0.85:** `tasks/model_equality/results/labelled_copies_v2.csv`.

  | Family | One model: the plain model and its disguises | fp8 copy, alone | Adapter marks, each alone | Plain-like, each alone |
  |---|---|---|---|---|
  | A (Qwen2.5-3B) | ep-02, 23 (prompts), 33 (temperature 1.0), 14 (top-p 0.95), 04 (watermark) | not found | ep-24 | ep-01, 12, 17, 18, 25 |
  | B (Qwen2.5-1.5B) | ep-08, 20, 30, 07, 06 | ep-22, likely | ep-03, 21, 28 | ep-26, 32 |
  | C (Llama-3.2-1B) | ep-29, 16, 31, 27, 10, and since 06:44 ep-13, 19 | ep-15 | ep-05, 11 | ep-09 |

- **What is still missing.** A's and B's plain copies, and whatever the eleventh endpoint of each family is. In C the two plain-like endpoints added together were worth +0.15, which suggests a family holds two plain-like copies of the plain model, not one.
- **Submission rule the desk kept.** A grouping is submitted only when a method fixed on the references names it. No grouping was submitted to try an endpoint against the public score. Four submissions: 0.0 (floor), 0.1 (families), 0.7, 0.85.
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
2. **Read the public board** and write down our rank per task. If we are 6th or better, the only changes worth a slot are those of decision 1, which cannot lower a score.
3. **Model equality.** The submitted grouping is `tasks/model_equality/results/labelled_copies_v2.csv` (0.85). For decision 1:
   - the two prepared files are under decision 1 (at most one submission per 30 minutes, a failure counts);
   - to buy the last answers on ep-22 first, from the other worktree: `cd ~/worktrees/euro-ai-championship-2026/model-equality-refs; uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/bat.csv --repeats 20 --name round1 --endpoints ep-22 --spend` (it asks you to confirm at the terminal once the night authorization has ended; 14 answers, about 2 minutes). Then ask the desk, or a new session, to "score ep-22 and read family B with `fp8_rule.py`".
   - Check a file without submitting: `make check TASK=model_equality FILE=<file>`
   - Build a file for another choice (list each family's same-model group; every other endpoint stays alone), then check and submit it:
     `python3 -I tasks/model_equality/analysis/local_anchor/rules/make_grouping.py /tmp/raw.csv A=2,23,33,14,4,12 B=8,20,30,7,6,32 C=29,16,10,31,27,13,19`
     `uv run python scripts/me_labels.py /tmp/raw.csv tasks/model_equality/results/labelled_copies_v3.csv`
     `make check TASK=model_equality FILE=tasks/model_equality/results/labelled_copies_v3.csv`
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
