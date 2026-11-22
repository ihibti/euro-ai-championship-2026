# The desk's scripts for the night of 21 to 22 November 2026

Copies of the scripts the desk ran from its session's scratch folder, kept as the record of how the cluster and the scored endpoints were used. `SCRATCHPAD` stands for that folder and `$HOME` for the home folder.

- `stage_*.sh`, `stage_*_remote.sh`: staging of a GPU batch on scratch. Each checks that no job of ours is queued, verifies that every checkpoint it is about to empty equals the laptop copy by hash, renames and empties old output files so that the batch creates no new file, uploads the run script, compares hashes with the commit, and prints the file count before and after.
- `batch_*.sh`, `batch_*.sbatch`: the wrapper and the rendered job file of each batch.
- `wait_job.sh`: asks once a minute whether a job has left the queue, through a connection that is already open.
- `pull_ckpt.sh`: copies checkpoints to the laptop and checks their hashes.
- `next_unlearning.sh`: builds and submits the unlearning agent's queued line at its slot.
- `spend_*.sh`: purchases of answers, references first.
- `watch.sh`: polls the issues and pull requests for new comments.
- `hold.sh`: a holder loop from before the lead's rules of 23:53 on 21 November. It kept a job alive while steps ran inside it. Not used after 23:54, and kept only as the record of that practice.

The team check and the `sbatch` call were typed for each batch and are quoted in `docs_and_plans/work/doing/night-plan.md`.
