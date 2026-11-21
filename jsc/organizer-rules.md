# The organizers' rules for JURECA

Copied word for word. `AGENTS.md`, hard rule 5, is our summary of them; this file is the source.

## Rules for AI agents, file handed to the lead on 2026-11-21 at about 22:25

````markdown
# Hackathon rules for AI agents — JURECA, project `training2666`

## 1. GPUs: one job per team

**HARD LIMIT: your TEAM may run at most ONE job at a time in the `cispahack` reservation.**
Nodes are exclusive — 1 job = 1 whole node = 4 GPUs — and the limit is shared across **all
teammates** in your folder `/p/scratch/training2666/<lead>/<team>/`, counting **running AND
pending** jobs. This keeps nodes free so every team gets a turn.

**Before every `sbatch`, check the WHOLE team** (not just `squeue -u $USER`, which can't see teammates):

```bash
T=$(pwd | grep -oE '/p/scratch/training2666/[^/]+/[^/]+')
squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "$T"
```

If that shows any running or pending job, **do not submit** — your team already holds its node.

- **Need more than 1 GPU at once?** Use ONE job with `#SBATCH --gres=gpu:4` and run your parallel
  work as steps inside it (multiple `srun ... &`, or a multi-GPU program). Never two jobs.
- **Follow-up work?** Chain it: `sbatch --dependency=afterany:<jobid> next.sh`. Never stack pending jobs.
- **Job arrays:** throttle to one at a time with `--array=0-N%1`.
- **Always `sbatch` from inside your team folder** `/p/scratch/training2666/<lead>/<team>/...` —
  jobs submitted from your home directory are untracked and count as violations.
- **Already on more than one node?** `scancel` down to a single job now.
- **Do not work around this limit** (other partitions, accounts, or reservations). If it blocks you,
  stop and ask a human organizer — do not route around it.

## 2. Files: keep the file COUNT low — scratch is limited by number of files, not just size

**The whole hackathon shares one file-count (inode) quota on `/p/scratch/training2666`, and it is
almost full.** When it hits the limit, *nobody* can create a file there and every team's jobs fail
with `Disk quota exceeded`. Every file and every directory counts as one, however small. A 10 GB
archive costs 1; the same data unpacked into 50,000 images costs 50,000.

Check before any step that creates many files, and after:

```bash
df --output=itotal,iused,iavail /p/scratch/training2666   # whole project: total / used / free
find . -xdev | wc -l                                      # what the current folder costs
```

**Rule of thumb: nothing you do should add more than a few hundred files.** If a step would add
thousands, stop and restructure it as below.

**Reading data — do not unpack archives onto scratch.** Read straight from the container:
- `.tar` / `.tar.gz`: `tarfile.open(p)` and `extractfile(member)`, or `webdataset` for training loops.
- `.zip`: `zipfile.ZipFile(p).open(name)` (random access, no extraction).
- Hugging Face datasets: load the parquet/arrow files (`datasets.load_dataset(...)`,
  `pd.read_parquet`); do not export them to one image/JSON per sample.
- `.npz` / `.h5` / `.pt`: open and index in place.
- A tool that truly needs a few files on disk: fine — extract only those, and delete them afterwards.

**Writing results — one container per run, not one file per item.**
- Per-sample outputs (predictions, images, masks, embeddings): append to a single `.jsonl`,
  `.parquet`, `.npz`/`.h5`, or write a `.tar` shard with `tarfile`/`webdataset.TarWriter` —
  never `out/00001.png`, `out/00002.png`, …
- Checkpoints: keep `last` and `best` only; overwrite, do not write one per epoch/step.
- Logs and metrics: one file per run. No per-step dumps, no debug images written to disk.
- Finished with a directory of many files you still need? Pack it and remove the tree:
  `tar -cf run1.tar run1/ && rm -rf run1/`.

**Environments and caches are the biggest hidden cost** (a Python venv is 10,000–30,000 files).
- Reuse the venv that already exists in the task folder. Do not create another venv, conda env or
  `pip install --target` copy, and do not clone large repositories you only need one file from.
- `export PYTHONDONTWRITEBYTECODE=1` so Python stops writing `__pycache__` files.
- After installing packages: `uv cache clean` (existing venvs keep working).
- Delete what you no longer need and created yourself: old venvs, `.downloads/`, unpacked
  datasets, `wandb/` run folders, stale checkpoints, `__pycache__`.

Only delete files your own team created, and ask your human before deleting anything you did not
create in this session. Do not move files to other projects or filesystems to dodge the limit
without asking an organizer.
````

## Message relayed by the lead on 2026-11-21 at about 22:45

> If you're waiting on the GPU queue, it's also possible launch slurm jobs without the reservation. Just remove the cispahack reservation. Those go into the general juelich supercomputers queue, so you might have to wait a bit, but they don't count for your one-node limit

This message is later than the file above, whose section 1 says not to work around the limit through other partitions, accounts or reservations. We read the two together: a job without the reservation line is allowed and does not count, and nothing else routes around the limit.

## Messages on the file-count limit, relayed by the lead on 2026-11-21 at about 21:55

> we have now hit the indode hardcap - we urge everyone to reconsider condense thier file usage- espeically deleting unused logfiles. Montioring of innode usage and (at first) firendly reminders will comence soon.

> For context, scratch has 4,000k (4 million) inodes in total, shared by all of us. Every file counts as one inode, however small it is, so lots of tiny files fill it up long before disk space runs out. Please delete small files you no longer need, for example: logs, .json files, .txt files, ... This doesn't apply to model checkpoints. Please keep those. Tip: find /path/to/your/scratch -type f | wc -l shows how many files you have. If you want to keep a folder of small files, pack it into one tar archive, which uses only one inode. Thanks!

## First version of the GPU rule, 2026-11-21 at 19:50

Replaced by section 1 of the file above, which says the same with two additions: a job submitted from the home directory counts as a violation, and the limit must not be worked around.
