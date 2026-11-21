# JSC / JURECA-DC: connecting and running GPU jobs

This page records official JSC documentation fetched on 2026-11-21 and the organizer slides (`GPU-usage-paris.pdf`, received during the talk and not committed). It tells you which source each fact comes from and where the two sources disagree.

**Status (2026-11-21):** the team folder is set up and a GPU job has run on the event reservation; see [Team folder](#team-folder-verified-2026-11-21). Jupyter-JSC is still untested.

## Event setup (2026-11-21, from the organizers)

The organizers publish a README and two scripts in the Hugging Face dataset `SprintML/hackathon`. For this event they replace most of the manual steps further down this page.

1. **Team lead, once:**
   - Download `hackathon_setup.sh` and set `TEAM_LEAD` (it must equal your own Jülich username), `TEAMMATE_1`…`TEAMMATE_3` (at least one) and `TEAM_FOLDER`.
   - Export `HF_TOKEN` from a Hugging Face account approved for the gated watermark dataset.
   - Run `source hackathon_setup.sh`. It creates `/p/scratch/training2666/<lead>/<team-folder>/` and sets ACLs for the listed users. It also installs uv, downloads the three task datasets into `Hackathon_Watermark_Localisation_Task/`, `model_equality/` and `unlearning/`, and builds a `.venv` for each task folder that has a top-level `requirements.txt`. The watermark dataset has none (its `requirements.txt` is inside `starter_kit.zip`), so that task gets no `.venv`; we build it ourselves (see `tasks/watermark_localization/README.md`).
   - Done for our team: the lead `hibti1` ran it with `TEAM_FOLDER=Begg` (exit 0). See Team folder below.
2. **Everyone else, after the lead finishes:**
   - Download `teammate.sh`, set `TEAM_LEAD=hibti1` and `TEAM_FOLDER=Begg`, and run `source teammate.sh`. Both teammates have access through ACLs. Their agents take them through it with `docs_and_plans/guides/teammate-onboarding.md`.
   - **Re-download `teammate.sh` if you fetched it before 2026-11-21 15:44 Paris time.** Earlier copies set `PROJECT_NAME="training2625"`.
3. **Failed downloads.** A task whose download fails is skipped, and a rerun skips every task folder that already exists. Delete a half-created task folder before rerunning. The lead grants access to teammates added later (organizers' README, section 6).
4. **Jobs**, per the organizers' README:
   - Flags: `--account=training2666`, `--reservation=cispahack`, `--partition=dc-gpu`, `--gres=gpu:1`, `--cpus-per-task=8`, `--output=output/%j.out`, `--error=output/%j.err`.
   - Run `mkdir -p output` before `sbatch`.
   - Our `jureca_gpu.sbatch.template` uses these values; see Submitting a job.
5. **This repo on JURECA.** SSH agent forwarding is disabled on the server, so cloning the private repo from JURECA needs its own credential there, such as a read-only deploy key. Otherwise rsync from a laptop over the open connection.

### Team folder (verified 2026-11-21)

`/p/scratch/training2666/hibti1/Begg/`, created by the organizers' `hackathon_setup.sh`.

| Folder | Environment | Contents |
|---|---|---|
| `unlearning/` | uv project, Python 3.12 `.venv`: torch 2.15.0+cu132, timm 1.0.31 | `model.pt`, `forget_images.npz` |
| `model_equality/` | uv project, Python 3.12 `.venv`: requests, numpy, pandas, scikit-learn, scipy | the public starter files |
| `Hackathon_Watermark_Localisation_Task/` | built by us (the organizers' script builds none): Python 3.11 `.venv` with the kit's `requirements.txt` | the kit unpacked at the folder root (`wmloc/`, `submission_template.py`, `download_models.py`); `models/`; data in `release/{train,test}` |

- Watermark `models/` holds videoseal at the pinned commit, the PixelSeal checkpoint and the StegaStamp ONNX. All three wrappers were loaded once on the login node, so the lazily fetched TrustMark weights are cached and jobs can run offline.
- The official all-zeros watermark baseline scores **0.2518** on the 510 train images. A valid test file is at `Hackathon_Watermark_Localisation_Task/output/baseline_allzeros.pt`; it has not been submitted.
- **GPU check:** job 15901345 on reservation `cispahack` saw one A100-SXM4-40GB inside `srun` with `--gres=gpu:1` (driver 620.70.06, `torch.cuda.is_available()` True).
- **Reservation `cispahack`:** active 2026-11-21 11:00 to 2026-11-22 14:00 (Jülich time, the same as Paris), 30 `dc-gpu` nodes, accounts `admin` and `training2666`.

### Laptop agents and the shared connection

A human opens one authenticated connection. It stays up for 8 hours (`ControlPersist 8h`):

```
ssh -fN jureca          # key passphrase and TOTP, once
ssh -O check jureca     # prints "Master running"
```

Agents use only the `jureca-mux` alias. It reuses that connection and never opens a new one, so it causes no prompts and no extra connections that could trip JSC's firewall. It is included in `ssh_config.template`:

```
Host jureca-mux
  HostName jureca.fz-juelich.de
  User <JUDOOR_USERNAME>
  ControlMaster no
  ControlPath ~/.ssh/control-%h-%p-%r
  ProxyCommand /usr/bin/false
  BatchMode yes
```

While the connection is open, `ssh jureca-mux '<command>'` and `rsync -e ssh <src> jureca-mux:<path>` work. When it is closed, they fail at once without touching the network. The `ControlPath` must be the same as in the `jureca` block.

Official sources:
- **ACC**: https://apps.fz-juelich.de/jsc/hps/jureca/access.html
- **BATCH**: …/jureca/batchsystem.html
- **GPU**: …/jureca/gpu-computing.html
- **ENV**: …/jureca/environment.html
- **FS**: …/jureca/filesystems.html
- **FAQ**: …/jureca/faq.html
- **TOOLS**: …/jureca/jsctools.html
- **MOD**: …/jureca/software-modules.html
- **SCL**: …/jureca/stages-changelog.html
- **CHG**: …/jureca/changelog.html
- **KI**: …/jureca/known-issues.html
- **JUDOOR**: https://www.fz-juelich.de/en/jsc/services/user-support/how-to-get-access-to-systems/judoor
- **JJ**: https://jupyterjsc.pages.jsc.fz-juelich.de/docs/jupyterjsc/
- **SDL**: https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/ (JSC's Simulation and Data Lab for Applied Machine Learning)
- **PD**: https://www.fz-juelich.de/en/jsc/services/data-services/processing-personal-data-on-jscs-infrastructure

## Verified facts

| Topic | Value | Source |
|---|---|---|
| Login host | `jureca.fz-juelich.de`. Also `jureca-ipv4` and `jureca01`…`jureca12`. Use the FQDN from outside the FZJ campus. | ACC |
| Data gateway | `judac.fz-juelich.de`, for transfers (UFTP, GridFTP, …). Keys are uploaded per system. | JUDAC docs |
| Key type | Ed25519. **The private key must have a passphrase.** Password login is impossible. Agent forwarding is disabled server-side. | ACC |
| Key upload | In JuDoor, separately for each system. Active "within 15 minutes". | ACC |
| `from=` clause | Comma-separated IPs, CIDRs or hostname patterns. Netmask must be ≥ 8. No trailing `*`. With IPv6 you must list your IPv6 address or force IPv4. | ACC |
| MFA | Mandatory for HPC accounts. Until it is set up, the account is read-only. SSH asks for `JSC TOTP Verification code:`. | JUDOOR, ACC |
| Avoiding repeated TOTP | `ControlMaster auto` / `ControlPersist`. This is officially documented (Linux/macOS only). | ACC |
| Too many connections | Too many SSH connections in a short time trigger an automatic firewall block. | ACC |
| GPU node | 4× A100 40 GB, 2× EPYC 7742 (128 physical cores), 512 GB RAM. Part of the memory is reserved. | BATCH, CONF, slides |
| Partitions | `dc-gpu`: max 24 h ("24 h / 6 h"), default 1 h, 1–24 nodes, **no internet**. `dc-gpu-devel`: max 2 h, default 30 min, 1–4 nodes, at most 2 running jobs, **has internet**. | BATCH |
| Allocation | Nodes are exclusive and charged per node-hour regardless of the GPU count. "Production workloads must use all available GPU resources per node." With no GPU request, `--gres=gpu:4` is added automatically. | BATCH, GPU |
| GPU visibility | One task sees one GPU by default. For one task using 4 GPUs, `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU cgroup constraints have been enabled since 2025-03-13. | GPU, CHG |
| srun CPUs | srun does not inherit `--cpus-per-task`, so `export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}`. Use `srun`, not `mpiexec`. | BATCH, AFF |
| Budget | `#SBATCH --account=<budget>`. `jutil env activate -p <project> [-A <budget>]` exports `$PROJECT`, `$SCRATCH`, … | BATCH, TOOLS, ENV |
| Internet | Available on login nodes and devel partitions only (HTTP/S ports). Download models and datasets on a login node, then run offline. | BATCH |
| Modules | Lmod. The default stage is **Stages/2026** (Python 3.13.5, GCC 14.3.0, CUDA 13.0.0). Use `module spider <name>` to search. | MOD, SCL, CHG |
| Conda | The Anaconda default channel is blocked or prohibited. conda-forge and miniforge are acceptable. | KI, SDL |
| Personal data | Allowed only after a GDPR assessment by the JSC Data Protection Team plus a signed addendum. Pseudonymised data is included in this rule. Only anonymous data is unrestricted. | PD |

### Storage

All of these are available after `jutil env activate -p <project>`.

| Variable | Purpose | Quota (soft/hard) | Notes |
|---|---|---|---|
| `$HOME` | Small personal config only | 20/22 GB, 80k/82k files | Don't put caches or venvs here |
| `$PROJECT` | Code, venvs, small artifacts; backed up | 15/17 TB, 3M/3.3M files | Shared by the whole project. Work in `$PROJECT/$USER` |
| `$SCRATCH` | Large data, checkpoints, caches; **not** backed up | 90/95 TB, 4M/4.4M files | Files are deleted 90 days after last access or modification |
| `$DATA` | Long-term data projects | separate application | ENV and FS disagree on whether it is visible from compute nodes |

Check usage with `jutil project dataquota -p <project>` (updated every 8 h). Always refer to storage through these variables: the docs show both `/p/project1/…` and `/p/project/…` paths.

## Organizer slides vs. official docs

| Slide content | Assessment |
|---|---|
| `jutil env activate -p training2614` (screenshot shows `training2557`) | These IDs come from earlier events. Our project is **training2666**, and the slide's own sbatch uses `--account=training2666`. |
| `#SBATCH –reservation=cispahack` | The slide uses an en-dash, so the line will not parse as an option. Retype it with `--`. The name `cispahack` is **confirmed** for this event. |
| `--gres=gpu:1` | The organizers ask for it, and our template uses it; a job then sees one GPU. BATCH says nodes are allocated whole and production jobs should use all 4 GPUs. Whether `gpu:4` is allowed in the reservation is an **open question for the organizers**. |
| MOTD 2025-03-31: "Do not constrain GPU devices with Slurm … not constrained by cgroups" | Contradicted by CHG 2025-03-13 (cgroup constraints enabled) and absent from the current KI page. It is probably stale. Check on login. |
| `MACs hmac-sha2-256-etm@openssh.com` in `~/.ssh/config` | Official docs don't require it for macOS. Left out of our template; see Troubleshooting. |
| uv installed in `$HOME`, `UV_CACHE_DIR` and related variables under `$PROJECT/$USER/uv`, `uv add torch` (pip wheels) | Works, and our `env_uv.sh` follows it. JSC/SDL instead recommend module-provided PyTorch plus a venv (`env_modules.sh.template`). Caches under `$PROJECT` count toward its 3M-file quota. |
| The sample job submits to a leaderboard server from a compute node | That contradicts "no internet on compute nodes" unless the server sits inside FZJ or the reservation is configured differently. **Ask the organizers.** |

## One-time setup (after the talk)

For the event, teammates follow `docs_and_plans/guides/teammate-onboarding.md` instead of this section. It covers the same ground step by step, with a check per step, and assumes a phone hotspot. This section stays as the general reference.

### 1. JuDoor (browser)
1. Confirm that MFA is enabled, because SSH requires TOTP. Only you can change account settings.
2. Usage agreements: you report they are accepted.
3. On the JuDoor key page, note the IPv4 address it shows and the ranges it suggests for your current network (the venue Wi-Fi). If you will also connect from home or university, list those ranges in the same `from="…"` clause, separated by commas.

### 2. Key on the laptop (nushell-compatible)
```
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519_jsc
open --raw ~/.ssh/id_ed25519_jsc.pub
```
Set a passphrase; JSC requires one. Paste `from="<ranges>" ssh-ed25519 AAAA…` into the JuDoor upload form, then wait up to 15 minutes.

### 3. SSH config (nushell)
Review `jsc/ssh_config.template`, then append it with your username:
```
open --raw jsc/ssh_config.template | str replace '<JUDOOR_USERNAME>' '<your-judoor-username>' | save --append ~/.ssh/config
ssh jureca
```
The first login asks for the key passphrase and a TOTP code. Compare the host key fingerprint with the one JuDoor shows. `ControlPersist 8h` then reuses that connection for further SSH, scp, rsync and VS Code sessions.

### 4. Project workspace (on the login node, bash)
```
jutil user projects
jutil env activate -p training2666
echo "$PROJECT" "$SCRATCH"
mkdir -p "$PROJECT/$USER" && chmod o-rwx "$PROJECT/$USER"
```
`jutil user projects` should list training2666 and its budget account. Use that budget for `--account`.

### 5. Code onto the cluster
There is no git remote yet. Copy the repo from the laptop (nushell; one connection through ControlMaster):
```
rsync -av --exclude .venv --exclude data --exclude outputs --exclude 'papers/*.pdf' ./ jureca:<PROJECT_DIR>/<your-judoor-username>/euro-ai-championship-2026/
```
Here `<PROJECT_DIR>` is the `$PROJECT` value printed in step 4. Alternatively, once a private remote exists, `git clone` on the login node.

### 6. Python environment

Pick one option. **Option A (uv)** is the organizers' tested path and matches this repo. **Option B (modules)** is JSC's recommendation.

**Option A: uv (login node, bash)**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$PROJECT/$USER/euro-ai-championship-2026"
source jsc/env_uv.sh
uv sync --extra dev --extra torch
python scripts/check_env.py --no-probe
```
- The install script is the organizers' method; review it first if you prefer.
- The PyPI torch wheels bundle their own CUDA runtime. Whether the node driver supports it is **unverified** until a GPU job prints `cuda_available: true`.

**Option B: modules (login node, bash)**
```
module spider PyTorch
cp jsc/env_modules.sh.template jsc/env_modules.sh
```
Edit `<EXTRA_MODULES>` in `jsc/env_modules.sh` to the module names you found, for example `PyTorch torchvision` if Stage 2026 provides those. Then:
```
source jsc/env_modules.sh
```
The first `source` creates the venv and runs `pip install -e .[dev]`, which needs internet, so do it on a login node. For MPI between GPUs, also `module load MPI-settings/CUDA` (GPU).

### 7. Pre-download models and data (login node)
Compute nodes have no internet. Fetch everything with `HF_HOME` set (done by the env scripts) on a login node or in a `dc-gpu-devel` job. Jobs then run with `HF_HUB_OFFLINE=1`.

## Submitting a job

`jsc/jureca_gpu.sbatch.template` carries the event values: account `training2666`, reservation `cispahack`, partition `dc-gpu`, one node, one task, `--gres=gpu:1`, 8 CPUs per task, and `SRUN_CPUS_PER_TASK`. Logs go to the task folder's `output/`. Four placeholders remain:

| Placeholder | Value |
|---|---|
| `JOB_NAME` | a short name, shown by `squeue` |
| `TASK_DIR` | a task folder, e.g. `/p/scratch/training2666/hibti1/Begg/unlearning`; the job runs there and activates its `.venv` |
| `TIME` | walltime `HH:MM:SS`; the reservation ends 2026-11-22 14:00 |
| `COMMAND` | what `srun` runs in `TASK_DIR`, e.g. `python unlearn.py` |

On a login node (bash), from a checkout of this repo:
```
TASK_DIR=/p/scratch/training2666/hibti1/Begg/unlearning
python3 scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=unlearn TASK_DIR=$TASK_DIR TIME=01:00:00 COMMAND='python unlearn.py' > $TASK_DIR/job.sbatch
mkdir -p $TASK_DIR/output
sbatch $TASK_DIR/job.sbatch
squeue -u "$USER"
sacct -j <jobid>
```
- `render_template.py` refuses to write a script while any placeholder is still unfilled.
- **Allocation is per whole node** (4 A100s), whatever the GPU request. The organizers ask for `gpu:1`; whether `gpu:4` is allowed in the reservation is an open question for them.
- Compute nodes have no internet: download everything on a login node first. The template sets the Hugging Face and uv offline flags.
- To test quickly outside the reservation, use `--partition=dc-gpu-devel` (max 2 h, has internet) and drop the reservation line.

## Jupyter-JSC

- At https://jupyter.jsc.fz-juelich.de, log in with **JSC Account**. Helmholtz ID logins cannot reach the HPC systems (JJ).
- In the launch form, choose JURECA, project and partition. Reservations appear there if one exists.
- To use our venv as a kernel, follow JJ "kernels_hpc_venv":
  1. Create a venv with `--system-site-packages` under `${PROJECT}/${USER}/jupyter/kernels/<name>`.
  2. `pip install ipykernel`.
  3. Write a `kernel.sh` that loads exactly the same modules.
- The login has **not been tested**.

## Troubleshooting

- **`Permission denied (publickey)`:**
  - The key may not be active yet; wait the full 15 minutes.
  - Your current IP may be outside `from=`. Check the IP JuDoor reports.
  - With IPv6, the client prefers v6. Our template sets `AddressFamily inet` to force IPv4.
- **Prompt hangs or you are blocked:** scripted rsync/scp loops can trigger the firewall. Keep one ControlMaster connection open.
- **MAC errors from macOS OpenSSH:** only then try the slides' `MACs hmac-sha2-256-etm@openssh.com`.
- **Only 1 GPU visible:** check `CUDA_VISIBLE_DEVICES` in the `check_env` output.

## Still unverified

1. Whether `gpu:4` (a whole node) is allowed in the `cispahack` reservation. Ask the organizers.
2. PyTorch module names in Stage 2026 (the task `.venv`s use pip wheels, which work: torch 2.15.0+cu132 sees the GPU).
3. The meaning of "24 h / 6 h" for `dc-gpu`.
4. Whether Slurm creates a missing `--output` directory; create it first.
5. The actual `$PROJECT` path. The team folder is on scratch.
6. How the leaderboard is reachable from compute nodes.
7. VS Code Remote-SSH behaviour with TOTP and ControlMaster; it is undocumented by JSC.
8. The text of the system usage agreement (inside JuDoor).

Verified since the first version of this page: the reservation name and account (`cispahack`, `training2666`), and that pip-wheel CUDA works with the node driver (job 15901345).
