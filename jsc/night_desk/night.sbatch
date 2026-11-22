#!/bin/bash
# GPU job on the event reservation: one node and its four GPUs, as the organizers ask since 2026-11-21 19:50.
# The whole team may have one job at a time, running or queued (AGENTS.md, hard rule 5).
# Render with scripts/render_template.py and see jsc/README.md ("Submitting a job").
# TASK_DIR is a task folder under /p/scratch/training2666/hibti1/Begg/ with its own .venv.
# Create TASK_DIR/output before sbatch, as the organizers' README asks.
#SBATCH --job-name=begg-ul
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/p/scratch/training2666/hibti1/Begg/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/hibti1/Begg/unlearning/output/%j.err

set -euo pipefail

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "/p/scratch/training2666/hibti1/Begg/unlearning"
source .venv/bin/activate
echo "job ${SLURM_JOB_ID} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun bash output/desk/hold.sh 60
