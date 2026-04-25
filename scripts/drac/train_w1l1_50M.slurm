#!/bin/bash
# Train PPO Mario agent on world 1 / level 1 for 50M steps on a single GPU.
#
# Submit from the repo root after running scripts/drac/setup.sh (or after
# activating your own venv via the VENV env var):
#
#     sbatch scripts/drac/train_w1l1_50M.slurm
#     sbatch --export=ALL,VENV=$HOME/python_envs/tuto scripts/drac/train_w1l1_50M.slurm
#
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=mario_w1l1_50M
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs models

module load StdEnv/2023 python/3.10 cuda/12.2
# Activate the venv: defaults to ./env (created by scripts/drac/setup.sh);
# override with VENV=/path/to/venv when submitting.
source "${VENV:-$REPO_ROOT/env}/bin/activate"

echo "============================================================"
echo "Mario PPO — job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Levels: w1l1   Steps: 50000000"
echo "Repo  : $REPO_ROOT"
echo "Venv  : ${VENV:-$REPO_ROOT/env}"
echo "============================================================"
nvidia-smi -L

python scripts/train_mario_agent.py \
    --gpu \
    --timesteps 50000000 \
    --levels w1l1 \
    --save_checkpoints \
    --checkpoint_freq 500000 \
    --gif_freq 200
