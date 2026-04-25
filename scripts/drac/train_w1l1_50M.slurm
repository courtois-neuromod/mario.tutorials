#!/bin/bash
# Train PPO Mario agent on world 1 / level 1 for 50M steps on a single GPU.
# Submit from the repo root: sbatch scripts/drac/train_w1l1_50M.slurm
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
mkdir -p logs models

module load StdEnv/2023 python/3.10 cuda/12.2
source ~/python_envs/tuto/bin/activate

python scripts/train_mario_agent.py \
    --gpu \
    --timesteps 50000000 \
    --levels w1l1 \
    --save_checkpoints \
    --checkpoint_freq 500000
