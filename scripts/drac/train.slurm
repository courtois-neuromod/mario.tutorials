#!/bin/bash
# Train the PPO Mario agent on a DRAC GPU node.
#
# First-time setup (login node, once per project):
#     bash scripts/drac/setup.sh
#
# Submit a default run (5M steps, 4 levels, ~3h on V100 or A100):
#     sbatch scripts/drac/train.slurm
#
# Override knobs via --export (training-script args go here, not in #SBATCH):
#     sbatch --export=ALL,STEPS=10000000,LEVELS=all scripts/drac/train.slurm
#     sbatch --export=ALL,STEPS=10000               scripts/drac/train.slurm   # 5-min smoke test
#
# Pin a specific GPU model by editing --gres (e.g. gpu:a100:1, gpu:v100l:1).
#
#SBATCH --account=def-CHANGEME       # EDIT to your allocation (def-<pi> or rrg-<pi>)
#SBATCH --job-name=mario_ppo
#SBATCH --time=04:00:00              # 5M steps fits well under 4h on V100/A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs models

STEPS="${STEPS:-5000000}"
LEVELS="${LEVELS:-w1l1,w1l2,w1l3,w2l1}"

echo "============================================================"
echo "Mario PPO training — DRAC job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Steps : $STEPS"
echo "Levels: $LEVELS"
echo "Repo  : $REPO_ROOT"
echo "============================================================"
nvidia-smi -L || echo "(nvidia-smi unavailable — check --gres allocation)"

module load StdEnv/2023 python/3.10 cuda/12.2
# shellcheck disable=SC1091
source env/bin/activate

python scripts/train_mario_agent.py \
    --gpu \
    --timesteps "$STEPS" \
    --levels "$LEVELS" \
    --save_checkpoints \
    --checkpoint_freq 250000 \
    --gif_freq 200

echo
echo "==> Training finished. Latest checkpoints:"
ls -lt models/*.pth 2>/dev/null | head -3
