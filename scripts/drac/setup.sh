#!/bin/bash
# One-time setup on a Compute Canada / Digital Research Alliance (DRAC) login
# node (Beluga, Narval, Cedar, Graham). Run from the repo root.
#
#   ssh <user>@beluga.alliancecan.ca
#   cd ~/projects/def-<pi>/$USER/mario.tutorials       # or $SCRATCH/...
#   bash scripts/drac/setup.sh
#
# The login node is the only place that has outbound internet, so this is also
# where we (a) build the venv from DRAC's local wheelhouse and (b) pre-stage the
# Mario ROM. Compute nodes will then be able to run the training without
# needing any network access.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Loading modules (StdEnv/2023, python 3.10, CUDA 12.2)"
module load StdEnv/2023 python/3.10 cuda/12.2

echo "==> Creating virtualenv at env/ (using DRAC's --no-download wheelhouse)"
if [[ ! -d env ]]; then
    virtualenv --no-download env
fi
# shellcheck disable=SC1091
source env/bin/activate

echo "==> Upgrading pip from the local wheelhouse"
pip install --no-index --upgrade pip

echo "==> Installing pinned dependencies"
# Try the lockfile first (full reproducibility); fall back to requirements.txt
# with a mixed index for any wheel that DRAC doesn't ship locally
# (stable-retro, gdown, …).
if ! pip install --no-index -r requirements.lock; then
    echo "    Some wheels missing from DRAC's local index — falling back to PyPI"
    pip install -r requirements.txt
fi

echo "==> Editable install of the local package (so 'from src import …' works)"
pip install --no-deps -e .

echo "==> Staging Mario ROM (login nodes have internet; compute nodes do not)"
python -c "
import sys
sys.path.insert(0, 'src')
from utils import download_stimuli
download_stimuli()
"

echo
echo "==> Setup complete. Edit the --account line in scripts/drac/train.slurm,"
echo "    then submit a smoke-test run with:"
echo "        sbatch --time=00:30:00 --export=ALL,STEPS=10000 scripts/drac/train.slurm"
echo "    or a full 5M-step training:"
echo "        sbatch scripts/drac/train.slurm"
