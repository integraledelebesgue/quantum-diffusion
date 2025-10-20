#!/usr/bin/bash -l
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=16
# SBATCH --gres=gpu:1
# SBATCH --time=01:00:00
# SBATCH --account=plgkwantowy5
# SBATCH --partition=plgrid-gpu-gh200
# SBATCH --output=results/job-%j.out
# SBATCH --error=results/job-%j.err

ml ML-bundle/24.06a

cd $SCRATCH

python -m venv  .venv
source .venv/bin/activate

pip install --no-cache-dir torch==2.5.1+cu124.post3

python scripts/test_torch_cuda.py
