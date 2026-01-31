#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=results/test_diffusion.out
#SBATCH --error=results/test_diffusion.err

eval "$(/net/home/plgrid/plgjsmolka/.local/bin/mise activate bash)"  # TODO: Make this generic
export UV_CACHE_DIR=$SCRATCH/.cache/uv/

set -ex

module add ML-bundle/24.06a

uv sync --all-groups
uv run scripts/train_diffusion.py
