#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-cpu
#SBATCH --partition=plgrid
#SBATCH --output=results/test_diffusion.out
#SBATCH --error=results/test_diffusion.err

# ml ML-bundle/24.06a

# cd $SCRATCH

uv sync --no-cache

uv run quantum-diffusion  --target data --epochs 10 --device cpu
