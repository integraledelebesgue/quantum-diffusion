#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=results/test_diffusion.out
#SBATCH --error=results/test_diffusion.err

module add ML-bundle/24.06a

python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate

pip install --cache-dir $SCRATCH/.cache/pip .

python3.11 -m quantum_diffusion --no-interactive --target data --epochs 10 --device cuda
