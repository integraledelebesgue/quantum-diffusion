#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=results/check_python.out
#SBATCH --error=results/check_python.err

set -ex

module add ML-bundle/25.04

echo $PIP_EXTRA_INDEX_URL

# python3.11 --version

rm -r .venv-torch-test
python3.11 -m venv .venv-torch-test
source .venv-torch-test/bin/activate

pip install --no-cache-dir numpy
pip install --no-cache-dir torch==2.8.0+cu128
# pip install --no-cache-dir --index-url=file:///net/software/aarch64/el8/wheels/ML-bundle/25.04/simple torch

srun python3.11 scripts/test_torch_cuda.py
