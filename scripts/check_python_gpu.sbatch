#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=results/check_python.out
#SBATCH --error=results/check_python.err

module add ML-bundle/24.06a

whereis python
whereis python3
python3 --version
python3.11 --version
