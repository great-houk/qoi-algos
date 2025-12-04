#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=2
#SBATCH --output=output

cd "$SLURM_SUBMIT_DIR"

module load nvidia/cuda/

# Prefer g++-14 when available (sbatch job); otherwise fall back to system g++.
if command -v g++-14 >/dev/null 2>&1; then
	export CXX=g++-14
else
	export CXX=g++
fi

# Build and run
make
./bin/bench 1 images_small
