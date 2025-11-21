#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=0
#SBATCH --output=output-%j.txt

cd "$SLURM_SUBMIT_DIR"

# Prefer g++-14 when available (sbatch job); otherwise fall back to system g++.
if command -v g++-14 >/dev/null 2>&1; then
    export CXX=g++-14
else
    export CXX=g++
fi

# Make OpenMP use exactly the CPUs Slurm assigned
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Build your binaries
make

# Run your batch-multi program on all .qoi files in the images/ directory
# (adjust the path/pattern if your images live somewhere else)
srun ./bin/batch-multi images/*.qoi