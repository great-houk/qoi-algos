#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=0
#SBATCH --output=batch-multi-%j.out

# Start in the directory where you ran sbatch (multi-cpu/)
cd "$SLURM_SUBMIT_DIR"

# Prefer g++-14 (like bench.slurm); fall back to system g++
if command -v g++-14 >/dev/null 2>&1; then
    export CXX=g++-14
else
    export CXX=g++
fi

# Optional OpenMP settings for your nested parallelism
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_NESTED=true

# Make sure the *parent* bin/ exists: qoi-algos/bin
mkdir -p ../bin

# Compile batch-multi; note we link into ../bin
$CXX -O3 -std=gnu++17 -fopenmp \
    -I.. -I../reference \
    batch-multi.cpp \
    qoi-mc.cpp \
    qoi-decode.cpp \
    ../reference/qoi-reference.cpp \
    -o ../bin/batch-multi

# Run on all .qoi images in qoi-algos/images
srun ../bin/batch-multi ../images/*.qoi