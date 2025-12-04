#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=0
#SBATCH --output=batch-single-36thread-%j.out

# Start in the directory where you ran sbatch (multi-cpu/)
cd "$SLURM_SUBMIT_DIR"

# Prefer g++-14 (like bench.slurm); fall back to system g++
if command -v g++-14 >/dev/null 2>&1; then
    export CXX=g++-14
else
    export CXX=g++
fi


# Make sure the *parent* bin/ exists: qoi-algos/bin
mkdir -p ../bin

# Compile batch-multi; note we link into ../bin
$CXX -O3 -std=gnu++17 -fopenmp \
    -I.. -I../reference \
    batch-single.cpp \
    qoi-sc.cpp \
    qoi-decode.cpp \
    ../reference/qoi-reference.cpp \
    -o ../bin/batch-single

# Run on all .qoi images in qoi-algos/images
srun ../bin/batch-single ../images/*.qoi


