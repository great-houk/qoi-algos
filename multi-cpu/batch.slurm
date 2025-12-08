#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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

# Default OpenMP settings for nested parallelism (can be overridden below per-run)
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_NESTED=true

# Make sure the *parent* bin/ exists: qoi-algos/bin
mkdir -p ../bin

# Compile batch-multi into ../bin
$CXX -O3 -std=gnu++17 -fopenmp \
    -I.. -I../reference \
    batch-multi.cpp \
    qoi-mc.cpp \
    qoi-decode.cpp \
    ../reference/qoi-reference.cpp \
    -o ../bin/batch-multi

# -- Test batching across multiple outer-thread counts.
# For each outer thread count we compute an inner thread budget as
# floor(allocated_cpus / outer), and run the program with those values.

# The job requested CPUs per task (above). Use that allocation as the pool.
ALLOC_CPUS=${SLURM_CPUS_ON_NODE:-64}
echo "SLURM job id=${SLURM_JOB_ID}, allocated cpus=${ALLOC_CPUS}"

# Define the outer thread counts we want to test (adjust as desired)
OUTER_LIST=(1 2 4 8 16 32 64)

# Images to process (relative to repo root)
IMAGES_DIR=../images

for outer in "${OUTER_LIST[@]}"; do
    if [ "$outer" -le 0 ]; then
        continue
    fi
    # compute inner threads so product <= ALLOC_CPUS
    inner=$(( ALLOC_CPUS / outer ))
    if [ "$inner" -lt 1 ]; then
        inner=1
    fi

    out_log="../batch-multi-outer${outer}-inner${inner}-${SLURM_JOB_ID}.log"
    echo "Running outer=$outer inner=$inner -> log=$out_log"

    # Export per-run OMP settings
    export OMP_NUM_THREADS=${inner}
    export OMP_THREAD_LIMIT=${inner}

    # Run the benchmark. We pass --outer and --inner to the binary so it can
    # coordinate nested parallelism accordingly. The binary prints timing and throughput.
    ../bin/batch-multi --outer ${outer} --inner ${inner} ${IMAGES_DIR}/*.qoi > "${out_log}" 2>&1

    echo "Completed outer=${outer} inner=${inner} (log: ${out_log})"
    sleep 1
done

echo "All runs finished. Logs are in the parent directory (../) named batch-multi-outer*-inner*-${SLURM_JOB_ID}.log"
