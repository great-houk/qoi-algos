#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=test_params.%j.out

cd "$SLURM_SUBMIT_DIR"
# Prefer g++-14 when available (sbatch job); otherwise fall back to system g++.
if command -v g++-14 >/dev/null 2>&1; then
	export CXX=g++-14
else
	export CXX=g++
fi

# Configurable list of CPU counts to sweep. Adjust as needed.
CPUS_LIST=(1 2 4 8 16 32)

# Path to the CSV output (will be created/updated by test_params)
OUT_CSV="test_params.csv"

# Number of runs to average per configuration
NUM_RUNS=5

# Images directory
IMAGES_DIR="images"

# Build the binary if needed
make bin/test_params

# Loop over cpu counts. We request the max CPUs in SBATCH, then set OMP_NUM_THREADS/NUM_THREADS per run.
for cpus in "${CPUS_LIST[@]}"; do
  echo "Running test_params with cpus=$cpus"
  export OMP_NUM_THREADS=$cpus
  export NUM_THREADS=$cpus
  # Run and append to the CSV. test_params will append if the CSV exists.
  ./bin/test_params $NUM_RUNS "$IMAGES_DIR" "$OUT_CSV"
done

# Move output to a job-specific file for archival
mv "$OUT_CSV" "${OUT_CSV%.csv}.job${SLURM_JOB_ID}.csv" || true

echo "Wrote ${OUT_CSV%.csv}.job${SLURM_JOB_ID}.csv"
