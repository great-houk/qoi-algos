#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=0
#SBATCH --output=bench.output

cd $SLURM_SUBMIT_DIR
make
./bin/bench 1 images_small