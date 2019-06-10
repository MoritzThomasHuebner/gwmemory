#!/usr/bin/env bash

ARGS={-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8}
#
#SBATCH --job-name=real_event
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de
#
#SBATCH --array=0-16

srun python production_real_event.py $1 64 $ARGS{${SLURM_ARRAY_TASK_ID}}