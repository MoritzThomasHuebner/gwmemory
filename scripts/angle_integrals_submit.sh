#!/usr/bin/env bash
#
#SBATCH --job-name=angles
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de
#
#SBATCH --array=0-17
ARGS=(-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8)
srun python angle_integrals.py ${ARGS[$SLURM_ARRAY_TASK_ID]}