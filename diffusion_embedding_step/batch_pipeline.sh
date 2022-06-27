#!/bin/bash
#

#SBATCH --job-name=pipeline_step_2_trial_1

#SBATCH --output=pipeline_step_2_trial_1.txt

#SBATCH --error=pipeline_step_2_trial_1.err

#SBATCH --cpus-per-task=10

#SBATCH --mem=20G

#SBATCH --array=1-76


python pipeline.py ${SUBJ[$SLURM_ARRAY_TASK_ID-1]}

