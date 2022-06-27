#!/bin/bash
#
#SBATCH --job-name=vol2surf_step_1_trial_1

#SBATCH --output=vol2surf_step_1_trial_1.txt

#SBATCH --error=vol2surf_step_1_trial_1.err

#SBATCH --cpus-per-task=2

#SBATCH --array=1-76

# where the output of fmriprep is located
FMRIPREP_ROOT='/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives/fmriprep-latest'

# find all DICOM directories that start with "voice"
SUBJDIRS=(`find $FMRIPREP_ROOT -mindepth 1 -maxdepth 1  -name sub* -type d`)


# ./x.mri_vol2surf_test.sh ${SUBJDIRS[$SLURM_ARRAY_TASK_ID-1]}
./x.mri_vol2surf.sh ${SUBJDIRS[$SLURM_ARRAY_TASK_ID-1]}

