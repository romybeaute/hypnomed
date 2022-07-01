#!/bin/bash
#
#SBATCH --job-name=vol2surf_step_1_trial_1

#SBATCH --output=vol2surf_step_1_trial_1.txt

#SBATCH --error=vol2surf_step_1_trial_1.err

#SBATCH --cpus-per-task=3

#SBATCH --mem=20G

#SBATCH --array=1-40

"""
1. Project volume to surface using FreeSurfer
"""


# where the output of fmriprep is located
FMRIPREP_ROOT='/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives/fmriprep-latest'
echo output directory: ${FMRIPREP_ROOT}

# find all DICOM directories that start with "voice"
SUBJDIRS=(`find $FMRIPREP_ROOT -mindepth 1 -maxdepth 1  -name sub* -type d`)
echo subjdirs: ${SUBJDIRS}


while read subject;
do echo $subject;
  ./x.mri_vol2surf.sh ${subject};
done < subject_list.txt

