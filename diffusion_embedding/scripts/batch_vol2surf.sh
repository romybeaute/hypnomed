#!/bin/bash
#
#SBATCH --job-name=vol2surf_step_1_trial_1

#SBATCH --output=vol2surf_step_1_trial_1.txt

#SBATCH --error=vol2surf_step_1_trial_1.err

#SBATCH --cpus-per-task=2

#SBATCH --array=1-76

"""
Project volume to surface using FreeSurfer
"""




# where the output of fmriprep is located
FMRIPREP_ROOT='/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives/fmriprep-latest'
echo output directory: ${FMRIPREP_ROOT}

# find all DICOM directories that start with "voice"
SUBJDIRS=(`find $FMRIPREP_ROOT -mindepth 1 -maxdepth 1  -name sub* -type d`)
echo subjdirs: ${SUBJDIRS}


# SUBJ=(sub-01
# sub-02
# sub-03
# sub-04
# sub-05
# sub-06
# sub-07
# sub-08
# sub-09
# sub-10
# sub-11
# sub-12
# sub-13
# sub-14
# sub-15
# sub-16
# sub-17
# sub-18
# sub-19
# sub-20
# sub-21
# sub-22
# sub-23
# sub-24
# sub-25
# sub-26
# sub-27
# sub-28
# sub-29
# sub-30
# sub-31
# sub-32
# sub-33
# sub-34
# sub-35
# sub-36
# sub-37
# sub-38
# sub-39
# sub-40
# )



# ./x.mri_vol2surf.sh ${SUBJDIRS[$SLURM_ARRAY_TASK_ID-1]}

while read subject;
do
  ./x.mri_vol2surf.sh ${subject}
done <subject_list.txt