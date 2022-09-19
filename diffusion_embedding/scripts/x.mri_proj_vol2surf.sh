#!/bin/bash

#SBATCH --output=x.mri_proj_vol2surf.txt

#SBATCH --error=x.mri_proj_vol2surf.err

####### Should be launched from batch_vol2surf.sh ####################

# Load Freesurfer functions
export FREESURFER_HOME='/mnt/data/romy/packages/freesurfer'
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#export subject from batch_vol2surf.sh
# export SUBJDIRS
# echo ${SUBJDIRS}


# subject=${1}

# subject=${SUBJDIRS##*/}
# echo ${subject}

#Set the path to your FMRIPREP output
FMRIPREP_ROOT='/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives/fmriprep-latest'
#Set the path to your output
dir=/mnt/data/romy/hypnomed/git/diffusion_embedding/vol2surf_derivatives

# for subject in sub-01;do
while read subject;do
# for ses in ses-001 ses-002 ses-003;do
  for ses in ses-001;do
  mkdir -p ${dir}/${subject}/${ses}

    #get the functional scans of your subject

    for scan in task-rs_run-1 task-rs_run-2 task-rs_run-3;do

      for hemi in lh rh;do


      # To run from MRI space:
      # mri_vol2surf --mov $FMRIPREP_ROOT/$subject/${ses}/func/${subject}_${ses}_${scan}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz \
      mri_vol2surf --mov $FMRIPREP_ROOT/${subject}/${ses}/func/${subject}_${ses}_${scan}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz \
        --mni152reg \
        --projfrac-avg 0.2 0.8 0.1 \
        --trgsubject fsaverage5 \
        --interp nearest \
        --hemi ${hemi} \
        --surf-fwhm 6.0 --cortex --noreshape \
        --o ${dir}/${subject}/${ses}/${subject}_${ses}_${scan}.fsa5.${hemi}.mgz

      done

    done

  done
done < subject_list.txt
