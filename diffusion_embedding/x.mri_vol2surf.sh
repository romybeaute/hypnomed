#!/bin/bash

####### Should be launched from batch_vol2surf.sh ####################

# Load Freesurfer functions
export FREESURFER_HOME='/mnt/data/sebastien/packages/freesurfer'
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#export subject from batch_vol2surf.sh
subject=${1}
#Set the path to your FMRIPREP output
FMRIPREP_ROOT='/mnt/data/sebastien/LONGIMED/raw_data/BIDS/derivatives/fmriprep-latest'
#Set the path to your output
output=/mnt/data/sebastien/LONGIMED/embedding/vol2surf_derivatives

for ses in ses-001 ses-002 ses-003;do
mkdir -p ${output}/${subject}/${ses}

  #get the functional scans of your subject
  for scan in task-rest task-fa task-om;do

    for hemi in lh rh;do


    # To run from MRI space:
    mri_vol2surf --mov $FMRIPREP_ROOT/$subject/${ses}/func/${subject}_${ses}_${scan}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz \
      --mni152reg \
      --projfrac-avg 0.2 0.8 0.1 \
      --trgsubject fsaverage5 \
      --interp nearest \
      --hemi ${hemi} \
      --surf-fwhm 6.0 --cortex --noreshape \
      --o ${output}/${subject}/${ses}/${subject}_${ses}_${scan}.fsa5.${hemi}.mgz

    done

  done

done
