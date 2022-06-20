#!/bin/bash
#
#SBATCH -J fmriprep
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

# Outputs ----------------------------------
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=romybeaute.univ@gmail.com
#SBATCH --mail-type=ALL
# ------------------------------------------

export BIDS_DIR="$STUDY/MRI_raw/BIDS"
DERIVS_DIR="derivatives/fmriprep-latest"

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=$STUDY/git/code/pre.cache/templateflow
FMRIPREP_HOST_CACHE=$STUDY/git/code/.cache/fmriprep

mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
export SINGULARITYENV_FS_LICENSE=/work/license.txt

# Make sure fmriprep image is defined in the container.
export IMG='/mnt/data/romy/singularity_images/fmriprep-21.0.2.simg'


# Prepare derivatives folder
mkdir -p ${BIDS_DIR}/${DERIVS_DIR}

# This trick will help you reuse freesurfer results across pipelines and fMRIPrep versions
mkdir -p ${BIDS_DIR}/derivatives/freesurfer-6.0.1
if [ ! -d ${BIDS_DIR}/${DERIVS_DIR}/freesurfer ]; then
    ln -s ${BIDS_DIR}/derivatives/freesurfer-6.0.1 ${BIDS_DIR}/${DERIVS_DIR}/freesurfer
fi


# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"
SINGULARITY_CMD="singularity run --cleanenv -B $BIDS_DIR:/data -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} -B $STUDY:/work $IMG"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Remove IsRunning files from FreeSurfer
# find ${BIDS_DIR}/derivatives/freesurfer-6.0.1/sub-$subject/ -name "*IsRunning*" -type f -delete

# Compose the command line
cmd="${SINGULARITY_CMD} /data /data/${DERIVS_DIR} participant --participant-label $subject -w /work/ -vv --fs-license-file $SINGULARITYENV_FS_LICENSE --omp-nthreads 6 --nthreads 8 --mem_mb 20000 --cifti-output --skip_bids_validation --notrack --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 --use-aroma"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode