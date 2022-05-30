#!/bin/bash
#SBATCH --job-name=run_heudiconv
#SBATCH --output=run_heudiconv.txt
#SBATCH --error=run_heudiconv.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

set -eu

# where we want to output the data
OUTDIR='/mnt/data/romy/hypnomed/MRI_raw/BIDS/'
# where to find the heurtic.py file for heudiconv
HEURISTIC_PATH='/mnt/data/romy/hypnomed/MRI_raw/BIDS/code/heuristic.py'
# where the DICOMs are located
DCMROOT='/mnt/data/romy/hypnomed/MRI_raw/DICOM'
# receive all directories, and index them per job array
SUBJDIRS=(${@:1})
SUBJDIR=${SUBJDIRS[${SLURM_ARRAY_TASK_ID}]}

SESSDIRS=(`find $DCMROOT -mindepth 3 -maxdepth 3  -name ses* -type d`)

echo Submitted directory: ${SUBJDIR:42}

IMG="/mnt/data/romy/singularity_images/heudiconv_0.5.4.sif"

SESSION=(001)

for SESSION_ID in ${SESSION[@]} ; do
    CMD="singularity run -B ${DCMROOT}:/base:ro -B ${OUTDIR}:/output -e ${IMG} -d /base/sub-{subject}/ses-{session}/*/*.dcm -o /output -f /output/code/heuristic.py -c dcm2niix -b --overwrite -s ${SUBJDIR:42} -ss $SESSION_ID"
    printf "Command:\n${CMD}\n"
    ${CMD}
    echo "Successful process"
done 