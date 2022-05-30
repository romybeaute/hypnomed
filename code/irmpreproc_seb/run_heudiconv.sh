#!/bin/bash
#SBATCH --job-name=run_heudiconv
#SBATCH --output=run_heudiconv.txt
#SBATCH --error=run_heudiconv.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

set -eu

# where we want to output the data
OUTDIR='/mnt/data/sebastien/LONGIMED/BIDS/'
# where to find the heurtic.py file for heudiconv
HEURISTIC_PATH='/mnt/data/sebastien/LONGIMED/BIDS/code/heuristic.py'
# where the DICOMs are located
DCMROOT='/mnt/data/sebastien/LONGIMED/RAW_DATA/DICOM'
# receive all directories, and index them per job array
SUBJDIRS=(${@:1})
SUBJDIR=${SUBJDIRS[${SLURM_ARRAY_TASK_ID}]}

SESSDIRS=(`find $DCMROOT -mindepth 3 -maxdepth 3  -name ses* -type d`)

echo Submitted directory: ${SUBJDIR:48}

IMG="/mnt/data/sebastien/singularity_images/heudiconv.sif"

SESSION=(001 002 003)

for SESSION_ID in ${SESSION[@]} ; do
    CMD="singularity run -B ${DCMROOT}:/base:ro -B ${OUTDIR}:/output -e ${IMG} -d /base/sub-{subject}/ses-{session}/*/*.dcm -o /output -f /output/code/heuristic.py -c dcm2niix -b --overwrite -s ${SUBJDIR:48} -ss $SESSION_ID"
    printf "Command:\n${CMD}\n"
    ${CMD}
    echo "Successful process"
done 