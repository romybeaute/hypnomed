#!/bin/bash
#SBATCH --job-name=run_heudiconv
#SBATCH --output=run_heudiconv.txt
#SBATCH --error=run_heudiconv.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

set -eu

# where the DICOMs are located
DCMROOT='/mnt/data/romy/hypnomed/MRI_raw/DICOM'

# find all DICOM directories that start with "voice"
SUBJDIRS=(`find $DCMROOT -mindepth 1 -maxdepth 1  -name sub* -type d`)

# submit to another script as a job array on SLURM
sbatch --array=0-`expr ${#SUBJDIRS[@]} - 1` run_heudiconv.sh ${SUBJDIRS[@]}
