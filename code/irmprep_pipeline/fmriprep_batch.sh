#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=fmriprep.txt
#SBATCH --error=fmriprep.err
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=10G
#SBATCH --array=1-60

BIDSDIR='/mnt/data/sebastien/LONGIMED/BIDS/'
OUTDIR='/mnt/data/sebastien/LONGIMED/recon_all'
IMG='/mnt/data/sebastien/singularity_images/fmriprep.sif'

SUBJ=(`find $BIDSDIR -mindepth 1 -maxdepth 1  -name sub* -type d`)

echo ${SUBJ[$SLURM_ARRAY_TASK_ID-1]:48} 
singularity run --cleanenv -B /mnt:/mnt $IMG $BIDSDIR $OUTDIR participant\
      --participant-label ${SUBJ[$SLURM_ARRAY_TASK_ID-1]:48} --low-mem --stop-on-first-crash \
      --use-aroma --cifti-output --notrack \
      --output-spaces fsaverage5 --fs-license-file /mnt/data/sebastien/packages/license.txt \
      --omp-nthreads 9 --nthreads 9 --mem_mb 10000


