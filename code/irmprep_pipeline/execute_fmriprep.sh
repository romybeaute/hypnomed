export STUDY=/mnt/data/romy/hypnomed/
export PARTCIPANTS_INFO=$STUDY/MRI_raw/BIDS/participants.tsv

sbatch --array=1-$(( $( wc -l $PARTCIPANTS_INFO | cut -f1 -d' ' ) - 1 )) fmriprep_slurm.sh