export STUDY=/mnt/data/sebastien/LONGIMED/
export PARTCIPANTS_INFO=$STUDY/raw_data/BIDS/participants.tsv

sbatch --array=1-$(( $( wc -l $PARTCIPANTS_INFO | cut -f1 -d' ' ) - 1 )) fmriprep_slurm.sh