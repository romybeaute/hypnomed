#!/bin/bash
#

#SBATCH --job-name=pipeline_step_2_trial_1

#SBATCH --output=pipeline_step_2_trial_1.txt

#SBATCH --error=pipeline_step_2_trial_1.err

#SBATCH --cpus-per-task=10

#SBATCH --mem=20G

#SBATCH --array=1-76

"""
2. Correlate RS data across sessions and embed

NB : This command will call load_fs.py to read in the data. This is the most time consuming step, but the script has some fancy numpy optimizations to speed it up dramatically.
"""


echo Directory pipeline : ${SUBJ[$SLURM_ARRAY_TASK_ID-1]}

python pipeline.py ${SUBJ[$SLURM_ARRAY_TASK_ID-1]}


# while read subject;
# do
#   python pipeline.py ${subject}
# done <subject_list.txt