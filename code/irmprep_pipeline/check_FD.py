#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:28:41 2022

Check Framewise Displacements, notably if >15% volumes is >0.5mm
Files come from fmriprep folder
(see Power et all., NeuroImage, 2012)
@author: sebastien
@adapted:romy
"""

import os
import numpy as np
import pandas as pd
import time

class parameters:
    pass

outcome_path = '/home/romy.beaute/projects/hypnomed/code/irmprep_pipeline/infos'

p = parameters()
# p.sub_path = '/media/sebastien/LaCie/ERC_Antoine/fmriprep_preprocessing_step/recon_all_success/fmriprep'
# p.sub_path = 'G:\ERC_Antoine/fmriprep_preprocessing_step/recon_all_success/fmriprep'
p.sub_path = '/crnldata/eduwell/meditation/HYPNOMED/BIDS/derivatives/fmriprep-latest'

p.data_path = 'ses-001/func'
# p.states_name = ['openmonitoring','compassion','restingstate']
p.states_name = ['rs_run-1','rs_run-2','rs_run-3']




subj_dir = [ name for name in os.listdir(p.sub_path) if os.path.isdir(os.path.join(p.sub_path, name)) ]
subj_dir = [ name for name in subj_dir if 'sub' in name ]


start_time = time.time()

txt=open(outcome_path+'/frame_displacements_checked.txt','w')
txt.write(70*'-')
txt.write('\n Check Framewise Displacements, notably if >15% volumes is >0.5mm\n Files come from fmriprep folder\n (see Power et all., NeuroImage, 2012)\n')
txt.write(70*'-')
txt.write('\n')

for subj in subj_dir:
    for state in p.states_name:
        # data_name = f'{subj}_ses-1_task-{state}_desc-confounds_regressors.tsv'
        try:
            data_name = f'{subj}_ses-001_task-{state}_desc-confounds_timeseries.tsv'
            df = pd.read_csv(os.path.join(p.sub_path,subj,p.data_path,data_name),sep='\t',
                            usecols=['framewise_displacement'])
            df = df.drop([0])
            if np.any(np.mean(df>0.5,axis=0)>0.15):
                # print(subj,state,'(mean {})'.format(np.mean(df>0.5,axis=0)))
                print('- {} for state {} (mean FD = {})\n'.format(subj,state,np.mean(df>0.5,axis=0)[0]))
                txt.write('- {} for state {} (mean FD = {})\n'.format(subj,state,np.mean(df>0.5,axis=0)[0]))
        except:
            print('- {} for state {} (state not found or processed)\n'.format(subj,state))
            txt.write('- {} for state {} (state not found or processed)\n'.format(subj,state))
            pass
print('Time elapsed: ',time.time() - start_time)
txt.close()
