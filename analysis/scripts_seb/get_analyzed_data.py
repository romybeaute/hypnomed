#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:42:44 2021

@author: sebastien
"""

#get averaged and pvalues from data on the cluster
#data are too heavy to be downloaded

import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon 
from nilearn import surface

class p:
    pass

class d:
    pass

cluster = True
if cluster:
    # # careful on cluster, use path from root !!!!
    p.data_path = '/mnt/data/sebastien/data'
    p.results_path = '/mnt/data/sebastien/results/b2b_emb_quest_searchlight'
else:
    p.data_path = 'data'
    p.results_path = 'results/emb_decoding'
    
p.repetitions_nb = 100 #the number of parallel nodes used
p.quest_names = ['attentional','constructive','deconstructive']


yeo_r = surface.load_surf_data('/mnt/data/sebastien/recon_all_success/freesurfer/label/rh.Yeo2011_7Networks_N1000.annot')
yeo_l = surface.load_surf_data('/mnt/data/sebastien/recon_all_success/freesurfer/label/lh.Yeo2011_7Networks_N1000.annot')
d.yeo = np.hstack([yeo_l,yeo_r])


for job_id in range(p.repetitions_nb):    
    if job_id == 0:
        d.df = pd.read_pickle(os.path.join(p.results_path,f'decoding_results_{job_id+1}.pkl'),'zip')
    else:
        current_df = pd.read_pickle(os.path.join(p.results_path,f'decoding_results_{job_id+1}.pkl'),'zip')
        for row_id in range(d.df.shape[0]):
            d.df.iloc[row_id]['factors'] = np.concatenate((d.df.iloc[row_id]['factors'],current_df.iloc[row_id]['factors']))
            d.df.iloc[row_id]['factors_random'] = np.concatenate((d.df.iloc[row_id]['factors_random'],current_df.iloc[row_id]['factors_random']))

for analysis_id in range(d.df['factors'].shape[0]):
    current_df = d.df.iloc[analysis_id]
    p_values = np.ones((current_df['factors'].shape[1],current_df['factors'].shape[2]))
    for quest_id in range(current_df['factors'].shape[2]):
        for voxel_id in range(current_df['factors'].shape[1]):
            if d.yeo[voxel_id]!=0:
                _,p_values[voxel_id,quest_id] = wilcoxon(current_df['factors'][:,voxel_id,quest_id],
                                      current_df['factors_random'][:,voxel_id,quest_id],
                                      alternative='greater')
            
    d.df['factors'][analysis_id] = np.mean(d.df['factors'][analysis_id],0)
    d.df['factors_random'][analysis_id] = p_values
    
d.df = d.df.rename(columns={'p_values' : 'factors_random'})
d.df.to_pickle(os.path.join(p.results_path,'analyzed_data'))